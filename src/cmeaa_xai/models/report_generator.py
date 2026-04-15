from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoImageProcessor, AutoModel, AutoTokenizer, T5ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput

from cmeaa_xai.config import ModelConfig
from cmeaa_xai.models.explainable_adapter import ExplainableCmEAAAdapter


@dataclass
class ForwardOutput:
    loss: torch.Tensor
    lm_loss: torch.Tensor
    observation_loss: torch.Tensor
    nmi_loss: torch.Tensor
    prototype_loss: torch.Tensor
    logits: torch.Tensor
    observation_logits: torch.Tensor
    observation_attention: torch.Tensor


class ExplainableReportGenerator(nn.Module):
    def __init__(
        self,
        model_config: ModelConfig,
        vision_encoder: Optional[nn.Module] = None,
        text_decoder: Optional[T5ForConditionalGeneration] = None,
        tokenizer=None,
        image_processor=None,
    ) -> None:
        super().__init__()
        self.model_config = model_config
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(model_config.text_model_name)
        self.image_processor = image_processor or AutoImageProcessor.from_pretrained(model_config.vision_model_name)
        self.vision_encoder = vision_encoder or AutoModel.from_pretrained(model_config.vision_model_name)
        self.text_decoder = text_decoder or T5ForConditionalGeneration.from_pretrained(model_config.text_model_name)
        hidden_size = getattr(self.vision_encoder.config, "hidden_size")
        text_hidden = self.text_decoder.config.d_model
        self.vision_to_text = nn.Linear(hidden_size, text_hidden)
        self.text_embed_proj = nn.Identity()
        self.adapter = ExplainableCmEAAAdapter(
            hidden_size=text_hidden,
            num_observations=model_config.num_observations,
            observation_dim=model_config.observation_dim,
            alignment_dim=model_config.alignment_dim,
            dropout=model_config.dropout,
        )
        if model_config.freeze_vision_encoder:
            for param in self.vision_encoder.parameters():
                param.requires_grad = False
        if model_config.freeze_text_decoder:
            for param in self.text_decoder.parameters():
                param.requires_grad = False

    @classmethod
    def from_random_configs(cls, model_config: ModelConfig):
        from transformers import T5Config, ViTConfig, ViTModel

        vit_config = ViTConfig(hidden_size=128, num_hidden_layers=2, num_attention_heads=4, intermediate_size=256)
        t5_config = T5Config(
            vocab_size=32128,
            d_model=128,
            d_ff=256,
            num_layers=2,
            num_decoder_layers=2,
            num_heads=4,
            decoder_start_token_id=0,
            eos_token_id=1,
            pad_token_id=0,
        )
        vision_encoder = ViTModel(vit_config)
        text_decoder = T5ForConditionalGeneration(t5_config)
        tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
        image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
        return cls(model_config, vision_encoder=vision_encoder, text_decoder=text_decoder, tokenizer=tokenizer, image_processor=image_processor)

    def encode_phrases(self, observation_phrase_ids: torch.Tensor, observation_phrase_mask: torch.Tensor) -> torch.Tensor:
        embed = self.text_decoder.get_input_embeddings()(observation_phrase_ids)
        mask = observation_phrase_mask.unsqueeze(-1)
        return (embed * mask).sum(dim=2) / mask.sum(dim=2).clamp_min(1.0)

    def encode_labels_as_text_state(self, labels: torch.Tensor) -> torch.Tensor:
        safe_labels = labels.masked_fill(labels == -100, self.tokenizer.pad_token_id)
        embed = self.text_decoder.get_input_embeddings()(safe_labels)
        return embed

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        observation_labels: torch.Tensor,
        observation_phrase_ids: torch.Tensor,
        observation_phrase_mask: torch.Tensor,
        lambda_nmi: float = 0.1,
        lambda_obs: float = 0.5,
        lambda_proto: float = 0.1,
    ) -> ForwardOutput:
        vision_outputs = self.vision_encoder(pixel_values=pixel_values)
        visual_tokens = self.vision_to_text(vision_outputs.last_hidden_state)
        phrase_states = self.encode_phrases(observation_phrase_ids, observation_phrase_mask)
        text_states = self.encode_labels_as_text_state(labels)
        adapter_outputs = self.adapter(visual_tokens, phrase_states, text_states, observation_labels)

        prompt_embeds = self.text_decoder.get_input_embeddings()(input_ids)
        encoder_hidden = torch.cat([adapter_outputs.fused_tokens, prompt_embeds], dim=1)
        encoder_mask = torch.cat(
            [
                torch.ones(adapter_outputs.fused_tokens.shape[:2], device=pixel_values.device, dtype=attention_mask.dtype),
                attention_mask,
            ],
            dim=1,
        )
        outputs = self.text_decoder(
            encoder_outputs=BaseModelOutput(last_hidden_state=encoder_hidden),
            attention_mask=encoder_mask,
            labels=labels,
            return_dict=True,
        )
        observation_loss = F.binary_cross_entropy_with_logits(adapter_outputs.observation_logits, observation_labels)
        total_loss = outputs.loss + lambda_obs * observation_loss + lambda_nmi * adapter_outputs.nmi_loss + lambda_proto * adapter_outputs.prototype_loss
        return ForwardOutput(
            loss=total_loss,
            lm_loss=outputs.loss,
            observation_loss=observation_loss,
            nmi_loss=adapter_outputs.nmi_loss,
            prototype_loss=adapter_outputs.prototype_loss,
            logits=outputs.logits,
            observation_logits=adapter_outputs.observation_logits,
            observation_attention=adapter_outputs.observation_attention,
        )

    @torch.no_grad()
    def generate(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        observation_phrase_ids: torch.Tensor,
        observation_phrase_mask: torch.Tensor,
        max_new_tokens: int = 128,
        num_beams: int = 4,
    ):
        vision_outputs = self.vision_encoder(pixel_values=pixel_values)
        visual_tokens = self.vision_to_text(vision_outputs.last_hidden_state)
        phrase_states = self.encode_phrases(observation_phrase_ids, observation_phrase_mask)
        adapter_outputs = self.adapter(visual_tokens, phrase_states, None, None)
        prompt_embeds = self.text_decoder.get_input_embeddings()(input_ids)
        encoder_hidden = torch.cat([adapter_outputs.fused_tokens, prompt_embeds], dim=1)
        encoder_mask = torch.cat(
            [
                torch.ones(adapter_outputs.fused_tokens.shape[:2], device=pixel_values.device, dtype=attention_mask.dtype),
                attention_mask,
            ],
            dim=1,
        )
        sequences = self.text_decoder.generate(
            encoder_outputs=BaseModelOutput(last_hidden_state=encoder_hidden),
            attention_mask=encoder_mask,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
        )
        return sequences, adapter_outputs.observation_logits, adapter_outputs.observation_attention
