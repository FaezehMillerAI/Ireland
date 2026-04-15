from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class AdapterOutput:
    fused_tokens: torch.Tensor
    enhanced_tokens: torch.Tensor
    aligned_tokens: torch.Tensor
    observation_logits: torch.Tensor
    observation_attention: torch.Tensor
    nmi_loss: torch.Tensor
    prototype_loss: torch.Tensor
    sparse_loss: torch.Tensor
    explainability: Dict[str, torch.Tensor]


class CrossModalFeatureEnhancer(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.visual_norm = nn.LayerNorm(hidden_size)
        self.phrase_norm = nn.LayerNorm(hidden_size)
        self.visual_self_attn = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.LayerNorm(hidden_size * 2),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, visual_tokens: torch.Tensor, phrase_tokens: torch.Tensor) -> torch.Tensor:
        visual_tokens = self.visual_norm(visual_tokens)
        phrase_tokens = self.phrase_norm(phrase_tokens)
        visual_sa, _ = self.visual_self_attn(visual_tokens, visual_tokens, visual_tokens)
        cross, _ = self.cross_attn(visual_sa, phrase_tokens, phrase_tokens)
        return self.ffn(torch.cat([visual_sa, cross], dim=-1))


class NeuralMutualInformationAligner(nn.Module):
    def __init__(self, hidden_size: int, alignment_dim: int):
        super().__init__()
        self.image_proj = nn.Linear(hidden_size, alignment_dim)
        self.text_proj = nn.Linear(hidden_size, alignment_dim)
        self.back_proj = nn.Linear(alignment_dim, hidden_size)
        self.scorer = nn.Sequential(
            nn.Linear(alignment_dim * 2, alignment_dim),
            nn.GELU(),
            nn.Linear(alignment_dim, 1),
        )

    def dv_bound(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        joint = self.scorer(torch.cat([a, b], dim=-1))
        shuffled = b[torch.randperm(b.size(0), device=b.device)]
        marginal = self.scorer(torch.cat([a, shuffled], dim=-1))
        return joint.mean() - torch.log(torch.exp(marginal).mean() + 1e-8)

    def infer(self, visual_tokens: torch.Tensor) -> torch.Tensor:
        pooled_visual = visual_tokens.mean(dim=1)
        image_aligned = self.image_proj(pooled_visual)
        return self.back_proj(image_aligned).unsqueeze(1).expand(-1, visual_tokens.size(1), -1)

    def forward(self, visual_tokens: torch.Tensor, text_states: Optional[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        if text_states is None:
            return self.infer(visual_tokens), torch.tensor(0.0, device=visual_tokens.device)
        pooled_visual = visual_tokens.mean(dim=1)
        pooled_text = text_states.mean(dim=1)
        image_aligned = self.image_proj(pooled_visual)
        text_aligned = self.text_proj(pooled_text)
        fused = 0.5 * (image_aligned + text_aligned)
        nmi_loss = -(self.dv_bound(image_aligned, fused) + self.dv_bound(text_aligned, fused))
        aligned_token = self.back_proj(fused).unsqueeze(1).expand(-1, visual_tokens.size(1), -1)
        return aligned_token, nmi_loss


class PrototypeObservationExplainer(nn.Module):
    def __init__(self, hidden_size: int, num_observations: int, observation_dim: int):
        super().__init__()
        self.visual_proj = nn.Linear(hidden_size, observation_dim)
        self.phrase_proj = nn.Linear(hidden_size, observation_dim)
        self.prototype_bank = nn.Parameter(torch.randn(num_observations, observation_dim) * 0.02)
        self.classifier = nn.Linear(observation_dim, 1)

    def forward(
        self,
        visual_tokens: torch.Tensor,
        phrase_states: torch.Tensor,
        observation_labels: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        visual = F.normalize(self.visual_proj(visual_tokens), dim=-1)
        phrase = F.normalize(self.phrase_proj(phrase_states), dim=-1)
        prototypes = F.normalize(self.prototype_bank + phrase, dim=-1)
        attention_logits = torch.einsum("bnh,boh->bon", visual, prototypes)
        attention = attention_logits.softmax(dim=-1)
        evidence = torch.einsum("bon,bnh->boh", attention, visual)
        logits = self.classifier(evidence).squeeze(-1)
        sparse_loss = -(attention * (attention.clamp_min(1e-8).log())).sum(dim=-1).mean()
        prototype_loss = torch.tensor(0.0, device=visual_tokens.device)
        if observation_labels is not None:
            target_proto = observation_labels.unsqueeze(-1) * phrase
            prototype_loss = F.mse_loss(evidence, target_proto)
        return logits, attention, evidence, prototype_loss + 0.01 * sparse_loss


class ExplainableCmEAAAdapter(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_observations: int,
        observation_dim: int,
        alignment_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.cfe = CrossModalFeatureEnhancer(hidden_size, dropout=dropout)
        self.nmia = NeuralMutualInformationAligner(hidden_size, alignment_dim)
        self.explainer = PrototypeObservationExplainer(hidden_size, num_observations, observation_dim)
        self.weight_gen = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, 2),
        )
        self.output_norm = nn.LayerNorm(hidden_size)

    def forward(
        self,
        visual_tokens: torch.Tensor,
        phrase_states: torch.Tensor,
        text_states: Optional[torch.Tensor],
        observation_labels: Optional[torch.Tensor] = None,
    ) -> AdapterOutput:
        enhanced_tokens = self.cfe(visual_tokens, phrase_states)
        aligned_tokens, nmi_loss = self.nmia(visual_tokens, text_states)
        pooled = visual_tokens.mean(dim=1)
        weights = torch.softmax(self.weight_gen(pooled), dim=-1)
        fused_tokens = (
            weights[:, 0].view(-1, 1, 1) * enhanced_tokens
            + weights[:, 1].view(-1, 1, 1) * visual_tokens
            + aligned_tokens
        )
        obs_logits, obs_attention, evidence, aux_loss = self.explainer(visual_tokens, phrase_states, observation_labels)
        fused_tokens = self.output_norm(fused_tokens + evidence.mean(dim=1, keepdim=True))
        return AdapterOutput(
            fused_tokens=fused_tokens,
            enhanced_tokens=enhanced_tokens,
            aligned_tokens=aligned_tokens,
            observation_logits=obs_logits,
            observation_attention=obs_attention,
            nmi_loss=nmi_loss,
            prototype_loss=aux_loss,
            sparse_loss=aux_loss,
            explainability={"weights": weights, "evidence": evidence},
        )
