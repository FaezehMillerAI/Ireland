import json
import tempfile
from pathlib import Path

import torch
from PIL import Image
from transformers import T5Config, T5ForConditionalGeneration, ViTConfig, ViTImageProcessor, ViTModel

from cmeaa_xai.config import ModelConfig
from cmeaa_xai.data.iu_xray import IUXrayCollator, IUXrayDataset
from cmeaa_xai.models.report_generator import ExplainableReportGenerator


class DummyTokenizer:
    pad_token_id = 0

    def __init__(self):
        vocab = ["<pad>", "</s>"] + [chr(i) for i in range(97, 123)] + [" ", ".", ";", ":"]
        self.stoi = {token: idx for idx, token in enumerate(vocab)}
        self.itos = {idx: token for token, idx in self.stoi.items()}

    def _encode_text(self, text: str, max_length: int | None = None):
        ids = [self.stoi.get(ch, 2) for ch in text.lower()]
        if max_length is not None:
            ids = ids[:max_length]
        return ids or [1]

    def __call__(self, texts, truncation=True, padding=False, max_length=None, return_tensors="pt"):
        if isinstance(texts, str):
            texts = [texts]
        encoded = [self._encode_text(text, max_length) for text in texts]
        max_len = max(len(item) for item in encoded)
        ids = []
        mask = []
        for item in encoded:
            pad_len = max_len - len(item)
            ids.append(item + [0] * pad_len)
            mask.append([1] * len(item) + [0] * pad_len)
        return {"input_ids": torch.tensor(ids), "attention_mask": torch.tensor(mask)}

    def batch_decode(self, sequences, skip_special_tokens=True):
        outputs = []
        for seq in sequences:
            chars = [self.itos.get(int(idx), "") for idx in seq if int(idx) != 0]
            outputs.append("".join(chars).strip())
        return outputs


def main():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        images_dir = root / "images"
        images_dir.mkdir()
        for idx in range(2):
            Image.new("RGB", (224, 224), color=(idx * 40, idx * 40, idx * 40)).save(images_dir / f"{idx}.png")
        csv_path = root / "train.csv"
        csv_path.write_text(
            "study_id,image_path,report,obs_No Finding,obs_Enlarged Cardiomediastinum,obs_Cardiomegaly,obs_Lung Lesion,obs_Lung Opacity,obs_Edema,obs_Consolidation,obs_Pneumonia,obs_Atelectasis,obs_Pneumothorax,obs_Pleural Effusion,obs_Pleural Other,obs_Fracture,obs_Support Devices\n"
            f"0,{images_dir / '0.png'},no acute disease,1,0,0,0,0,0,0,0,0,0,0,0,0,0\n"
            f"1,{images_dir / '1.png'},mild atelectasis with support device,0,0,0,0,0,0,0,0,1,0,0,0,0,1\n"
        )
        phrase_bank = root / "phrase_bank.json"
        phrase_bank.write_text(json.dumps({"Atelectasis": ["mild atelectasis"], "Support Devices": ["support device"]}))

        tokenizer = DummyTokenizer()
        image_processor = ViTImageProcessor(size={"height": 224, "width": 224})
        vit = ViTModel(ViTConfig(hidden_size=64, num_hidden_layers=1, num_attention_heads=4, intermediate_size=128, image_size=224, patch_size=16))
        t5 = T5ForConditionalGeneration(T5Config(vocab_size=64, d_model=64, d_ff=128, num_layers=1, num_decoder_layers=1, num_heads=4, decoder_start_token_id=0, pad_token_id=0, eos_token_id=1))
        model_cfg = ModelConfig(observation_dim=64, alignment_dim=64, num_observations=14)
        model = ExplainableReportGenerator(model_cfg, vision_encoder=vit, text_decoder=t5, tokenizer=tokenizer, image_processor=image_processor)

        dataset = IUXrayDataset(str(csv_path), image_processor, tokenizer, str(phrase_bank), max_target_length=32)
        batch = IUXrayCollator(tokenizer)([dataset[0], dataset[1]])
        outputs = model(
            pixel_values=batch.pixel_values,
            input_ids=batch.input_ids,
            attention_mask=batch.attention_mask,
            labels=batch.labels,
            observation_labels=batch.observation_labels,
            observation_phrase_ids=batch.observation_phrase_ids,
            observation_phrase_mask=batch.observation_phrase_mask,
        )
        assert torch.isfinite(outputs.loss)
        sequences, obs_logits, obs_attention = model.generate(
            pixel_values=batch.pixel_values,
            input_ids=batch.input_ids,
            attention_mask=batch.attention_mask,
            observation_phrase_ids=batch.observation_phrase_ids,
            observation_phrase_mask=batch.observation_phrase_mask,
            max_new_tokens=8,
            num_beams=1,
        )
        assert sequences.shape[0] == 2
        assert obs_logits.shape == (2, 14)
        assert obs_attention.ndim == 3
        print("Smoke test passed.")


if __name__ == "__main__":
    main()
