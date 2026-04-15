import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from cmeaa_xai.config import ModelConfig
from cmeaa_xai.data.iu_xray import IUXrayCollator, IUXrayDataset
from cmeaa_xai.models.report_generator import ExplainableReportGenerator
from scripts.smoke_test import DummyTokenizer
from transformers import T5Config, T5ForConditionalGeneration, ViTConfig, ViTImageProcessor, ViTModel


class SmokeTest(unittest.TestCase):
    def test_forward_and_generate(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            images_dir = root / "images"
            images_dir.mkdir()
            for idx in range(2):
                Image.new("RGB", (224, 224), color=(idx * 50, idx * 50, idx * 50)).save(images_dir / f"{idx}.png")
            csv_path = root / "data.csv"
            csv_path.write_text(
                "study_id,image_path,report,obs_No Finding,obs_Enlarged Cardiomediastinum,obs_Cardiomegaly,obs_Lung Lesion,obs_Lung Opacity,obs_Edema,obs_Consolidation,obs_Pneumonia,obs_Atelectasis,obs_Pneumothorax,obs_Pleural Effusion,obs_Pleural Other,obs_Fracture,obs_Support Devices\n"
                f"0,{images_dir / '0.png'},no acute disease,1,0,0,0,0,0,0,0,0,0,0,0,0,0\n"
                f"1,{images_dir / '1.png'},atelectasis with support device,0,0,0,0,0,0,0,0,1,0,0,0,0,1\n"
            )
            phrase_bank = root / "phrase_bank.json"
            phrase_bank.write_text(json.dumps({"Atelectasis": ["atelectasis"], "Support Devices": ["support device"]}))
            tokenizer = DummyTokenizer()
            image_processor = ViTImageProcessor(size={"height": 224, "width": 224})
            vit = ViTModel(ViTConfig(hidden_size=64, num_hidden_layers=1, num_attention_heads=4, intermediate_size=128, image_size=224, patch_size=16))
            t5 = T5ForConditionalGeneration(T5Config(vocab_size=64, d_model=64, d_ff=128, num_layers=1, num_decoder_layers=1, num_heads=4, decoder_start_token_id=0, pad_token_id=0, eos_token_id=1))
            model = ExplainableReportGenerator(ModelConfig(observation_dim=64, alignment_dim=64), vision_encoder=vit, text_decoder=t5, tokenizer=tokenizer, image_processor=image_processor)
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
            self.assertTrue(outputs.loss.isfinite())
            seq, obs_logits, obs_attn = model.generate(
                pixel_values=batch.pixel_values,
                input_ids=batch.input_ids,
                attention_mask=batch.attention_mask,
                observation_phrase_ids=batch.observation_phrase_ids,
                observation_phrase_mask=batch.observation_phrase_mask,
                max_new_tokens=8,
                num_beams=1,
            )
            self.assertEqual(seq.shape[0], 2)
            self.assertEqual(tuple(obs_logits.shape), (2, 14))
            self.assertEqual(obs_attn.ndim, 3)


if __name__ == "__main__":
    unittest.main()
