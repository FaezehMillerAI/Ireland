import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from cmeaa_xai.config import ExperimentConfig
from cmeaa_xai.data.iu_xray import IUXrayCollator, IUXrayDataset
from cmeaa_xai.eval.metrics import save_metrics
from cmeaa_xai.models.report_generator import ExplainableReportGenerator
from cmeaa_xai.training.trainer import validate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    cfg = ExperimentConfig.from_yaml(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() and cfg.train.device == "cuda" else "cpu")
    model = ExplainableReportGenerator(cfg.model)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()

    test_set = IUXrayDataset(cfg.data.test_csv, model.image_processor, model.tokenizer, str(Path(cfg.data.processed_dir) / "phrase_bank.json"), cfg.data.max_target_length)
    loader = DataLoader(test_set, batch_size=cfg.eval.batch_size, shuffle=False, num_workers=cfg.data.num_workers, collate_fn=IUXrayCollator(model.tokenizer))
    metrics, predictions, references, obs_scores, attn_maps, image_paths = validate(
        model,
        loader,
        device,
        model.tokenizer,
        num_beams=cfg.train.num_beams,
        max_new_tokens=cfg.data.max_target_length,
    )
    out_dir = Path(cfg.train.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_metrics(metrics, str(out_dir / "test_metrics.json"))
    rows = []
    for pred, ref, image_path, scores in zip(predictions, references, image_paths, obs_scores):
        rows.append({"image_path": image_path, "prediction": pred, "reference": ref, "observation_scores": scores})
    (out_dir / "test_predictions.json").write_text(json.dumps(rows, indent=2))
    print(metrics)


if __name__ == "__main__":
    main()
