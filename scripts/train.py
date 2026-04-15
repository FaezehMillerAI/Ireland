import argparse
from pathlib import Path

import torch
from torch.cuda.amp import GradScaler
from torch.optim import AdamW
from torch.utils.data import DataLoader

from cmeaa_xai.config import ExperimentConfig
from cmeaa_xai.data.iu_xray import IUXrayCollator, IUXrayDataset
from cmeaa_xai.models.report_generator import ExplainableReportGenerator
from cmeaa_xai.training.trainer import save_checkpoint, save_predictions, set_seed, train_one_epoch, validate
from cmeaa_xai.eval.metrics import save_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = ExperimentConfig.from_yaml(args.config)
    set_seed(cfg.train.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and cfg.train.device == "cuda" else "cpu")

    model = ExplainableReportGenerator(cfg.model)
    model.to(device)
    collator = IUXrayCollator(model.tokenizer)
    train_set = IUXrayDataset(cfg.data.train_csv, model.image_processor, model.tokenizer, str(Path(cfg.data.processed_dir) / "phrase_bank.json"), cfg.data.max_target_length)
    val_set = IUXrayDataset(cfg.data.val_csv, model.image_processor, model.tokenizer, str(Path(cfg.data.processed_dir) / "phrase_bank.json"), cfg.data.max_target_length)
    train_loader = DataLoader(train_set, batch_size=cfg.train.batch_size, shuffle=True, num_workers=cfg.data.num_workers, collate_fn=collator)
    val_loader = DataLoader(val_set, batch_size=cfg.eval.batch_size, shuffle=False, num_workers=cfg.data.num_workers, collate_fn=collator)

    optimizer = AdamW((p for p in model.parameters() if p.requires_grad), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    scaler = GradScaler(enabled=cfg.train.fp16 and device.type == "cuda")
    out_dir = Path(cfg.train.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    best_f1 = -1.0
    for epoch in range(1, cfg.train.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, scaler, device, cfg.train)
        val_metrics, predictions, references, obs_scores, attn_maps, image_paths = validate(
            model,
            val_loader,
            device,
            model.tokenizer,
            num_beams=cfg.train.num_beams,
            max_new_tokens=cfg.data.max_target_length,
        )
        print({"epoch": epoch, **train_metrics, **val_metrics})
        save_metrics({f"train_{k}": v for k, v in train_metrics.items()} | {f"val_{k}": v for k, v in val_metrics.items()}, str(out_dir / f"metrics_epoch_{epoch}.json"))
        save_predictions(str(out_dir / f"predictions_epoch_{epoch}.json"), predictions, references, obs_scores, attn_maps, image_paths)
        save_checkpoint(model, optimizer, str(out_dir / f"checkpoint_epoch_{epoch}.pt"))
        if val_metrics["clinical_f1"] > best_f1:
            best_f1 = val_metrics["clinical_f1"]
            save_checkpoint(model, optimizer, str(out_dir / "best.pt"))


if __name__ == "__main__":
    main()
