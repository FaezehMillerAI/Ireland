import json
import random
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from cmeaa_xai.eval.metrics import compute_report_metrics


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def move_batch_to_device(batch, device: torch.device):
    return {
        "pixel_values": batch.pixel_values.to(device),
        "input_ids": batch.input_ids.to(device),
        "attention_mask": batch.attention_mask.to(device),
        "labels": batch.labels.to(device),
        "observation_labels": batch.observation_labels.to(device),
        "observation_phrase_ids": batch.observation_phrase_ids.to(device),
        "observation_phrase_mask": batch.observation_phrase_mask.to(device),
        "report_text": batch.report_text,
        "image_path": batch.image_path,
    }


def train_one_epoch(model, loader: DataLoader, optimizer, scaler, device, cfg) -> Dict[str, float]:
    model.train()
    running = {"loss": 0.0, "lm_loss": 0.0, "obs_loss": 0.0, "nmi_loss": 0.0}
    optimizer.zero_grad(set_to_none=True)
    autocast_device = "cuda" if device.type == "cuda" else "cpu"
    for step, batch in enumerate(tqdm(loader, desc="train", leave=False), start=1):
        batch_dict = move_batch_to_device(batch, device)
        with torch.autocast(device_type=autocast_device, enabled=cfg.fp16 and device.type == "cuda"):
            outputs = model(
                **{k: v for k, v in batch_dict.items() if k not in {"report_text", "image_path"}},
                lambda_nmi=cfg.lambda_nmi,
                lambda_obs=cfg.lambda_obs,
                lambda_proto=cfg.lambda_proto,
            )
            loss = outputs.loss / cfg.grad_accum_steps
        scaler.scale(loss).backward()
        if step % cfg.grad_accum_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        running["loss"] += outputs.loss.item()
        running["lm_loss"] += outputs.lm_loss.item()
        running["obs_loss"] += outputs.observation_loss.item()
        running["nmi_loss"] += outputs.nmi_loss.item()
    num_steps = max(len(loader), 1)
    return {key: value / num_steps for key, value in running.items()}


@torch.no_grad()
def predict_reports(model, loader: DataLoader, device, tokenizer, num_beams: int = 4, max_new_tokens: int = 128):
    model.eval()
    predictions: List[str] = []
    references: List[str] = []
    observation_scores = []
    attention_maps = []
    image_paths = []
    for batch in tqdm(loader, desc="predict", leave=False):
        batch_dict = move_batch_to_device(batch, device)
        sequences, obs_logits, obs_attention = model.generate(
            pixel_values=batch_dict["pixel_values"],
            input_ids=batch_dict["input_ids"],
            attention_mask=batch_dict["attention_mask"],
            observation_phrase_ids=batch_dict["observation_phrase_ids"],
            observation_phrase_mask=batch_dict["observation_phrase_mask"],
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
        )
        texts = tokenizer.batch_decode(sequences, skip_special_tokens=True)
        predictions.extend(texts)
        references.extend(batch.report_text)
        observation_scores.extend(torch.sigmoid(obs_logits).cpu().tolist())
        attention_maps.extend(obs_attention.cpu().tolist())
        image_paths.extend(batch.image_path)
    return predictions, references, observation_scores, attention_maps, image_paths


def save_checkpoint(model, optimizer, path: str) -> None:
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict()}, path_obj)


def save_predictions(path: str, predictions, references, observation_scores, attention_maps, image_paths) -> None:
    rows = []
    for pred, ref, obs, attn, image_path in zip(predictions, references, observation_scores, attention_maps, image_paths):
        rows.append(
            {
                "image_path": image_path,
                "prediction": pred,
                "reference": ref,
                "observation_scores": obs,
                "observation_attention": attn,
            }
        )
    Path(path).write_text(json.dumps(rows, indent=2))


def validate(model, loader: DataLoader, device, tokenizer, num_beams: int = 4, max_new_tokens: int = 128):
    predictions, references, observation_scores, attention_maps, image_paths = predict_reports(
        model,
        loader,
        device,
        tokenizer,
        num_beams=num_beams,
        max_new_tokens=max_new_tokens,
    )
    metrics = compute_report_metrics(predictions, references, observation_scores)
    return metrics, predictions, references, observation_scores, attention_maps, image_paths
