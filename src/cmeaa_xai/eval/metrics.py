import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from cmeaa_xai.constants import OBSERVATIONS
from cmeaa_xai.data.observation_miner import label_report_with_lexicon


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def compute_bleu_like(predictions: List[str], references: List[str]) -> Dict[str, float]:
    try:
        import evaluate

        bleu = evaluate.load("bleu")
        rouge = evaluate.load("rouge")
        bleu_result = bleu.compute(predictions=predictions, references=[[ref] for ref in references])
        rouge_result = rouge.compute(predictions=predictions, references=references)
        return {
            "bleu": float(bleu_result["bleu"]),
            "rougeL": float(rouge_result["rougeL"]),
        }
    except Exception:
        overlap = []
        for pred, ref in zip(predictions, references):
            p = set(pred.lower().split())
            r = set(ref.lower().split())
            overlap.append(_safe_div(len(p & r), len(r)))
        return {"bleu": float(np.mean(overlap)), "rougeL": float(np.mean(overlap))}


def compute_clinical_metrics(predictions: List[str], references: List[str]) -> Dict[str, float]:
    pred_labels = [label_report_with_lexicon(text) for text in predictions]
    ref_labels = [label_report_with_lexicon(text) for text in references]
    tp = fp = fn = 0
    per_obs_f1 = {}
    for obs in OBSERVATIONS:
        obs_tp = obs_fp = obs_fn = 0
        for pred, ref in zip(pred_labels, ref_labels):
            p = pred[obs]
            r = ref[obs]
            obs_tp += int(p == 1 and r == 1)
            obs_fp += int(p == 1 and r == 0)
            obs_fn += int(p == 0 and r == 1)
        tp += obs_tp
        fp += obs_fp
        fn += obs_fn
        precision = _safe_div(obs_tp, obs_tp + obs_fp)
        recall = _safe_div(obs_tp, obs_tp + obs_fn)
        per_obs_f1[f"f1_{obs.lower().replace(' ', '_')}"] = _safe_div(2 * precision * recall, precision + recall)
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    return {
        "clinical_precision": precision,
        "clinical_recall": recall,
        "clinical_f1": _safe_div(2 * precision * recall, precision + recall),
        **per_obs_f1,
    }


def compute_explainability_metrics(observation_scores: List[List[float]], attention_maps: Optional[List[List[List[float]]]] = None) -> Dict[str, float]:
    if not observation_scores:
        return {"obs_score_mean": 0.0}
    scores = np.array(observation_scores)
    metrics = {
        "obs_score_mean": float(scores.mean()),
        "obs_score_std": float(scores.std()),
    }
    if attention_maps:
        attn = np.array(attention_maps)
        entropy = -(attn * np.log(np.clip(attn, 1e-8, 1.0))).sum(axis=-1).mean()
        max_mass = attn.max(axis=-1).mean()
        metrics["explanation_entropy"] = float(entropy)
        metrics["explanation_peak_mass"] = float(max_mass)
    return metrics


def compute_report_metrics(predictions: List[str], references: List[str], observation_scores=None, attention_maps=None) -> Dict[str, float]:
    metrics = {}
    metrics.update(compute_bleu_like(predictions, references))
    metrics.update(compute_clinical_metrics(predictions, references))
    metrics.update(compute_explainability_metrics(observation_scores or [], attention_maps))
    return metrics


def save_metrics(metrics: Dict[str, float], path: str) -> None:
    Path(path).write_text(json.dumps(metrics, indent=2))
