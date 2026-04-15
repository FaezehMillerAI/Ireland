# Explainable CmEAA for IU-Xray

This repository implements a Colab-friendly radiology report generation pipeline inspired by the COLING 2025 `CmEAA` paper, with a small explainability-focused extension:

- The paper's idea is preserved through cross-modal feature enhancement and low-dimensional neural mutual-information alignment.
- The new piece is a `PrototypeObservationExplainer` that predicts the 14 CheXpert-style observations and exposes patch-level evidence maps for each observation.
- This makes the model easier to inspect than the original method while staying close to its overall design.

## Proposed Methodological Extension

The original paper improves report generation with:

- `CFE`: observation-conditioned enhancement of image tokens.
- `NMIA`: low-dimensional cross-modal alignment using neural mutual information.

This implementation adds:

- `Sparse Prototype Observation Explainer (SPOX)`: each observation has a trainable prototype; image patches attend to those prototypes and produce:
  - observation logits,
  - observation-to-patch attention maps,
  - prototype evidence vectors.
- `Prototype consistency loss`: keeps learned evidence close to report-derived observation phrase representations.

This is a slight methodological change rather than a rewrite of the paper: the model still uses enhanced visual tokens and MI alignment, but now it also learns explicit observation evidence that can be exported and evaluated.

## IU-Xray Kaggle Layout Supported

The data preparation script supports the Kaggle layout you shared:

- `indiana_reports.csv`
- `images/images_normalized/*.png`

It builds:

- `train.csv`
- `val.csv`
- `test.csv`
- `phrase_bank.json`

## Quick Start in Colab

```bash
git clone <your-repo-url>
cd Ireland
pip install -r requirements.txt
pip install -e .
```

```python
import kagglehub
dataset_path = kagglehub.dataset_download("raddar/chest-xrays-indiana-university")
print(dataset_path)
```

```bash
python scripts/prepare_iu_xray.py \
  --dataset-root "$DATASET_PATH" \
  --output-dir /content/iu_xray_processed
```

Or let the script download it directly:

```bash
python scripts/prepare_iu_xray.py \
  --download-kaggle \
  --output-dir /content/iu_xray_processed
```

Update [configs/iu_xray_colab.yaml](/Users/fs525/Desktop/Ireland/configs/iu_xray_colab.yaml) if your `dataset_path` differs.

```bash
python scripts/train.py --config configs/iu_xray_colab.yaml
python scripts/evaluate.py \
  --config configs/iu_xray_colab.yaml \
  --checkpoint /content/iu_xray_runs/explainable_cmeaa/best.pt
```

## Outputs

Training/evaluation writes:

- checkpoints,
- per-epoch metrics,
- prediction JSON files,
- observation scores,
- observation attention maps.

These let you inspect both text quality and model evidence.

## Evaluation Pipeline

The evaluation code reports:

- text metrics: BLEU, ROUGE-L,
- clinical efficacy proxy: micro precision, recall, F1 over 14 observations using a report lexicon labeler,
- explainability diagnostics:
  - mean/std of observation confidence,
  - attention entropy,
  - attention peak mass.

The clinical/explainability part is intentionally modular, so you can later swap in a stronger labeler like CheXbert.

## Smoke Test

A lightweight end-to-end smoke test is included and does not need dataset downloads:

```bash
python scripts/smoke_test.py
python -m unittest tests.test_smoke
```

## Important Notes

- This code is designed to be runnable end to end in Colab, not to claim exact reproduction of the paper's reported scores.
- The paper uses Swin + LLaMA2 + GPT-4-assisted observation extraction. This implementation uses a more accessible setup for open Colab execution:
  - ViT image encoder,
  - T5 text decoder,
  - rule-based report observation bootstrapping,
  - explainable observation prototypes.
- If you want, the next natural extension is to replace the weak report labeler with CheXbert and add report-level evidence visualizations.
