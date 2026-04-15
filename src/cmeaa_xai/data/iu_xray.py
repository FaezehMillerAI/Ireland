import random
import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from cmeaa_xai.constants import DEFAULT_INSTRUCTION, OBSERVATIONS
from cmeaa_xai.data.observation_miner import label_report_with_lexicon, load_phrase_bank


def _text_from_element(root: ET.Element, tag: str) -> str:
    values = []
    for node in root.iter():
        if node.tag.lower().endswith(tag.lower()) and node.text:
            values.append(node.text.strip())
    return " ".join(values).strip()


def build_iu_xray_dataframe(dataset_root: str) -> pd.DataFrame:
    root = Path(dataset_root)
    report_dir_candidates = list(root.rglob("*.xml"))
    rows: List[Dict[str, str]] = []
    for report_path in report_dir_candidates:
        try:
            tree = ET.parse(report_path)
        except ET.ParseError:
            continue
        xml_root = tree.getroot()
        findings = _text_from_element(xml_root, "findings")
        impression = _text_from_element(xml_root, "impression")
        report = f"{findings} {impression}".strip()
        if not report:
            continue
        image_paths = []
        for node in xml_root.iter():
            attrib_id = node.attrib.get("id")
            if attrib_id and ("image" in node.tag.lower() or "parentimage" in node.tag.lower()):
                matches = list(root.rglob(f"{attrib_id}*"))
                matches = [m for m in matches if m.suffix.lower() in {".png", ".jpg", ".jpeg"}]
                if matches:
                    image_paths.append(str(matches[0]))
        if not image_paths:
            continue
        labels = label_report_with_lexicon(report)
        rows.append(
            {
                "study_id": report_path.stem,
                "image_path": image_paths[0],
                "report": report,
                **{f"obs_{k}": v for k, v in labels.items()},
            }
        )
    df = pd.DataFrame(rows).drop_duplicates("study_id")
    return df


def build_iu_xray_dataframe_from_kaggle_layout(dataset_root: str) -> pd.DataFrame:
    root = Path(dataset_root)
    reports_path = root / "indiana_reports.csv"
    if not reports_path.exists():
        raise FileNotFoundError(f"Could not find {reports_path}")
    reports_df = pd.read_csv(reports_path).copy()
    for col in ["findings", "impression", "indication", "comparison"]:
        if col not in reports_df:
            reports_df[col] = ""
        reports_df[col] = reports_df[col].fillna("")

    reports_df = reports_df[
        (reports_df["findings"].str.len() > 0) | (reports_df["impression"].str.len() > 0)
    ].copy()
    images_dir = root / "images" / "images_normalized"
    if not images_dir.exists():
        images_dir = root / "images_normalized"
    pattern = re.compile(r"(\d+)_IM-\d+-\d+\.dcm\.png")
    uid_to_images: Dict[int, List[str]] = defaultdict(list)
    for image_path in images_dir.iterdir():
        match = pattern.match(image_path.name)
        if match:
            uid_to_images[int(match.group(1))].append(str(image_path))

    reports_df["image_files"] = reports_df["uid"].apply(lambda uid: uid_to_images.get(int(uid), []))
    reports_df["num_images"] = reports_df["image_files"].apply(len)
    reports_df = reports_df[reports_df["num_images"] > 0].copy()
    reports_df["report"] = (
        reports_df["findings"].str.strip() + " " + reports_df["impression"].str.strip()
    ).str.strip()

    label_rows = reports_df["report"].apply(label_report_with_lexicon)
    for obs in OBSERVATIONS:
        reports_df[f"obs_{obs}"] = label_rows.apply(lambda row: row[obs])

    return pd.DataFrame(
        {
            "study_id": reports_df["uid"].astype(str),
            "image_path": reports_df["image_files"].apply(lambda items: items[0]),
            "report": reports_df["report"],
            **{f"obs_{obs}": reports_df[f"obs_{obs}"] for obs in OBSERVATIONS},
        }
    )


def make_splits(df: pd.DataFrame, seed: int = 42) -> Dict[str, pd.DataFrame]:
    study_ids = list(df["study_id"].unique())
    rng = random.Random(seed)
    rng.shuffle(study_ids)
    n = len(study_ids)
    train_cut = int(0.7 * n)
    val_cut = int(0.8 * n)
    split_map = {
        "train": set(study_ids[:train_cut]),
        "val": set(study_ids[train_cut:val_cut]),
        "test": set(study_ids[val_cut:]),
    }
    return {name: df[df["study_id"].isin(ids)].reset_index(drop=True) for name, ids in split_map.items()}


def save_splits(splits: Dict[str, pd.DataFrame], out_dir: str) -> Dict[str, str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    results = {}
    for split_name, frame in splits.items():
        path = out / f"{split_name}.csv"
        frame.to_csv(path, index=False)
        results[split_name] = str(path)
    return results


@dataclass
class BatchEncoding:
    pixel_values: torch.Tensor
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor
    observation_labels: torch.Tensor
    observation_phrase_ids: torch.Tensor
    observation_phrase_mask: torch.Tensor
    report_text: List[str]
    image_path: List[str]


class IUXrayDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        image_processor,
        tokenizer,
        phrase_bank_path: str,
        max_target_length: int = 128,
        instruction: str = DEFAULT_INSTRUCTION,
    ) -> None:
        self.df = pd.read_csv(csv_path)
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.instruction = instruction
        self.max_target_length = max_target_length
        self.phrase_bank = load_phrase_bank(phrase_bank_path)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        row = self.df.iloc[idx]
        image = Image.open(row["image_path"]).convert("RGB")
        pixel_values = self.image_processor(images=image, return_tensors="pt")["pixel_values"].squeeze(0)
        prompt = self.instruction
        prompt_tokens = self.tokenizer(prompt, truncation=True, return_tensors="pt")
        label_tokens = self.tokenizer(
            row["report"],
            truncation=True,
            max_length=self.max_target_length,
            return_tensors="pt",
        )
        obs_labels = torch.tensor([int(row[f"obs_{obs}"]) for obs in OBSERVATIONS], dtype=torch.float32)
        phrases = [" ; ".join(self.phrase_bank.get(obs, [])[:6]) or obs.lower() for obs in OBSERVATIONS]
        phrase_tokens = self.tokenizer(
            phrases,
            truncation=True,
            padding="max_length",
            max_length=16,
            return_tensors="pt",
        )
        return {
            "pixel_values": pixel_values,
            "input_ids": prompt_tokens["input_ids"].squeeze(0),
            "attention_mask": prompt_tokens["attention_mask"].squeeze(0),
            "labels": label_tokens["input_ids"].squeeze(0),
            "observation_labels": obs_labels,
            "observation_phrase_ids": phrase_tokens["input_ids"],
            "observation_phrase_mask": phrase_tokens["attention_mask"],
            "report_text": row["report"],
            "image_path": row["image_path"],
        }


class IUXrayCollator:
    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer

    def __call__(self, batch: List[Dict[str, object]]) -> BatchEncoding:
        labels = [item["labels"] for item in batch]
        prompt_ids = [item["input_ids"] for item in batch]
        prompt_mask = [item["attention_mask"] for item in batch]
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels[labels == self.tokenizer.pad_token_id] = -100
        prompt_ids = torch.nn.utils.rnn.pad_sequence(prompt_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        prompt_mask = torch.nn.utils.rnn.pad_sequence(prompt_mask, batch_first=True, padding_value=0)
        return BatchEncoding(
            pixel_values=torch.stack([item["pixel_values"] for item in batch]),
            input_ids=prompt_ids,
            attention_mask=prompt_mask,
            labels=labels,
            observation_labels=torch.stack([item["observation_labels"] for item in batch]),
            observation_phrase_ids=torch.stack([item["observation_phrase_ids"] for item in batch]),
            observation_phrase_mask=torch.stack([item["observation_phrase_mask"] for item in batch]),
            report_text=[str(item["report_text"]) for item in batch],
            image_path=[str(item["image_path"]) for item in batch],
        )
