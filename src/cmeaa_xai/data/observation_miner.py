import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from cmeaa_xai.constants import OBSERVATIONS
from cmeaa_xai.utils.text import normalize_text, ngrams, tokenize_for_pmi


OBSERVATION_PATTERNS = {
    "No Finding": [r"\bno acute cardiopulmonary abnormalit", r"\bno active disease", r"\bno acute disease"],
    "Enlarged Cardiomediastinum": [r"widened mediastinum", r"enlarged cardiomediastinal"],
    "Cardiomegaly": [r"cardiomegaly", r"enlarged cardiac silhouette"],
    "Lung Lesion": [r"lung lesion", r"pulmonary nodule", r"mass lesion"],
    "Lung Opacity": [r"opacity", r"infiltrate", r"airspace disease"],
    "Edema": [r"edema", r"pulmonary vascular congestion"],
    "Consolidation": [r"consolidation"],
    "Pneumonia": [r"pneumonia"],
    "Atelectasis": [r"atelecta"],
    "Pneumothorax": [r"pneumothorax"],
    "Pleural Effusion": [r"pleural effusion", r"blunting of the costophrenic"],
    "Pleural Other": [r"pleural thickening", r"pleural plaque"],
    "Fracture": [r"fracture", r"osseous deformity"],
    "Support Devices": [r"catheter", r"pacer", r"line tip", r"tube", r"device"],
}


def label_report_with_lexicon(report: str) -> Dict[str, int]:
    report = normalize_text(report)
    labels = {}
    for obs in OBSERVATIONS:
        matched = any(re.search(pattern, report) for pattern in OBSERVATION_PATTERNS.get(obs, []))
        labels[obs] = int(matched)
    if labels["No Finding"] == 1:
        for obs in OBSERVATIONS:
            if obs != "No Finding":
                labels[obs] = 0
    return labels


def mine_observation_ngrams(
    reports: Sequence[str],
    observation_rows: Sequence[Dict[str, int]],
    ngram_range: Iterable[int] = (1, 2, 3),
    top_k: int = 12,
    min_count: int = 2,
) -> Dict[str, List[str]]:
    total_docs = len(reports)
    obs_doc_count = Counter()
    ngram_doc_count = Counter()
    joint_count = defaultdict(int)

    for report, labels in zip(reports, observation_rows):
        tokens = tokenize_for_pmi(report)
        seen = set()
        for n in ngram_range:
            seen.update(ngrams(tokens, n))
        for item in seen:
            ngram_doc_count[item] += 1
        for obs, value in labels.items():
            if value:
                obs_doc_count[obs] += 1
                for item in seen:
                    joint_count[(obs, item)] += 1

    results = {}
    for obs in OBSERVATIONS:
        scores = []
        for gram, gram_count in ngram_doc_count.items():
            co = joint_count[(obs, gram)]
            if co < min_count:
                continue
            p_og = co / total_docs
            p_o = max(obs_doc_count[obs] / total_docs, 1e-8)
            p_g = max(gram_count / total_docs, 1e-8)
            score = math.log(p_og / (p_o * p_g) + 1e-8)
            scores.append((score, gram))
        scores.sort(reverse=True)
        results[obs] = [gram for _, gram in scores[:top_k]]
    return results


def save_phrase_bank(bank: Dict[str, List[str]], path: str) -> None:
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    path_obj.write_text(json.dumps(bank, indent=2))


def load_phrase_bank(path: str) -> Dict[str, List[str]]:
    return json.loads(Path(path).read_text())
