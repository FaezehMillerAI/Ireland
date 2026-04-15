import re
from collections import Counter
from typing import Iterable, List


def normalize_text(text: str) -> str:
    text = text.lower().replace("\n", " ")
    text = re.sub(r"[^a-z0-9\s.,;:/-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def ngrams(tokens: List[str], n: int) -> Iterable[str]:
    for i in range(len(tokens) - n + 1):
        yield " ".join(tokens[i : i + n])


def tokenize_for_pmi(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", normalize_text(text))


def term_frequency(texts: Iterable[str]) -> Counter:
    counter = Counter()
    for text in texts:
        counter.update(tokenize_for_pmi(text))
    return counter
