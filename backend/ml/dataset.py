"""Dataset loading, validation, deterministic splitting, and label encoding.

Provides leak-free multi-label stratified group splitting to ensure
exact and near-duplicate texts never cross train/validation/test boundaries.
"""

from __future__ import annotations

import json
import random
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from . import config


def normalize_text(text: str) -> str:
    """Normalize text consistently for duplicate detection.

    Applies Unicode NFKC normalization, collapses consecutive whitespace,
    and strips leading/trailing spaces before lowercasing.
    Does not strip meaningful technical terms, punctuation, or numbers.
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


STOPWORDS: set[str] = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "to", "of", "in", "on", "at", "by", "for", "with", "about", "against",
    "between", "into", "through", "during", "before", "after", "above",
    "below", "from", "up", "down", "out", "off", "over", "under",
    "again", "further", "then", "once", "here", "there", "when", "where",
    "why", "how", "all", "any", "both", "each", "few", "more", "most",
    "other", "some", "such", "no", "nor", "not", "only", "own", "same",
    "so", "than", "too", "very", "can", "will", "just", "should", "now",
    "refers", "refer", "defined", "means", "term", "describes", "used",
}


def get_content_tokens(text: str) -> set[str]:
    """Extract content word tokens for near-duplicate Jaccard similarity."""
    norm = normalize_text(text)
    tokens = set(re.findall(r"\w+", norm))
    content = {tok for tok in tokens if tok not in STOPWORDS and len(tok) > 1}
    return content or tokens


def group_examples(
    examples: list[dict[str, Any]],
    jaccard_threshold: float = 0.85,
    min_token_len: int = 4,
) -> list[list[dict[str, Any]]]:
    """Group exact duplicates and likely near-duplicates into atomic clusters.

    Uses deterministic Union-Find:
    1. Exact normalized text matches are unioned.
    2. Candidate pairs sharing infrequent token shingles with Jaccard >= jaccard_threshold
       are unioned into near-duplicate groups.
    """
    if not examples:
        return []

    n = len(examples)
    token_sets = [get_content_tokens(ex["text"]) for ex in examples]

    # Inverted index for candidate selection
    shingles: dict[str, list[int]] = defaultdict(list)
    for i, tokens in enumerate(token_sets):
        for tok in tokens:
            if len(tok) >= min_token_len:
                shingles[tok].append(i)

    # Union-Find data structure
    parent = list(range(n))

    def find(i: int) -> int:
        path = []
        while parent[i] != i:
            path.append(i)
            i = parent[i]
        for p in path:
            parent[p] = i
        return i

    def union(i: int, j: int) -> None:
        root_i, root_j = find(i), find(j)
        if root_i != root_j:
            # Deterministic union: smaller index becomes parent
            if root_i < root_j:
                parent[root_j] = root_i
            else:
                parent[root_i] = root_j

    # 1. Exact match on normalized text
    exact_map: dict[str, list[int]] = defaultdict(list)
    for i, ex in enumerate(examples):
        exact_map[normalize_text(ex["text"])].append(i)

    for idxs in exact_map.values():
        for k in range(1, len(idxs)):
            union(idxs[0], idxs[k])

    # 2. Near-duplicates via inverted-index candidate pairs
    cand_pairs: dict[tuple[int, int], int] = defaultdict(int)
    for tok, idxs in shingles.items():
        if len(idxs) <= 30:  # Skip high-frequency common words
            for a_idx in range(len(idxs)):
                a = idxs[a_idx]
                for b_idx in range(a_idx + 1, len(idxs)):
                    b = idxs[b_idx]
                    pair = (a, b) if a < b else (b, a)
                    cand_pairs[pair] += 1

    for (a, b), shared_count in cand_pairs.items():
        if shared_count >= 2:
            sa, sb = token_sets[a], token_sets[b]
            inter = len(sa & sb)
            u = len(sa | sb)
            if u > 0 and (inter / u) >= jaccard_threshold:
                union(a, b)

    clusters: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for i in range(n):
        clusters[find(i)].append(examples[i])

    # Sort groups deterministically by minimum sample index
    sorted_cluster_roots = sorted(clusters.keys())
    return [clusters[root] for root in sorted_cluster_roots]


def deduplicate_exact_records(
    examples: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Deduplicate exact normalized text records, merging labels.

    Returns the deduplicated examples list and a summary dict.
    """
    groups_by_norm: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for ex in examples:
        groups_by_norm[normalize_text(ex["text"])].append(ex)

    deduped: list[dict[str, Any]] = []
    duplicate_count = 0

    for norm_text, items in groups_by_norm.items():
        if len(items) > 1:
            duplicate_count += len(items) - 1
            # Merge labels deterministically
            merged_labels = list(dict.fromkeys(
                label for item in items for label in item.get("labels", [])
            ))
            # Keep representative text (longest or first)
            rep_item = max(items, key=lambda x: len(x["text"]))
            deduped.append({
                "text": rep_item["text"].strip(),
                "labels": merged_labels,
            })
        else:
            deduped.append({
                "text": items[0]["text"].strip(),
                "labels": list(dict.fromkeys(items[0].get("labels", []))),
            })

    stats = {
        "original_count": len(examples),
        "deduplicated_count": len(deduped),
        "exact_duplicates_removed": duplicate_count,
    }
    return deduped, stats


def load_examples(
    path: Path = config.TRAIN_FILE,
    deduplicate: bool = False,
) -> list[dict[str, Any]]:
    """Load JSONL dataset examples with strict schema validation."""
    examples = []

    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue

            item = json.loads(line)

            if not isinstance(item.get("text"), str) or not item["text"].strip():
                raise ValueError(f"Line {line_no}: text must be non-empty")

            labels = item.get("labels")

            if not isinstance(labels, list) or not labels:
                raise ValueError(
                    f"Line {line_no}: labels must be a non-empty list"
                )

            unknown = set(labels) - set(config.LABELS)

            if unknown:
                raise ValueError(
                    f"Line {line_no}: unknown labels: {sorted(unknown)}"
                )

            examples.append(
                {
                    "text": item["text"].strip(),
                    "labels": list(dict.fromkeys(labels)),
                }
            )

    if len(examples) < 30:
        raise ValueError(
            "Starter dataset is too small; expected at least 30 examples."
        )

    if deduplicate:
        examples, _ = deduplicate_exact_records(examples)

    return examples


def multilabel_vector(labels: list[str]) -> list[float]:
    """Convert label names into a multi-hot binary float vector."""
    label_set = set(labels)
    return [
        1.0 if label in label_set else 0.0
        for label in config.LABELS
    ]


def split_examples(
    examples: list[dict[str, Any]],
    seed: int = config.SEED,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    deduplicate: bool = True,
    group_near_duplicates: bool = True,
) -> dict[str, list[dict[str, Any]]]:
    """Create a reproducible multilabel-stratified disjoint train/val/test split.

    Guarantees:
    1. Exact duplicate texts never cross split boundaries. If deduplicate=True,
       exact duplicate records are merged to single unique canonical examples.
    2. Near-duplicate groups never cross split boundaries.
    3. Multi-label group stratification balances label distribution.
    4. Rare classes (< 3 groups) are handled safely without crashing.
    5. Disjointness is verified before returning.
    """
    if not examples:
        return {"train": [], "validation": [], "test": []}

    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-5:
        raise ValueError("train_ratio, val_ratio, and test_ratio must sum to 1.0")
    if train_ratio <= 0 or val_ratio <= 0 or test_ratio <= 0:
        raise ValueError("All split ratios must be positive")

    items = list(examples)
    if deduplicate:
        items, _ = deduplicate_exact_records(items)

    rng = random.Random(seed)

    # Group records into atomic clusters
    if group_near_duplicates:
        groups = group_examples(items)
    else:
        # Group exact normalized matches only
        exact_map = defaultdict(list)
        for ex in items:
            exact_map[normalize_text(ex["text"])].append(ex)
        groups = list(exact_map.values())

    # Shuffle groups deterministically
    rng.shuffle(groups)

    total_records = sum(len(g) for g in groups)
    target_sizes = {
        "train": max(1, int(round(total_records * train_ratio))),
        "validation": max(1, int(round(total_records * val_ratio))),
    }
    target_sizes["test"] = total_records - target_sizes["train"] - target_sizes["validation"]
    if target_sizes["test"] < 1:
        target_sizes["validation"] = max(1, target_sizes["validation"] - 1)
        target_sizes["test"] = 1

    # Extract labels for each group
    group_labels = [
        set(label for ex in g for label in ex.get("labels", []))
        for g in groups
    ]

    # Frequency of each label across groups
    label_group_counts = Counter()
    for gl in group_labels:
        for lab in gl:
            label_group_counts[lab] += 1

    # Order labels by rarity (least frequent first), breaking ties by name
    sorted_labels = [
        lab for lab, _ in sorted(label_group_counts.items(), key=lambda x: (x[1], x[0]))
    ]

    assigned_group_indices: set[int] = set()
    splits_groups: dict[str, list[list[dict[str, Any]]]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    split_record_counts = {"train": 0, "validation": 0, "test": 0}

    # STEP 1: Guarantee label coverage where groups permit
    for lab in sorted_labels:
        avail = [
            i for i, gl in enumerate(group_labels)
            if lab in gl and i not in assigned_group_indices
        ]
        if not avail:
            continue

        if len(avail) >= 3:
            splits_to_cover = ["train", "validation", "test"]
        elif len(avail) == 2:
            splits_to_cover = ["train", "validation"]
        else:
            splits_to_cover = ["train"]

        for s in splits_to_cover:
            candidates = [i for i in avail if i not in assigned_group_indices]
            if not candidates:
                break
            chosen = candidates[0]
            assigned_group_indices.add(chosen)
            splits_groups[s].append(groups[chosen])
            split_record_counts[s] += len(groups[chosen])

    # STEP 2: Distribute remaining groups according to capacity
    remaining = [i for i in range(len(groups)) if i not in assigned_group_indices]
    rng.shuffle(remaining)

    for i in remaining:
        g = groups[i]
        # Choose split furthest below its target size
        chosen_split = max(
            ["train", "validation", "test"],
            key=lambda s: target_sizes[s] - split_record_counts[s],
        )
        splits_groups[chosen_split].append(g)
        split_record_counts[chosen_split] += len(g)

    # Flatten and shuffle records within each split
    result: dict[str, list[dict[str, Any]]] = {}
    for s in ("train", "validation", "test"):
        rows = [ex for g in splits_groups[s] for ex in g]
        rng.shuffle(rows)
        result[s] = rows

    # STEP 3: Verify strict disjointness by normalized text
    train_norm = {normalize_text(x["text"]) for x in result["train"]}
    val_norm = {normalize_text(x["text"]) for x in result["validation"]}
    test_norm = {normalize_text(x["text"]) for x in result["test"]}

    if train_norm & val_norm:
        raise RuntimeError(
            f"Leakage detected: {len(train_norm & val_norm)} normalized texts overlap between train and validation."
        )
    if train_norm & test_norm:
        raise RuntimeError(
            f"Leakage detected: {len(train_norm & test_norm)} normalized texts overlap between train and test."
        )
    if val_norm & test_norm:
        raise RuntimeError(
            f"Leakage detected: {len(val_norm & test_norm)} normalized texts overlap between validation and test."
        )

    return result


def validate_splits(
    splits: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    """Validate that splits are disjoint and report comprehensive statistics."""
    for key in ("train", "validation", "test"):
        if key not in splits:
            raise ValueError(f"Missing required split '{key}'")

    train_rows = splits["train"]
    val_rows = splits["validation"]
    test_rows = splits["test"]

    train_exact = {x["text"] for x in train_rows}
    val_exact = {x["text"] for x in val_rows}
    test_exact = {x["text"] for x in test_rows}

    train_norm = {normalize_text(x["text"]) for x in train_rows}
    val_norm = {normalize_text(x["text"]) for x in val_rows}
    test_norm = {normalize_text(x["text"]) for x in test_rows}

    exact_overlap = {
        "train_val": len(train_exact & val_exact),
        "train_test": len(train_exact & test_exact),
        "val_test": len(val_exact & test_exact),
    }

    norm_overlap = {
        "train_val": len(train_norm & val_norm),
        "train_test": len(train_norm & test_norm),
        "val_test": len(val_norm & test_norm),
    }

    is_leak_free = all(v == 0 for v in exact_overlap.values()) and all(v == 0 for v in norm_overlap.values())
    if not is_leak_free:
        raise ValueError(f"Splits contain data leakage! Exact overlap: {exact_overlap}, Normalized overlap: {norm_overlap}")

    total_samples = len(train_rows) + len(val_rows) + len(test_rows)
    all_rows = train_rows + val_rows + test_rows

    report = {
        "is_leak_free": is_leak_free,
        "total_samples": total_samples,
        "split_sizes": {
            "train": len(train_rows),
            "validation": len(val_rows),
            "test": len(test_rows),
        },
        "exact_overlap": exact_overlap,
        "normalized_overlap": norm_overlap,
        "class_distribution": {
            "overall": label_distribution(all_rows),
            "train": label_distribution(train_rows),
            "validation": label_distribution(val_rows),
            "test": label_distribution(test_rows),
        },
    }
    return report


def save_splits(
    splits: dict[str, list[dict[str, Any]]],
    path: Path = config.SPLITS_FILE,
) -> None:
    """Save verified splits to JSON format."""
    validate_splits(splits)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(splits, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def label_distribution(
    examples: list[dict[str, Any]],
) -> dict[str, int]:
    """Count occurrences of each configured label in the example list."""
    return {
        label: sum(label in item.get("labels", []) for item in examples)
        for label in config.LABELS
    }


class TopicDataset(Dataset):
    """PyTorch Dataset for multi-label topic classification."""

    def __init__(
        self,
        examples: list[dict[str, Any]],
        tokenizer,
        max_length: int = config.MAX_LENGTH,
    ):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(
        self,
        index: int,
    ) -> dict[str, torch.Tensor]:
        item = self.examples[index]
        encoded = self.tokenizer(
            item["text"],
            truncation=True,
            max_length=self.max_length,
            padding=False,
        )
        encoded["labels"] = multilabel_vector(item["labels"])
        return {
            key: torch.tensor(value)
            for key, value in encoded.items()
        }