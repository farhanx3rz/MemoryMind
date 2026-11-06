"""Evaluate the saved RECALL topic classifier on leakage-free splits."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from . import config
from .classifier import TopicClassifier
from .dataset import (
    label_distribution,
    load_examples,
    multilabel_vector,
    split_examples,
    validate_splits,
)


def evaluate_model(
    classifier: TopicClassifier | None = None,
    split_name: str = "test",
    report_output_path: Path | None = None,
) -> dict[str, Any]:
    """Run honest multi-label evaluation and return structured metrics."""
    try:
        from sklearn.metrics import (
            accuracy_score,
            classification_report,
            f1_score,
            hamming_loss,
            multilabel_confusion_matrix,
            precision_score,
            recall_score,
        )
    except ImportError as exc:
        raise SystemExit("Evaluation requires scikit-learn.") from exc

    if classifier is None:
        classifier = TopicClassifier()

    if not classifier.available:
        raise RuntimeError(
            f"Classifier checkpoint unavailable at {classifier.model_dir}."
        )

    # Load splits (from splits.json if valid and leak-free, otherwise re-split)
    splits = None
    if config.SPLITS_FILE.exists():
        try:
            saved = json.loads(config.SPLITS_FILE.read_text(encoding="utf-8"))
            val_check = validate_splits(saved)
            if val_check.get("is_leak_free"):
                splits = saved
        except Exception:
            splits = None

    if splits is None:
        examples = load_examples()
        splits = split_examples(examples, seed=config.SEED)

    overlap_audit = validate_splits(splits)
    rows = splits[split_name]
    labels_order = list(classifier.labels)

    # Batch prediction
    texts = [item["text"] for item in rows]
    probs_list = classifier.predict_chunks(texts)
    probs = np.array(probs_list)

    # Ground truth vectors
    y_true = np.array([
        [1.0 if lab in item.get("labels", []) else 0.0 for lab in labels_order]
        for item in rows
    ])

    threshold = float(classifier.threshold)
    y_pred = (probs >= threshold).astype(int)

    # Aggregate metrics
    micro_f1 = float(f1_score(y_true, y_pred, average="micro", zero_division=0))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    micro_precision = float(precision_score(y_true, y_pred, average="micro", zero_division=0))
    macro_precision = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_precision = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
    micro_recall = float(recall_score(y_true, y_pred, average="micro", zero_division=0))
    macro_recall = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_recall = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
    subset_accuracy = float(accuracy_score(y_true, y_pred))
    hamming = float(hamming_loss(y_true, y_pred))

    # Confusion matrix per label: [[TN, FP], [FN, TP]]
    mcm = multilabel_confusion_matrix(y_true, y_pred)
    per_class_metrics: dict[str, dict[str, Any]] = {}

    for idx, label in enumerate(labels_order):
        c_true = y_true[:, idx]
        c_pred = y_pred[:, idx]
        tn, fp, fn, tp = mcm[idx].ravel()
        per_class_metrics[label] = {
            "precision": round(float(precision_score(c_true, c_pred, zero_division=0)), 4),
            "recall": round(float(recall_score(c_true, c_pred, zero_division=0)), 4),
            "f1": round(float(f1_score(c_true, c_pred, zero_division=0)), 4),
            "support": int(np.sum(c_true)),
            "confusion_matrix": {
                "true_negative": int(tn),
                "false_positive": int(fp),
                "false_negative": int(fn),
                "true_positive": int(tp),
            },
        }

    model_dir_repr = "backend/ml/models/recall-topic-classifier"
    if classifier.model_dir:
        resolved_path = Path(classifier.model_dir).resolve()
        try:
            if config.BACKEND_DIR.parent in resolved_path.parents:
                model_dir_repr = str(resolved_path.relative_to(config.BACKEND_DIR.parent)).replace("\\", "/")
            elif config.BACKEND_DIR in resolved_path.parents:
                model_dir_repr = f"backend/{resolved_path.relative_to(config.BACKEND_DIR)}".replace("\\", "/")
            else:
                model_dir_repr = resolved_path.name
        except Exception:
            model_dir_repr = "backend/ml/models/recall-topic-classifier"

    report_dict: dict[str, Any] = {
        "evaluation_split": split_name,
        "sample_count": len(rows),
        "threshold": threshold,
        "random_seed": config.SEED,
        "model_dir": model_dir_repr,
        "model_available": classifier.available,
        "labels": labels_order,
        "metrics": {
            "subset_accuracy": round(subset_accuracy, 4),
            "hamming_loss": round(hamming, 4),
            "hamming_accuracy": round(1.0 - hamming, 4),
            "micro_f1": round(micro_f1, 4),
            "macro_f1": round(macro_f1, 4),
            "weighted_f1": round(weighted_f1, 4),
            "micro_precision": round(micro_precision, 4),
            "macro_precision": round(macro_precision, 4),
            "weighted_precision": round(weighted_precision, 4),
            "micro_recall": round(micro_recall, 4),
            "macro_recall": round(macro_recall, 4),
            "weighted_recall": round(weighted_recall, 4),
        },
        "per_class_metrics": per_class_metrics,
        "data_leakage_audit": {
            "is_leak_free": overlap_audit["is_leak_free"],
            "exact_overlap": overlap_audit["exact_overlap"],
            "normalized_overlap": overlap_audit["normalized_overlap"],
            "split_sizes": overlap_audit["split_sizes"],
        },
    }

    if report_output_path is None:
        report_output_path = config.DATA_DIR / "evaluation_report.json"

    report_output_path.parent.mkdir(parents=True, exist_ok=True)
    report_output_path.write_text(
        json.dumps(report_dict, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return report_dict


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate RECALL topic classifier.")
    parser.add_argument("--split", default="test", choices=["train", "validation", "test"])
    parser.add_argument("--output", default=None, help="Output path for evaluation report JSON")
    args = parser.parse_args()

    out_path = Path(args.output) if args.output else None
    print(f"Evaluating RECALL Topic Classifier on '{args.split}' split...")
    report = evaluate_model(split_name=args.split, report_output_path=out_path)

    metrics = report["metrics"]
    print("\n" + "=" * 60)
    print("RECALL TOPIC CLASSIFIER EVALUATION REPORT")
    print("=" * 60)
    print(f"Split:               {report['evaluation_split']} ({report['sample_count']} samples)")
    print(f"Threshold:           {report['threshold']}")
    print(f"Subset Accuracy:     {metrics['subset_accuracy'] * 100:.2f}%")
    print(f"Hamming Accuracy:    {metrics['hamming_accuracy'] * 100:.2f}% (Hamming loss: {metrics['hamming_loss']:.4f})")
    print(f"Micro F1:            {metrics['micro_f1'] * 100:.2f}%")
    print(f"Macro F1:            {metrics['macro_f1'] * 100:.2f}%")
    print(f"Weighted F1:         {metrics['weighted_f1'] * 100:.2f}%")
    print(f"Micro Precision:     {metrics['micro_precision'] * 100:.2f}%")
    print(f"Micro Recall:        {metrics['micro_recall'] * 100:.2f}%")
    print(f"Leakage Audit:       {'PASS (0 overlap)' if report['data_leakage_audit']['is_leak_free'] else 'FAIL'}")
    print("-" * 60)
    print(f"{'Class':<26} {'F1':<8} {'Precision':<11} {'Recall':<8} {'Support':<8}")
    print("-" * 60)
    for label, c_metrics in report["per_class_metrics"].items():
        print(
            f"{label:<26} "
            f"{c_metrics['f1'] * 100:>5.1f}%  "
            f"{c_metrics['precision'] * 100:>8.1f}%  "
            f"{c_metrics['recall'] * 100:>6.1f}%  "
            f"{c_metrics['support']:>7}"
        )
    print("=" * 60)


if __name__ == "__main__":
    main()

