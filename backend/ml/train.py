"""Train the RECALL multi-label topic classifier.

Usage:
    python -m ml.train
"""

from __future__ import annotations

import json
import time

import numpy as np

from . import config
from .dataset import (
    TopicDataset,
    label_distribution,
    load_examples,
    save_splits,
    split_examples,
)


def tune_threshold(
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> tuple[float, float]:
    """Select one global threshold on validation data using micro F1."""
    from sklearn.metrics import f1_score

    best_threshold = config.THRESHOLD
    best_score = -1.0

    for threshold in np.arange(0.20, 0.81, 0.05):
        predictions = (probabilities >= threshold).astype(int)

        score = f1_score(
            y_true,
            predictions,
            average="micro",
            zero_division=0,
        )

        if score > best_score:
            best_threshold = float(threshold)
            best_score = float(score)

    return best_threshold, best_score


def main() -> None:
    # ---------------------------------------------------------
    # Load ML dependencies
    # ---------------------------------------------------------
    try:
        import torch
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            DataCollatorWithPadding,
            Trainer,
            TrainingArguments,
        )

    except ImportError as exc:
        raise SystemExit(
            "Training requires transformers and torch. "
            "Install the ML dependencies from backend/requirements.txt, "
            "then rerun `python -m ml.train`."
        ) from exc

    from sklearn.metrics import (
        f1_score,
        precision_score,
        recall_score,
    )

    # ---------------------------------------------------------
    # Dataset
    # ---------------------------------------------------------
    examples = load_examples()
    splits = split_examples(examples)

    save_splits(splits)

    print(f"Dataset: {len(examples)} samples")

    for name, rows in splits.items():
        print(
            f"{name}: {len(rows)} | "
            f"label counts: {label_distribution(rows)}"
        )

    # ---------------------------------------------------------
    # Tokenizer
    # ---------------------------------------------------------
    print(f"\nLoading tokenizer: {config.MODEL_NAME}")

    tokenizer = AutoTokenizer.from_pretrained(
        config.MODEL_NAME
    )

    # ---------------------------------------------------------
    # Datasets
    # ---------------------------------------------------------
    train_ds = TopicDataset(
        splits["train"],
        tokenizer,
    )

    val_ds = TopicDataset(
        splits["validation"],
        tokenizer,
    )

    test_ds = TopicDataset(
        splits["test"],
        tokenizer,
    )

    # ---------------------------------------------------------
    # Model
    # ---------------------------------------------------------
    print(f"\nLoading model: {config.MODEL_NAME}")

    model = AutoModelForSequenceClassification.from_pretrained(
        config.MODEL_NAME,
        num_labels=len(config.LABELS),
        problem_type="multi_label_classification",
        id2label={
            str(i): label
            for i, label in enumerate(config.LABELS)
        },
        label2id={
            label: i
            for i, label in enumerate(config.LABELS)
        },
    )

    # ---------------------------------------------------------
    # IMPORTANT:
    # Dynamically pad examples within each batch.
    #
    # Without this, examples such as:
    #   [25 tokens]
    #   [22 tokens]
    #
    # cannot be stacked into a single tensor.
    # ---------------------------------------------------------
    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer
    )

    # ---------------------------------------------------------
    # Metrics
    # ---------------------------------------------------------
    def metrics(eval_pred):
        logits, labels = eval_pred

        probabilities = 1 / (
            1 + np.exp(-logits)
        )

        predictions = (
            probabilities >= config.THRESHOLD
        ).astype(int)

        return {
            "micro_f1": f1_score(
                labels,
                predictions,
                average="micro",
                zero_division=0,
            ),
            "macro_f1": f1_score(
                labels,
                predictions,
                average="macro",
                zero_division=0,
            ),
            "precision": precision_score(
                labels,
                predictions,
                average="micro",
                zero_division=0,
            ),
            "recall": recall_score(
                labels,
                predictions,
                average="micro",
                zero_division=0,
            ),
        }

    # ---------------------------------------------------------
    # Training configuration
    # ---------------------------------------------------------
    output_dir = config.MODEL_DIR / "checkpoints"

    args_kwargs = dict(
        output_dir=str(output_dir),

        learning_rate=config.LEARNING_RATE,

        per_device_train_batch_size=config.BATCH_SIZE,
        per_device_eval_batch_size=config.BATCH_SIZE,

        num_train_epochs=config.EPOCHS,

        weight_decay=0.01,

        seed=config.SEED,

        logging_steps=max(
            1,
            len(train_ds)
            // max(1, config.BATCH_SIZE),
        ),

        save_strategy="no",

        report_to="none",

        load_best_model_at_end=False,
    )

    # Transformers renamed evaluation_strategy
    # to eval_strategy in newer versions.
    try:
        args = TrainingArguments(
            eval_strategy="epoch",
            **args_kwargs,
        )
    except TypeError:
        args = TrainingArguments(
            evaluation_strategy="epoch",
            **args_kwargs,
        )

    # ---------------------------------------------------------
    # Trainer
    # ---------------------------------------------------------
    trainer = Trainer(
        model=model,
        args=args,

        train_dataset=train_ds,
        eval_dataset=val_ds,

        # THIS IS THE IMPORTANT FIX
        data_collator=data_collator,

        compute_metrics=metrics,
    )

    # ---------------------------------------------------------
    # Train
    # ---------------------------------------------------------
    print("\nStarting training...\n")

    start = time.perf_counter()

    trainer.train()

    training_seconds = time.perf_counter() - start

    print(
        f"\nTraining completed in "
        f"{training_seconds:.2f} seconds."
    )

    # ---------------------------------------------------------
    # Save trained model
    # ---------------------------------------------------------
    config.MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    trainer.save_model(
        str(config.MODEL_DIR)
    )

    tokenizer.save_pretrained(
        str(config.MODEL_DIR)
    )

    # ---------------------------------------------------------
    # Tune threshold using validation set
    # ---------------------------------------------------------
    print("\nTuning classification threshold...")

    validation_output = trainer.predict(
        val_ds
    )

    validation_probs = 1 / (
        1 + np.exp(
            -validation_output.predictions
        )
    )

    validation_labels = np.asarray(
        validation_output.label_ids
    )

    tuned_threshold, validation_micro_f1 = (
        tune_threshold(
            validation_labels,
            validation_probs,
        )
    )

    print(
        f"Selected threshold: "
        f"{tuned_threshold:.2f}"
    )

    print(
        f"Validation micro F1: "
        f"{validation_micro_f1:.4f}"
    )

    # ---------------------------------------------------------
    # Test evaluation
    # ---------------------------------------------------------
    print("\nEvaluating test set...")

    test_output = trainer.predict(
        test_ds
    )

    test_probs = 1 / (
        1 + np.exp(
            -test_output.predictions
        )
    )

    test_labels = np.asarray(
        test_output.label_ids
    )

    test_predictions = (
        test_probs >= tuned_threshold
    ).astype(int)

    test_result = {
        "test_micro_f1": float(
            f1_score(
                test_labels,
                test_predictions,
                average="micro",
                zero_division=0,
            )
        ),

        "test_macro_f1": float(
            f1_score(
                test_labels,
                test_predictions,
                average="macro",
                zero_division=0,
            )
        ),

        "test_precision": float(
            precision_score(
                test_labels,
                test_predictions,
                average="micro",
                zero_division=0,
            )
        ),

        "test_recall": float(
            recall_score(
                test_labels,
                test_predictions,
                average="micro",
                zero_division=0,
            )
        ),
    }

    # ---------------------------------------------------------
    # Save metadata
    # ---------------------------------------------------------
    metadata = {
        "model_name": config.MODEL_NAME,

        "labels": config.LABELS,

        "threshold": tuned_threshold,

        "threshold_selection": (
            "global threshold selected on validation "
            "set by micro F1"
        ),

        "validation_micro_f1_at_selected_threshold": (
            validation_micro_f1
        ),

        "max_length": config.MAX_LENGTH,

        "chunk_size": config.CHUNK_SIZE,

        "chunk_overlap": config.CHUNK_OVERLAP,

        "top_k_chunks": config.TOP_K_CHUNKS,

        "learning_rate": config.LEARNING_RATE,

        "batch_size": config.BATCH_SIZE,

        "epochs": config.EPOCHS,

        "seed": config.SEED,

        "dataset_samples": len(examples),

        "split_sizes": {
            key: len(value)
            for key, value in splits.items()
        },

        "training_seconds": round(
            training_seconds,
            2,
        ),

        "test_metrics": test_result,

        "note": (
            "Starter dataset; expand with real "
            "labeled webpage data before making "
            "strong generalization claims."
        ),
    }

    metadata_path = (
        config.MODEL_DIR
        / "recall_classifier_config.json"
    )

    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    # ---------------------------------------------------------
    # Final report
    # ---------------------------------------------------------
    print("\n" + "=" * 60)
    print("RECALL TRAINING COMPLETE")
    print("=" * 60)

    print(
        f"\nModel: {config.MODEL_NAME}"
    )

    print(
        f"Training samples: {len(examples)}"
    )

    print(
        f"Train: {len(splits['train'])}"
    )

    print(
        f"Validation: {len(splits['validation'])}"
    )

    print(
        f"Test: {len(splits['test'])}"
    )

    print(
        f"\nSelected threshold: "
        f"{tuned_threshold:.2f}"
    )

    print(
        f"Validation micro F1: "
        f"{validation_micro_f1:.4f}"
    )

    print("\nTest metrics:")

    for key, value in test_result.items():
        print(
            f"  {key}: {value:.4f}"
        )

    print(
        f"\nTraining time: "
        f"{training_seconds:.2f} seconds"
    )

    print(
        f"\nModel saved to:\n"
        f"{config.MODEL_DIR}"
    )

    print(
        f"\nMetadata saved to:\n"
        f"{metadata_path}"
    )

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()