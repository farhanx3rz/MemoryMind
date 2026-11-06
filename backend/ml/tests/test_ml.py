import json
import tempfile
import unittest
from pathlib import Path

from ml import config
from ml.classifier import TopicClassifier, chunk_text
from ml.dataset import (
    group_examples,
    label_distribution,
    load_examples,
    multilabel_vector,
    normalize_text,
    split_examples,
    validate_splits,
)


class MLTests(unittest.TestCase):
    def test_dataset_loads_and_has_all_labels(self):
        examples = load_examples()
        self.assertGreaterEqual(len(examples), 30)
        distribution = label_distribution(examples)
        self.assertTrue(all(distribution[label] > 0 for label in config.LABELS))

    def test_multilabel_encoding(self):
        vector = multilabel_vector(["RAG", "Artificial Intelligence"])
        self.assertEqual(len(vector), len(config.LABELS))
        self.assertEqual(vector[config.LABELS.index("RAG")], 1.0)
        self.assertEqual(vector[config.LABELS.index("Programming")], 0.0)

    def test_split_is_reproducible_and_disjoint(self):
        examples = load_examples()
        first = split_examples(examples, seed=42)
        second = split_examples(examples, seed=42)
        self.assertEqual(first, second)
        texts = [item["text"] for rows in first.values() for item in rows]
        self.assertEqual(len(texts), len(set(texts)))
        validation = validate_splits(first)
        self.assertTrue(validation["is_leak_free"])

    def test_duplicate_leakage_prevented(self):
        """Synthetic duplicate records must never cross train/val/test splits."""
        synthetic = [
            {"text": "Python is a versatile programming language.", "labels": ["Programming"]},
            {"text": "Python is a versatile programming language.  ", "labels": ["Programming"]},
            {"text": "ChromaDB is an open-source embedding database.", "labels": ["Databases", "RAG"]},
            {"text": "ChromaDB is an open-source embedding database.", "labels": ["Databases"]},
            {"text": "Deep learning models require substantial training data.", "labels": ["Deep Learning"]},
            {"text": "Transformers enable attention-based representation.", "labels": ["Large Language Models"]},
            {"text": "Reinforcement learning optimizes reward signals.", "labels": ["Machine Learning"]},
            {"text": "Information retrieval indexes unstructured text.", "labels": ["Information Retrieval"]},
            {"text": "Linear algebra underpins neural transformations.", "labels": ["Mathematics"]},
            {"text": "Cybersecurity protects distributed networked assets.", "labels": ["Cybersecurity"]},
        ]
        # Duplicate each item to guarantee duplicates in dataset
        duplicated = synthetic * 3
        splits = split_examples(duplicated, seed=42, deduplicate=False)
        train_norm = {normalize_text(x["text"]) for x in splits["train"]}
        val_norm = {normalize_text(x["text"]) for x in splits["validation"]}
        test_norm = {normalize_text(x["text"]) for x in splits["test"]}

        self.assertEqual(train_norm & val_norm, set())
        self.assertEqual(train_norm & test_norm, set())
        self.assertEqual(val_norm & test_norm, set())

    def test_near_duplicate_grouping(self):
        """Phrasing variations of identical definitions must cluster into the same group."""
        pairs = [
            {"text": "Kubernetes is an open-source system for automating deployment.", "labels": ["Cloud Computing"]},
            {"text": "Kubernetes refers to an open-source system for automating deployment.", "labels": ["Cloud Computing"]},
            {"text": "The Transformer is a neural network architecture based on attention.", "labels": ["Deep Learning"]},
            {"text": "The Transformer refers to a neural network architecture based on attention.", "labels": ["Deep Learning"]},
        ]
        groups = group_examples(pairs, jaccard_threshold=0.80)
        self.assertEqual(len(groups), 2)
        self.assertEqual(len(groups[0]), 2)
        self.assertEqual(len(groups[1]), 2)

    def test_rare_classes_handling(self):
        """Classes with < 3 examples must not trigger unhandled exceptions."""
        synthetic = [
            {"text": f"Common record number {i} for general topic.", "labels": ["Programming"]}
            for i in range(30)
        ]
        # Add one rare example for History and two for Science
        synthetic.append({"text": "The Roman Republic transitioned to empire.", "labels": ["History"]})
        synthetic.append({"text": "Quantum mechanics describes atomic phenomena.", "labels": ["Science"]})
        synthetic.append({"text": "Cellular respiration yields ATP molecules.", "labels": ["Science"]})

        splits = split_examples(synthetic, seed=42)
        validation = validate_splits(splits)
        self.assertTrue(validation["is_leak_free"])
        # History (1 sample) must be safely placed in train
        train_labels = [l for ex in splits["train"] for l in ex["labels"]]
        self.assertIn("History", train_labels)

    def test_empty_inputs(self):
        splits = split_examples([])
        self.assertEqual(splits, {"train": [], "validation": [], "test": []})

    def test_invalid_split_configurations(self):
        examples = [{"text": "Sample text", "labels": ["Other"]}] * 5
        with self.assertRaises(ValueError):
            split_examples(examples, train_ratio=0.5, val_ratio=0.5, test_ratio=0.5)
        with self.assertRaises(ValueError):
            split_examples(examples, train_ratio=-0.1, val_ratio=0.5, test_ratio=0.6)

    def test_validate_splits_catches_leakage(self):
        leaky_splits = {
            "train": [{"text": "Identical text across splits", "labels": ["Other"]}],
            "validation": [{"text": "Identical text across splits", "labels": ["Other"]}],
            "test": [{"text": "Unique text in test split", "labels": ["Other"]}],
        }
        with self.assertRaises(ValueError):
            validate_splits(leaky_splits)

    def test_long_text_chunking_has_overlap(self):
        text = " ".join(f"word{i}" for i in range(500))
        chunks = chunk_text(text, chunk_size=100, overlap=20)
        self.assertGreater(len(chunks), 1)
        self.assertIn("word80", chunks[1])

    def test_missing_model_does_not_claim_to_be_available(self):
        classifier = TopicClassifier(model_dir=Path(tempfile.mkdtemp()))
        self.assertFalse(classifier.available)


if __name__ == "__main__":
    unittest.main()

