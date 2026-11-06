"""Unit tests for Hugging Face model hosting, downloading, caching, and fallback."""
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from ml import config, hub
from ml.classifier import TopicClassifier


def create_dummy_model_dir(target_dir: Path, labels: list[str] | None = None, threshold: float = 0.5) -> None:
    """Helper to create a valid minimal model directory for testing."""
    target_dir.mkdir(parents=True, exist_ok=True)
    if labels is None:
        labels = ["Artificial Intelligence", "Machine Learning", "Databases"]

    # config.json
    config_data = {
        "architectures": ["DistilBertForSequenceClassification"],
        "model_type": "distilbert",
        "num_labels": len(labels),
        "id2label": {str(i): lbl for i, lbl in enumerate(labels)},
        "label2id": {lbl: i for i, lbl in enumerate(labels)},
    }
    (target_dir / "config.json").write_text(json.dumps(config_data), encoding="utf-8")

    # recall_classifier_config.json
    recall_cfg = {
        "threshold": threshold,
        "labels": labels,
        "model_name": "distilbert-base-uncased",
    }
    (target_dir / "recall_classifier_config.json").write_text(json.dumps(recall_cfg), encoding="utf-8")

    # Weights placeholder (simulates model.safetensors)
    (target_dir / "model.safetensors").write_bytes(b"dummy_safetensors_bytes")

    # Tokenizer files
    (target_dir / "tokenizer_config.json").write_text(json.dumps({"do_lower_case": True}), encoding="utf-8")
    (target_dir / "tokenizer.json").write_text(json.dumps({"version": "1.0"}), encoding="utf-8")
    (target_dir / "special_tokens_map.json").write_text(json.dumps({"unk_token": "[UNK]"}), encoding="utf-8")
    (target_dir / "vocab.txt").write_text("[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\n", encoding="utf-8")


class TestHuggingFaceHubIntegration(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp(prefix="test_hf_"))
        self.model_dir = self.test_dir / "recall-topic-classifier"

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)
        try:
            import transformers.utils.hub as t_hub
            import huggingface_hub
            t_hub.snapshot_download = huggingface_hub.snapshot_download
        except Exception:
            pass

    def test_validation_detects_valid_and_invalid_directories(self):
        # Empty directory should fail validation
        valid, msg = hub.validate_model_files(self.model_dir)
        self.assertFalse(valid)
        self.assertIn("does not exist", msg)

        # Directory with missing weights
        self.model_dir.mkdir(parents=True, exist_ok=True)
        (self.model_dir / "config.json").write_text("{}", encoding="utf-8")
        (self.model_dir / "recall_classifier_config.json").write_text(json.dumps({"labels": ["AI"]}), encoding="utf-8")
        (self.model_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
        valid, msg = hub.validate_model_files(self.model_dir)
        self.assertFalse(valid)
        self.assertIn("Missing model weights", msg)

        # Full valid directory passes
        create_dummy_model_dir(self.model_dir)
        valid, msg = hub.validate_model_files(self.model_dir)
        self.assertTrue(valid)
        self.assertIsNone(msg)

    def test_loading_from_existing_cache_skips_download(self):
        create_dummy_model_dir(self.model_dir)
        hub.write_cached_meta(self.model_dir, repo_id="test/model", revision="main")

        with patch("huggingface_hub.snapshot_download") as mock_download:
            result = hub.download_model_from_hf(
                repo_id="test/model",
                revision="main",
                target_dir=self.model_dir,
            )
            # Must return the cached path without invoking snapshot_download
            self.assertEqual(result, self.model_dir.resolve())
            mock_download.assert_not_called()

    def test_downloading_when_not_cached(self):
        def fake_snapshot_download(*args, **kwargs):
            local_dir = Path(kwargs["local_dir"])
            create_dummy_model_dir(local_dir, labels=["Artificial Intelligence", "Science"])
            return str(local_dir)

        with patch("huggingface_hub.snapshot_download", side_effect=fake_snapshot_download) as mock_download:
            result = hub.download_model_from_hf(
                repo_id="farhanx3rz/memorymind-classifier",
                revision="main",
                token="test_token_secret",
                target_dir=self.model_dir,
            )
            mock_download.assert_called_once()
            # Verify snapshot_download received correct args
            call_kwargs = mock_download.call_args[1]
            self.assertEqual(call_kwargs["repo_id"], "farhanx3rz/memorymind-classifier")
            self.assertEqual(call_kwargs["revision"], "main")
            self.assertEqual(call_kwargs["token"], "test_token_secret")

            # Verify files exist in target_dir
            self.assertTrue((self.model_dir / "model.safetensors").exists())
            self.assertTrue((self.model_dir / "config.json").exists())

            # Verify metadata was written
            meta = hub.read_cached_meta(self.model_dir)
            self.assertIsNotNone(meta)
            self.assertEqual(meta["repo_id"], "farhanx3rz/memorymind-classifier")
            self.assertEqual(meta["revision"], "main")
            self.assertIn("downloaded_at", meta)

    def test_missing_internet_with_existing_cache_preserves_model(self):
        # Cache a working model first
        create_dummy_model_dir(self.model_dir, labels=["Cached Topic"])
        hub.write_cached_meta(self.model_dir, repo_id="farhanx3rz/memorymind-classifier", revision="main")

        # Simulate network disconnect on forced update or revision check
        with patch("huggingface_hub.snapshot_download", side_effect=ConnectionError("Failed to resolve host")):
            result = hub.download_model_from_hf(
                repo_id="farhanx3rz/memorymind-classifier",
                revision="main",
                target_dir=self.model_dir,
                force=True,  # Force refresh while offline
            )
            # Should not raise exception; must gracefully fallback to existing cache
            self.assertEqual(result, self.model_dir.resolve())
            self.assertTrue((self.model_dir / "model.safetensors").exists())

    def test_failed_download_without_cache_raises_actionable_error(self):
        with patch("huggingface_hub.snapshot_download", side_effect=RuntimeError("Repository not found (404)")):
            with self.assertRaises(RuntimeError) as ctx:
                hub.download_model_from_hf(
                    repo_id="nonexistent/repo",
                    revision="main",
                    target_dir=self.model_dir,
                )
            self.assertIn("Failed to download topic classifier from Hugging Face", str(ctx.exception))
            self.assertIn("Check your internet connection or HF configuration", str(ctx.exception))

    def test_invalid_download_does_not_corrupt_cache(self):
        def fake_corrupt_download(*args, **kwargs):
            local_dir = Path(kwargs["local_dir"])
            local_dir.mkdir(parents=True, exist_ok=True)
            # Only write config.json, omitting weights and tokenizer
            (local_dir / "config.json").write_text("{}", encoding="utf-8")
            return str(local_dir)

        with patch("huggingface_hub.snapshot_download", side_effect=fake_corrupt_download):
            with self.assertRaises(ValueError) as ctx:
                hub.download_model_from_hf(
                    repo_id="test/corrupted",
                    revision="main",
                    target_dir=self.model_dir,
                )
            self.assertIn("Missing required file", str(ctx.exception))

    def test_revision_change_triggers_download(self):
        create_dummy_model_dir(self.model_dir, labels=["Old Revision"])
        hub.write_cached_meta(self.model_dir, repo_id="farhanx3rz/memorymind-classifier", revision="v1.0")

        def fake_new_revision_download(*args, **kwargs):
            local_dir = Path(kwargs["local_dir"])
            create_dummy_model_dir(local_dir, labels=["New Revision"])
            return str(local_dir)

        with patch("huggingface_hub.snapshot_download", side_effect=fake_new_revision_download) as mock_dl:
            hub.download_model_from_hf(
                repo_id="farhanx3rz/memorymind-classifier",
                revision="v2.0",
                target_dir=self.model_dir,
            )
            mock_dl.assert_called_once()
            meta = hub.read_cached_meta(self.model_dir)
            self.assertEqual(meta["revision"], "v2.0")
            cfg = json.loads((self.model_dir / "recall_classifier_config.json").read_text(encoding="utf-8"))
            self.assertEqual(cfg["labels"], ["New Revision"])

    def test_failed_update_preserves_previous_working_model(self):
        # Set up a working v1.0 model
        create_dummy_model_dir(self.model_dir, labels=["V1 Working Label"])
        hub.write_cached_meta(self.model_dir, repo_id="farhanx3rz/memorymind-classifier", revision="v1.0")

        # Attempt to upgrade to v2.0, which throws an HTTP/network error
        with patch("huggingface_hub.snapshot_download", side_effect=RuntimeError("500 Server Error")):
            result = hub.download_model_from_hf(
                repo_id="farhanx3rz/memorymind-classifier",
                revision="v2.0",
                target_dir=self.model_dir,
            )
            self.assertEqual(result, self.model_dir.resolve())
            # Previous v1 config and files must still be intact
            cfg = json.loads((self.model_dir / "recall_classifier_config.json").read_text(encoding="utf-8"))
            self.assertEqual(cfg["labels"], ["V1 Working Label"])

    def test_topic_classifier_auto_downloads_and_syncs_metadata(self):
        def fake_download(*args, **kwargs):
            local_dir = Path(kwargs["local_dir"])
            create_dummy_model_dir(local_dir, labels=["Topic A", "Topic B"], threshold=0.75)
            return str(local_dir)

        # Mock both download and HF AutoModel/AutoTokenizer to verify TopicClassifier wiring
        mock_model = MagicMock()
        mock_model.config.num_labels = 2
        mock_model.config.id2label = {0: "Topic A", 1: "Topic B"}

        with patch("huggingface_hub.snapshot_download", side_effect=fake_download), \
             patch("transformers.AutoTokenizer.from_pretrained", return_value=MagicMock()), \
             patch("transformers.AutoModelForSequenceClassification.from_pretrained", return_value=mock_model):

            classifier = TopicClassifier(model_dir=self.model_dir)
            self.assertFalse(classifier.available)  # Empty initially

            # Trigger _load()
            classifier._load()

            self.assertTrue(classifier.available)
            self.assertEqual(classifier.threshold, 0.75)
            self.assertEqual(classifier.labels, ["Topic A", "Topic B"])

            status = classifier.get_status()
            self.assertTrue(status["available"])
            self.assertTrue(status["loaded"])
            self.assertEqual(status["hf_repo"], config.HF_MODEL_REPO)
            self.assertEqual(status["threshold"], 0.75)


if __name__ == "__main__":
    unittest.main()

