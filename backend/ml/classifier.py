"""Transformer-backed multi-label topic classifier used by RECALL."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from . import config


def chunk_text(
    text: str,
    chunk_size: int = config.CHUNK_SIZE,
    overlap: int = config.CHUNK_OVERLAP,
) -> list[str]:
    """Source-neutral word-window chunking for standalone ML inference."""
    words = re.findall(r"\S+", (text or "").strip())
    if not words:
        return []
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("chunk overlap must be non-negative and smaller than chunk size")

    step = chunk_size - overlap
    chunks: list[str] = []
    for start in range(0, len(words), step):
        chunk = " ".join(words[start : start + chunk_size]).strip()
        if chunk:
            chunks.append(chunk)
        if start + chunk_size >= len(words):
            break
    return chunks


class TopicClassifier:
    """Small abstraction around Hugging Face so backend code stays model-agnostic."""

    def __init__(
        self,
        model_dir: Path | str | None = None,
        threshold: float = config.THRESHOLD,
        auto_download: bool = True,
    ):
        self.explicit_model_dir = model_dir is not None
        self.model_dir = Path(model_dir) if model_dir is not None else Path(config.MODEL_DIR)
        self.auto_download = auto_download

        self.threshold = threshold
        self.labels: list[str] = list(config.LABELS)
        self._sync_metadata(threshold)

        self._tokenizer = None
        self._model = None
        self._torch = None
        self._device = None
        self._load_error: str | None = None

    def _sync_metadata(self, fallback_threshold: float = config.THRESHOLD) -> None:
        saved_config = self.model_dir / "recall_classifier_config.json"
        saved_threshold = None
        saved_labels = None
        if saved_config.exists():
            try:
                meta = json.loads(saved_config.read_text(encoding="utf-8"))
                saved_threshold = float(meta.get("threshold"))
                saved_labels = meta.get("labels")
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                saved_threshold = None
                saved_labels = None

        self.threshold = (
            saved_threshold if (fallback_threshold == config.THRESHOLD and saved_threshold is not None)
            else fallback_threshold
        )
        self.labels = saved_labels if isinstance(saved_labels, list) and saved_labels else list(config.LABELS)

    @property
    def available(self) -> bool:
        """True if the model directory contains configuration and weights."""
        if not self.model_dir.exists():
            return False
        has_config = (self.model_dir / "config.json").exists()
        has_tokenizer = (
            (self.model_dir / "tokenizer_config.json").exists()
            or (self.model_dir / "tokenizer.json").exists()
            or (self.model_dir / "vocab.txt").exists()
        )
        has_weights = (
            (self.model_dir / "model.safetensors").exists()
            or (self.model_dir / "pytorch_model.bin").exists()
        )
        return has_config and has_tokenizer and has_weights

    def get_status(self) -> dict[str, Any]:
        """Return diagnostic health status for the classifier."""
        from . import hub
        cached_meta = hub.read_cached_meta(self.model_dir) if self.model_dir.exists() else None
        return {
            "available": self.available,
            "loaded": self._model is not None,
            "model_dir": str(self.model_dir),
            "threshold": self.threshold,
            "num_labels": len(self.labels),
            "labels": list(self.labels),
            "has_history_label": "History" in self.labels,
            "last_error": self._load_error,
            "hf_repo": config.HF_MODEL_REPO,
            "hf_revision": config.HF_MODEL_REVISION,
            "cached_meta": cached_meta,
        }

    def _load(self) -> None:
        if self._model is not None:
            return

        # Auto-download from Hugging Face if not cached or revision changed
        if self.auto_download:
            from . import hub
            if not hub.is_model_cached(self.model_dir, config.HF_MODEL_REPO, config.HF_MODEL_REVISION):
                try:
                    self.model_dir = hub.download_model_from_hf(
                        repo_id=config.HF_MODEL_REPO,
                        revision=config.HF_MODEL_REVISION,
                        token=config.HF_TOKEN,
                        target_dir=self.model_dir,
                    )
                    self._sync_metadata(self.threshold)
                except Exception as exc:
                    if not self.available:
                        self._load_error = f"Topic classifier unavailable: {exc}"
                        raise

        if not self.available:
            self._load_error = f"Trained topic classifier not found at {self.model_dir}."
            raise FileNotFoundError(
                f"{self._load_error} Run `python -m ml.train` or configure HF_MODEL_REPO."
            )

        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            self._load_error = "The topic classifier requires `torch` and `transformers`."
            raise RuntimeError(
                f"{self._load_error} Install ML dependencies from requirements.txt."
            ) from exc

        try:
            self._torch = torch
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir))
            self._model = AutoModelForSequenceClassification.from_pretrained(str(self.model_dir))
            self._model.to(self._device)
            self._model.eval()

            # Reconcile label order directly from the model configuration
            model_cfg = self._model.config
            num_labels = getattr(model_cfg, "num_labels", len(self.labels))
            id2label = getattr(model_cfg, "id2label", None) or {}

            if id2label:
                resolved_labels = []
                for i in range(num_labels):
                    lbl = id2label.get(i) or id2label.get(str(i))
                    if lbl:
                        resolved_labels.append(str(lbl))
                    elif i < len(self.labels):
                        resolved_labels.append(self.labels[i])
                    else:
                        resolved_labels.append(f"Label_{i}")
                self.labels = resolved_labels

            self._load_error = None
        except Exception as exc:
            self._load_error = f"Failed to load classifier: {type(exc).__name__}: {exc}"
            raise

    def refresh_from_hf(self, force: bool = True) -> None:
        """Force refresh the classifier model from Hugging Face and reload."""
        from . import hub
        self.model_dir = hub.download_model_from_hf(
            repo_id=config.HF_MODEL_REPO,
            revision=config.HF_MODEL_REVISION,
            token=config.HF_TOKEN,
            target_dir=self.model_dir,
            force=force,
        )
        self._sync_metadata(self.threshold)
        self._model = None
        self._tokenizer = None
        self._load()

    def predict_chunks(self, chunks: list[str]) -> list[list[float]]:
        """Predict probabilities for already-created webpage chunks."""
        cleaned = [item.strip() for item in chunks if item and item.strip()]
        if not cleaned:
            return []

        self._load()
        torch = self._torch
        encoded = self._tokenizer(
            cleaned,
            truncation=True,
            max_length=config.MAX_LENGTH,
            padding=True,
            return_tensors="pt",
        )
        encoded = {key: value.to(self._device) for key, value in encoded.items()}
        with torch.inference_mode():
            logits = self._model(**encoded).logits
            probs = torch.sigmoid(logits).cpu().tolist()
            return probs

    def predict(self, text: str) -> list[dict[str, float | str]]:
        """Predict topic probabilities for one model-context-sized text."""
        chunks_probs = self.predict_chunks([text])
        if not chunks_probs:
            return []
        probs = chunks_probs[0]
        return [
            {"label": label, "score": round(float(score), 4)}
            for label, score in zip(self.labels, probs)
        ]

    def predict_article(self, article_text: str) -> list[dict[str, float | str]]:
        """Chunk a long article and aggregate its chunk probabilities."""
        chunks = chunk_text(article_text)
        return self.predict_chunks_to_tags(chunks)

    def predict_chunks_to_tags(self, chunks: list[str]) -> list[dict[str, float | str]]:
        """Aggregate chunk probabilities and apply the configured threshold."""
        if not chunks:
            return []

        # For long articles, sample representative chunks (lead intro + distributed body)
        # to classify document topic efficiently without evaluating dozens of redundant chunks.
        if len(chunks) > 16:
            head = chunks[:4]
            rest = chunks[4:]
            needed = 16 - len(head)
            step = max(1, len(rest) // needed)
            eval_chunks = head + [rest[i] for i in range(0, len(rest), step)][:needed]
        else:
            eval_chunks = chunks

        try:
            probabilities = self.predict_chunks(eval_chunks)
        except Exception as exc:
            self._load_error = str(exc)
            return []

        if not probabilities:
            return []

        # Top-K chunk aggregation: for each label, average its strongest K
        # chunk probabilities. This lets a focused section identify a topic
        # without letting unrelated boilerplate dominate the article score.
        scores: list[float] = []
        for label_index in range(len(self.labels)):
            label_scores = sorted(
                (row[label_index] for row in probabilities if label_index < len(row)),
                reverse=True,
            )[: max(1, config.TOP_K_CHUNKS)]
            scores.append(sum(label_scores) / len(label_scores) if label_scores else 0.0)

        return [
            {"label": label, "score": round(score, 4)}
            for label, score in zip(self.labels, scores)
            if score >= self.threshold
        ]

    def predict_tags(self, article_text: str) -> list[dict[str, float | str]]:
        """Return only selected article-level topic tags."""
        return self.predict_article(article_text)

