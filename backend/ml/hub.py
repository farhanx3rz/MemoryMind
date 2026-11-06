"""Hugging Face model hosting, downloading, and caching integration for MemoryMind."""
from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import config

logger = logging.getLogger("memorymind.ml.hub")

META_FILENAME = "huggingface_meta.json"
REQUIRED_CONFIG_FILES = ["config.json", "recall_classifier_config.json"]
WEIGHT_FILE_CANDIDATES = ["model.safetensors", "pytorch_model.bin"]
TOKENIZER_FILE_CANDIDATES = ["tokenizer_config.json", "tokenizer.json", "vocab.txt"]


def get_hf_config() -> dict[str, Any]:
    """Return the current Hugging Face model configuration."""
    return {
        "repo_id": config.HF_MODEL_REPO,
        "revision": config.HF_MODEL_REVISION,
        "token": config.HF_TOKEN,
    }


def get_meta_path(model_dir: Path) -> Path:
    """Return the path to the Hugging Face cache metadata file."""
    return model_dir / META_FILENAME


def read_cached_meta(model_dir: Path) -> dict[str, Any] | None:
    """Read the cache metadata file from a model directory if it exists."""
    meta_file = get_meta_path(model_dir)
    if not meta_file.exists():
        return None
    try:
        data = json.loads(meta_file.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return data
    except (OSError, json.JSONDecodeError) as exc:
        logger.debug("Failed to read cache metadata from %s: %s", meta_file, exc)
    return None


def write_cached_meta(
    model_dir: Path,
    repo_id: str,
    revision: str,
    commit_hash: str | None = None,
) -> None:
    """Write the cache metadata file to record the downloaded model state."""
    meta_file = get_meta_path(model_dir)
    payload = {
        "repo_id": repo_id,
        "revision": revision,
        "commit_hash": commit_hash,
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
    }
    meta_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def validate_model_files(model_dir: Path) -> tuple[bool, str | None]:
    """Validate that a model directory contains all required configuration and weight files."""
    if not model_dir.exists() or not model_dir.is_dir():
        return False, f"Directory does not exist: {model_dir}"

    for req in REQUIRED_CONFIG_FILES:
        if not (model_dir / req).exists():
            return False, f"Missing required file: {req}"

    has_weights = any((model_dir / w).exists() for w in WEIGHT_FILE_CANDIDATES)
    if not has_weights:
        return False, f"Missing model weights ({' or '.join(WEIGHT_FILE_CANDIDATES)})"

    has_tokenizer = any((model_dir / t).exists() for t in TOKENIZER_FILE_CANDIDATES)
    if not has_tokenizer:
        return False, f"Missing tokenizer files ({' or '.join(TOKENIZER_FILE_CANDIDATES)})"

    # Validate recall_classifier_config.json is valid JSON with labels
    try:
        meta = json.loads((model_dir / "recall_classifier_config.json").read_text(encoding="utf-8"))
        if not isinstance(meta, dict) or "labels" not in meta:
            return False, "recall_classifier_config.json is missing required 'labels' field"
    except Exception as exc:
        return False, f"Invalid recall_classifier_config.json: {exc}"

    return True, None


def is_model_cached(
    model_dir: Path,
    repo_id: str = config.HF_MODEL_REPO,
    revision: str = config.HF_MODEL_REVISION,
) -> bool:
    """Check if a valid model matching repo_id and revision is already cached locally."""
    is_valid, _ = validate_model_files(model_dir)
    if not is_valid:
        return False

    meta = read_cached_meta(model_dir)
    if meta is not None:
        cached_repo = meta.get("repo_id")
        cached_rev = meta.get("revision")
        if cached_repo == repo_id and cached_rev == revision:
            return True
        return False

    # If all files exist but no metadata file (e.g. pre-existing model),
    # check if the configured repo matches the default repo and record meta
    if repo_id == config.HF_MODEL_REPO:
        write_cached_meta(model_dir, repo_id, revision)
        return True

    return False


def download_model_from_hf(
    repo_id: str = config.HF_MODEL_REPO,
    revision: str = config.HF_MODEL_REVISION,
    token: str | None = config.HF_TOKEN,
    target_dir: Path | str | None = None,
    force: bool = False,
) -> Path:
    """Download the model from Hugging Face if not cached, with validation and safe rollback.

    Never logs access tokens.
    """
    resolved_dir = Path(target_dir).resolve() if target_dir is not None else Path(config.MODEL_DIR).resolve()

    # Reuse local cache if valid and revision matches
    if not force and is_model_cached(resolved_dir, repo_id=repo_id, revision=revision):
        logger.info(
            "Cached model loaded from %s (repo: %s, revision: %s).",
            resolved_dir,
            repo_id,
            revision,
        )
        return resolved_dir

    old_meta = read_cached_meta(resolved_dir)
    old_revision = old_meta.get("revision") if old_meta else None

    logger.info("Model download started: %s (revision: %s)...", repo_id, revision)

    staging_dir = Path(tempfile.mkdtemp(prefix="mm_model_staging_"))
    try:
        try:
            from huggingface_hub import snapshot_download

            downloaded_path = snapshot_download(
                repo_id=repo_id,
                revision=revision,
                token=token,
                local_dir=staging_dir,
            )
        except Exception as dl_exc:
            logger.error("Model loading failed for %s (revision: %s): %s", repo_id, revision, dl_exc)
            # If a previously working model exists in resolved_dir, preserve it
            if validate_model_files(resolved_dir)[0]:
                logger.warning(
                    "Preserving previously working model at %s after failed download.",
                    resolved_dir,
                )
                return resolved_dir
            raise RuntimeError(
                f"Failed to download topic classifier from Hugging Face ({repo_id}@{revision}) "
                f"and no cached model exists: {dl_exc}. Check your internet connection or HF configuration."
            ) from dl_exc

        staged_dir = Path(downloaded_path)
        is_valid, err_msg = validate_model_files(staged_dir)
        if not is_valid:
            logger.error("Downloaded model failed validation: %s", err_msg)
            if validate_model_files(resolved_dir)[0]:
                logger.warning(
                    "Preserving previously working model at %s after invalid download.",
                    resolved_dir,
                )
                return resolved_dir
            raise ValueError(
                f"Model downloaded from Hugging Face ({repo_id}@{revision}) is invalid: {err_msg}"
            )

        # Move/copy validated files into target_dir
        resolved_dir.mkdir(parents=True, exist_ok=True)
        for item in staged_dir.iterdir():
            if item.name == ".cache":
                continue
            dest = resolved_dir / item.name
            if item.is_file():
                shutil.copy2(item, dest)
            elif item.is_dir():
                if dest.exists():
                    shutil.rmtree(dest)
                shutil.copytree(item, dest)

        write_cached_meta(resolved_dir, repo_id=repo_id, revision=revision)
        logger.info("Model downloaded successfully: %s (revision: %s).", repo_id, revision)
        if old_revision and old_revision != revision:
            logger.info(
                "Model update completed: %s (revision: %s -> %s).",
                repo_id,
                old_revision,
                revision,
            )

        return resolved_dir

    finally:
        if staging_dir.exists():
            shutil.rmtree(staging_dir, ignore_errors=True)


def refresh_model(
    repo_id: str = config.HF_MODEL_REPO,
    revision: str = config.HF_MODEL_REVISION,
    token: str | None = config.HF_TOKEN,
    target_dir: Path | str | None = None,
) -> Path:
    """Force download the latest model from Hugging Face for the configured revision."""
    return download_model_from_hf(
        repo_id=repo_id,
        revision=revision,
        token=token,
        target_dir=target_dir,
        force=True,
    )

