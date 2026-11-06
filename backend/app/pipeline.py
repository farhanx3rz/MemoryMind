"""Source-neutral RECALL webpage ingestion pipeline."""
from __future__ import annotations

import concurrent.futures
import logging
import time
from typing import Any

from . import config
from .webpage import WebDocument
from ml.classifier import TopicClassifier

logger = logging.getLogger("memorymind.pipeline")
_classifier: TopicClassifier | None = None


def get_topic_classifier() -> TopicClassifier:
    global _classifier
    if _classifier is None:
        _classifier = TopicClassifier(threshold=config.CLASSIFIER_THRESHOLD)
    return _classifier


def get_classifier_status() -> dict[str, Any]:
    """Return health/degraded-mode diagnostic status for topic classification."""
    if not config.CLASSIFIER_ENABLED:
        return {"status": "disabled", "enabled": False}
    try:
        classifier = get_topic_classifier()
        status = classifier.get_status()
        status["status"] = "ready" if status["available"] and not status.get("last_error") else "degraded"
        status["enabled"] = True
        return status
    except Exception as exc:
        return {"status": "degraded", "enabled": True, "error": str(exc)}


def classify_document(document: WebDocument) -> WebDocument:
    """Add topic probabilities/tags using content only, never source/domain."""
    if not config.CLASSIFIER_ENABLED:
        return document

    try:
        classifier = get_topic_classifier()
        # The webpage layer owns the chunks; the classifier only consumes text.
        document.topics = classifier.predict_chunks_to_tags(document.chunks)
    except Exception as error:
        # The trained model is optional during development; ingestion and RAG
        # remain usable when the classifier artifact is unavailable.
        logger.warning(
            "Topic classification unavailable; continuing without tags: %s",
            error,
        )
        document.topics = []
    return document


def extract_dynamic_tags(document: WebDocument, source_text: str | None = None) -> WebDocument:
    """Add fine-grained Grok concepts at document level.

    One tag extraction call is made per saved memory/page, not once per chunk.
    This keeps tags coherent across the chunks belonging to the same source.
    """
    if not config.DYNAMIC_TAGS_ENABLED:
        document.tags = []
        return document

    from . import llm

    text = source_text or document.extracted_text
    if not text.strip():
        document.tags = []
        return document

    try:
        document.tags = llm.extract_dynamic_tags(text)
    except Exception as error:
        # Tagging is enrichment, not a prerequisite for saving a memory.
        logger.warning(
            "Dynamic tag extraction unavailable; continuing without tags: %s",
            error,
        )
        document.tags = []
    return document


def store_document(document: WebDocument) -> str:
    """Embed and persist a WebDocument using the same store for every source."""
    from . import memory_store

    return memory_store.add_web_document(document)


# ---------------------------------------------------------------------------
# Persistent background executor for async tag extraction.
# max_workers=2 lets two concurrent "remember" actions each have a pending
# tag job without blocking each other.  Thread names help in debug logs.
# These are daemon threads, so they don't prevent process shutdown.
# ---------------------------------------------------------------------------
_tag_executor = concurrent.futures.ThreadPoolExecutor(
    max_workers=2,
    thread_name_prefix="mm-tags",
)


def _background_tag_update(
    document_id: str,
    extracted_text: str,
    dynamic_tag_text: str | None = None,
) -> None:
    """Fetch dynamic tags from the LLM and write them back to the DB.

    Runs in a background thread.  Failures are logged but never propagated —
    the memory is already safely stored without tags and fully queryable.
    """
    from . import llm, memory_store

    tag_text = dynamic_tag_text or extracted_text
    if not tag_text.strip():
        return

    try:
        t0 = time.perf_counter()
        tags = llm.extract_dynamic_tags(tag_text)
        llm_ms = (time.perf_counter() - t0) * 1000
        if tags:
            memory_store.update_document_tags(document_id, tags)
            logger.info(
                "Structured Timing: async_tag_update [document_id=%s, tags=%d, llm_ms=%.2f]",
                document_id, len(tags), llm_ms,
            )
    except Exception as error:
        logger.warning(
            "Background tag extraction failed [document_id=%s]: %s",
            document_id,
            error,
        )


def ingest_document(
    document: WebDocument,
    *,
    dynamic_tag_text: str | None = None,
) -> tuple[WebDocument, str]:
    """Run the optimized pipeline: classify -> store -> tag (async).

    Ordering rationale
    ------------------
    1. classify_document  -- local DistilBERT (~70 ms), must finish before
                             storage so topics are indexed with the chunks.
    2. store_document     -- embed + ChromaDB + SQLite FTS5 (~110 ms).
                             The memory is fully queryable the moment this
                             returns.
    3. extract_dynamic_tags -- external LLM call (~1-6 s of network I/O).
                             Submitted to _tag_executor as a fire-and-forget
                             background task.  The caller (and the Chrome
                             extension) receives a 200 response without
                             waiting for the LLM.  Tags are written to the
                             DB via update_document_tags() when they arrive
                             and are available to subsequent search queries.

    Trade-off: the immediate API response always returns tags=[] for newly
    ingested documents.  The extension status bar omits tag labels on first
    save but still confirms success accurately.  Tags appear in search results
    for any query made after the background job completes (typically 2-8 s).

    If DYNAMIC_TAGS_ENABLED is False or LLM_PROVIDER is "none", no background
    work is submitted and the pipeline is entirely synchronous.
    """
    t0 = time.perf_counter()

    # Step 1 -- local classification (DistilBERT, ~70 ms warm)
    document = classify_document(document)
    classify_ms = (time.perf_counter() - t0) * 1000

    # Step 2 -- store immediately without waiting for external LLM
    t1 = time.perf_counter()
    document.tags = []  # populated asynchronously below
    memory_id = store_document(document)
    store_ms = (time.perf_counter() - t1) * 1000

    # Step 3 -- fire-and-forget background tag extraction
    async_tags = config.DYNAMIC_TAGS_ENABLED and config.LLM_PROVIDER != "none"
    if async_tags:
        _tag_executor.submit(
            _background_tag_update,
            document.document_id,
            document.extracted_text,
            dynamic_tag_text,
        )

    total_ms = (time.perf_counter() - t0) * 1000
    logger.info(
        "Structured Timing: ingest_document [classify_ms=%.2f, store_ms=%.2f, "
        "total_ms=%.2f, async_tags=%s]",
        classify_ms, store_ms, total_ms, async_tags,
    )

    return document, memory_id
