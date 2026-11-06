from __future__ import annotations

"""Storing and hybrid-retrieving RECALL memories."""

import concurrent.futures
import json
import logging
import re
import sqlite3
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

logger = logging.getLogger("memorymind.memory_store")

from . import config
from .webpage import WebDocument, clean_text

if TYPE_CHECKING:
    from chromadb.api.client import Client
    from chromadb.api.models.Collection import Collection
    from sentence_transformers import SentenceTransformer

COLLECTION_NAME = "memories"

_model = None

# Common question words add noise to lexical matching. We deliberately keep this
# small because technical terms such as "model", "cloud", and "history" are useful.
STOPWORDS = {
    "a", "an", "and", "are", "am", "about", "as", "at", "be", "by", "can",
    "did", "do", "for", "from", "how", "i", "in", "is", "it", "me", "my",
    "of", "on", "or", "that", "the", "this", "to", "was", "what", "when",
    "where", "which", "who", "why", "with", "you", "your",
}


def get_embedding_model() -> SentenceTransformer:
    global _model
    if _model is None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "Embeddings require sentence-transformers. Install backend/requirements.txt."
            ) from exc
        print(f"Loading embedding model: {config.EMBEDDING_MODEL}")
        _model = SentenceTransformer(config.EMBEDDING_MODEL)
    return _model


def get_client() -> Client:
    try:
        import chromadb
    except ImportError as exc:
        raise RuntimeError(
            "Vector storage requires chromadb. Install backend/requirements.txt."
        ) from exc
    config.CHROMA_PATH.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(config.CHROMA_PATH))


def get_collection() -> Collection:
    client = get_client()
    settings = {"hnsw": {"space": "cosine"}}
    try:
        return client.get_or_create_collection(
            name=COLLECTION_NAME, configuration=settings
        )
    except TypeError:
        return client.get_or_create_collection(
            name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
        )


_embedding_cache: dict[str, list[float]] = {}


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Turn a list of strings into vectors with chunk caching and batching."""
    if not texts:
        return []

    global _embedding_cache
    results: list[list[float] | None] = [None] * len(texts)
    missing_indices: list[int] = []
    missing_texts: list[str] = []

    for idx, text in enumerate(texts):
        cached = _embedding_cache.get(text)
        if cached is not None:
            results[idx] = cached
        else:
            missing_indices.append(idx)
            missing_texts.append(text)

    if missing_texts:
        vectors = get_embedding_model().encode(
            missing_texts,
            batch_size=64,
            show_progress_bar=False,
        )
        for idx, text, vec in zip(missing_indices, missing_texts, vectors):
            vec_list = vec.tolist()
            if len(_embedding_cache) < 20000:
                _embedding_cache[text] = vec_list
            results[idx] = vec_list

    return [r for r in results if r is not None]


def add_memory(
    text: str,
    source_url: str = "",
    source_title: str = "",
    raw_text: str = "",
    tags: list[dict[str, Any]] | None = None,
    topics: list[dict[str, Any]] | None = None,
) -> str:
    """Backward-compatible single-memory insert."""
    document = WebDocument(
        url=source_url or "",
        title=clean_text(source_title) or "Untitled webpage",
        source=urlparse(source_url or "").netloc.lower(),
        extracted_text=clean_text(raw_text or text),
        chunks=[clean_text(text)],
        timestamp=datetime.now(timezone.utc).isoformat(),
        topics=topics or [],
        tags=tags or [],
        metadata={"source_type": "selection"},
    )
    return add_web_document(document, memory_text_override=clean_text(text))


def build_chunk_metadata(
    document: WebDocument,
    chunk_index: int,
    created_at: str,
) -> dict[str, Any]:
    """Build provenance + topic/tag metadata attached to every stored chunk."""
    return {
        "document_id": document.document_id or "",
        "chunk_index": chunk_index,
        "source_url": document.url or "",
        "source_title": document.title or "Untitled webpage",
        "source_domain": document.source or "",
        "created_at": created_at,
        "timestamp": created_at,
        "visited_at": created_at,
        "raw_text": clean_text(document.extracted_text)[:1000],
        # DistilBERT controlled taxonomy.
        "topics": json.dumps(document.topics or [], ensure_ascii=False),
        # Grok fine-grained concepts.
        "tags": json.dumps(document.tags or [], ensure_ascii=False),
        "metadata": json.dumps(document.metadata or {}, ensure_ascii=False),
    }


def add_web_document(
    document: WebDocument,
    *,
    memory_text_override: str | None = None,
) -> str:
    """Store each document chunk with complete source/topic/tag provenance."""
    t_start = time.perf_counter()
    raw_chunks = [clean_text(item) for item in document.chunks if clean_text(item)]
    if memory_text_override:
        raw_chunks = [clean_text(memory_text_override)]

    # Deduplicate identical chunks while preserving ordering
    seen: set[str] = set()
    chunks: list[str] = []
    for item in raw_chunks:
        if item not in seen:
            seen.add(item)
            chunks.append(item)

    if not chunks:
        raise ValueError("Cannot store an empty webpage document.")

    parent_id = document.document_id or str(uuid.uuid4())
    document.document_id = parent_id

    t_embed = time.perf_counter()
    vectors = embed_texts(chunks)
    embed_ms = (time.perf_counter() - t_embed) * 1000

    created_at = document.timestamp or datetime.now(timezone.utc).isoformat()
    collection = get_collection()

    ids = [f"{parent_id}:{index}" for index in range(len(chunks))]
    metadatas = []
    for index in range(len(chunks)):
        metadata = build_chunk_metadata(document, index, created_at)
        metadata["document_id"] = parent_id
        metadatas.append(metadata)

    t_chroma = time.perf_counter()
    collection.add(
        ids=ids,
        embeddings=vectors,
        documents=chunks,
        metadatas=metadatas,
    )
    chroma_ms = (time.perf_counter() - t_chroma) * 1000

    t_fts = time.perf_counter()
    # Incremental update into SQLite FTS index
    with get_fts_connection() as conn:
        for chunk_id, doc, meta in zip(ids, chunks, metadatas):
            conn.execute("""
                INSERT OR REPLACE INTO chunk_records
                (id, document_id, chunk_index, source_url, source_title, source_domain,
                 created_at, timestamp, visited_at, raw_text, topics, tags, metadata, text)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                chunk_id,
                meta.get("document_id", ""),
                int(meta.get("chunk_index", 0) or 0),
                meta.get("source_url", ""),
                meta.get("source_title", ""),
                meta.get("source_domain", ""),
                meta.get("created_at", ""),
                meta.get("timestamp", ""),
                meta.get("visited_at", ""),
                meta.get("raw_text", ""),
                meta.get("topics", "[]"),
                meta.get("tags", "[]"),
                meta.get("metadata", "{}"),
                doc or "",
            ))
            conn.execute("DELETE FROM chunk_fts WHERE id = ?", (chunk_id,))
            conn.execute(
                "INSERT INTO chunk_fts(id, source_title, text) VALUES (?, ?, ?)",
                (chunk_id, str(meta.get("source_title", "")), doc or ""),
            )
    fts_ms = (time.perf_counter() - t_fts) * 1000

    invalidate_vocab_cache()
    total_ms = (time.perf_counter() - t_start) * 1000
    logger.info(
        "Structured Timing: store_document [chunks=%d, total_ms=%.2f, embed_ms=%.2f, chroma_ms=%.2f, fts_ms=%.2f]",
        len(chunks), total_ms, embed_ms, chroma_ms, fts_ms,
    )
    return parent_id


def update_document_tags(document_id: str, tags: list[dict[str, Any]]) -> int:
    """Write dynamic tags back to all chunks belonging to a stored document.

    Called by the background tag extraction job after the external LLM responds.
    Patches both ChromaDB chunk metadatas and the SQLite chunk_records table so
    that subsequent retrieval queries can score on tag overlap.

    Returns the number of chunk rows updated.
    """
    t_start = time.perf_counter()
    tags_json = json.dumps(tags, ensure_ascii=False)

    # 1) Find all chunk IDs for this document from SQLite (fast lookup)
    updated_count = 0
    with get_fts_connection() as conn:
        rows = conn.execute(
            "SELECT id FROM chunk_records WHERE document_id = ?",
            (document_id,),
        ).fetchall()
        chunk_ids = [row["id"] for row in rows]

        if not chunk_ids:
            logger.warning(
                "update_document_tags: no chunks found for document_id=%s", document_id
            )
            return 0

        # 2) Update SQLite chunk_records in bulk
        conn.executemany(
            "UPDATE chunk_records SET tags = ? WHERE id = ?",
            [(tags_json, chunk_id) for chunk_id in chunk_ids],
        )
        updated_count = len(chunk_ids)

    # 3) Patch ChromaDB metadata for each chunk
    try:
        collection = get_collection()
        existing = collection.get(ids=chunk_ids, include=["metadatas"])
        if existing and existing.get("ids"):
            updated_metadatas = []
            for meta in existing["metadatas"]:
                patched = dict(meta or {})
                patched["tags"] = tags_json
                updated_metadatas.append(patched)
            collection.update(
                ids=existing["ids"],
                metadatas=updated_metadatas,
            )
    except Exception as exc:
        # SQLite is the authoritative source for retrieval; Chroma patch is
        # best-effort. Log but don't fail — tags are already in SQLite.
        logger.warning(
            "update_document_tags: ChromaDB patch failed [document_id=%s]: %s",
            document_id, exc,
        )

    total_ms = (time.perf_counter() - t_start) * 1000
    logger.info(
        "Structured Timing: update_document_tags [document_id=%s, chunks=%d, tags=%d, total_ms=%.2f]",
        document_id, updated_count, len(tags), total_ms,
    )
    return updated_count


def get_document_info(document_id: str | None = None, url: str | None = None) -> dict[str, Any] | None:
    """Retrieve document details (document_id, url, title, topics, tags, ready) by document_id or url."""
    if not document_id and not url:
        return None

    query_urls: list[str] = []
    if url:
        clean_url = url.split("#")[0].strip()
        query_urls.append(clean_url)
        if clean_url.endswith("/"):
            query_urls.append(clean_url.rstrip("/"))
        else:
            query_urls.append(clean_url + "/")

    with get_fts_connection() as conn:
        row = None
        if document_id:
            row = conn.execute(
                "SELECT document_id, source_url, source_title, topics, tags FROM chunk_records WHERE document_id = ? ORDER BY (tags != '[]' AND tags != '') DESC LIMIT 1",
                (document_id,),
            ).fetchone()

        if not row and query_urls:
            placeholders = ",".join("?" for _ in query_urls)
            row = conn.execute(
                f"SELECT document_id, source_url, source_title, topics, tags FROM chunk_records WHERE source_url IN ({placeholders}) ORDER BY (tags != '[]' AND tags != '') DESC, rowid DESC LIMIT 1",
                query_urls,
            ).fetchone()

        if not row and document_id:
            # Fall back: document_id might actually be a URL passed as the identifier
            clean_id_url = document_id.split("#")[0].strip()
            id_urls = [clean_id_url]
            if clean_id_url.endswith("/"):
                id_urls.append(clean_id_url.rstrip("/"))
            else:
                id_urls.append(clean_id_url + "/")
            placeholders = ",".join("?" for _ in id_urls)
            row = conn.execute(
                f"SELECT document_id, source_url, source_title, topics, tags FROM chunk_records WHERE source_url IN ({placeholders}) ORDER BY (tags != '[]' AND tags != '') DESC, rowid DESC LIMIT 1",
                id_urls,
            ).fetchone()

        if row:
            raw_tags = row["tags"] or "[]"
            raw_topics = row["topics"] or "[]"
            tags = []
            topics = []
            try:
                parsed_tags = json.loads(raw_tags)
                if isinstance(parsed_tags, list):
                    tags = parsed_tags
            except Exception:
                tags = []
            try:
                parsed_topics = json.loads(raw_topics)
                if isinstance(parsed_topics, list):
                    topics = parsed_topics
            except Exception:
                topics = []

            return {
                "document_id": row["document_id"],
                "url": row["source_url"],
                "title": row["source_title"],
                "topics": topics,
                "tags": tags,
                "ready": len(tags) > 0,
            }

    return None


def get_tags_for_document(document_id: str) -> list[dict[str, Any]]:
    """Retrieve dynamic tags for a stored document from SQLite."""
    if not document_id:
        return []
    info = get_document_info(document_id=document_id)
    if info and info.get("tags"):
        return info["tags"]
    return []


def get_tags_for_url(url: str) -> list[dict[str, Any]]:
    """Retrieve dynamic tags for a URL if previously stored."""
    if not url:
        return []
    info = get_document_info(url=url)
    if info and info.get("tags"):
        return info["tags"]
    return []


def add_memories(texts: list[str]) -> list[str]:
    """Bulk insert demo memories without a webpage source."""
    ids = []
    for text in texts:
        if text.strip():
            ids.append(add_memory(text=text))
    return ids


def _source_key(memory: dict[str, Any]) -> str:
    """Identify a saved document without collapsing separate reading events."""
    document_id = str(memory.get("document_id") or "").strip()
    if document_id:
        return f"document:{document_id}"

    source_url = str(memory.get("source_url") or "").strip()
    if source_url:
        return f"url:{source_url}"

    return f"memory:{memory.get('id', '')}"


def _merge_label_objects(
    target: dict[str, dict[str, Any]],
    values: list[dict[str, Any]],
) -> None:
    """Merge topic/tag objects by normalized label while keeping the best score."""
    for value in values:
        label = str(value.get("label", "")).strip()
        if not label:
            continue
        key = label.casefold()
        score = float(value.get("score", 1.0) or 1.0)
        existing = target.get(key)
        if existing is None or score > float(existing.get("score", 0.0)):
            target[key] = {"label": label, "score": score}


def build_source_results(
    matches: list[dict[str, Any]],
    limit: int = config.SOURCE_TOP_K,
) -> list[dict[str, Any]]:
    """Collapse chunk matches into source/article-level results.

    A source keeps the strongest chunk score and up to CHUNKS_PER_SOURCE
    relevant snippets, so one long article cannot dominate the result set.
    Separate document_ids remain separate reading records even when URLs match.
    """
    groups: dict[str, dict[str, Any]] = {}

    ordered_matches = sorted(
        matches,
        key=lambda item: (
            float(item.get("score", 0.0)),
            float(item.get("semantic_score", 0.0)),
            float(item.get("keyword_score", 0.0)),
        ),
        reverse=True,
    )

    for memory in ordered_matches:
        key = _source_key(memory)
        group = groups.get(key)
        if group is None:
            group = {
                "document_id": memory.get("document_id") or "",
                "title": memory.get("source_title") or "Saved memory",
                "url": memory.get("source_url") or "",
                "domain": memory.get("source_domain") or "",
                "visited_at": memory.get("visited_at")
                or memory.get("timestamp")
                or memory.get("created_at")
                or "",
                "score": float(memory.get("score", 0.0)),
                "chunk_count": 0,
                "snippets": [],
                "topics": {},
                "tags": {},
                "_chunk_ids": set(),
            }
            groups[key] = group

        chunk_id = memory.get("id")
        if chunk_id not in group["_chunk_ids"]:
            group["_chunk_ids"].add(chunk_id)
            group["chunk_count"] += 1

        group["score"] = max(group["score"], float(memory.get("score", 0.0)))
        _merge_label_objects(group["topics"], memory.get("topics", []))
        _merge_label_objects(group["tags"], memory.get("tags", []))

        snippet = clean_text(str(memory.get("text") or ""))
        if snippet and len(group["snippets"]) < config.CHUNKS_PER_SOURCE:
            group["snippets"].append(
                {
                    "text": snippet[:700],
                    "score": round(float(memory.get("score", 0.0)), 4),
                    "chunk_index": int(memory.get("chunk_index", 0) or 0),
                }
            )

    results = []
    for source_key, group in groups.items():
        tags = list(group["tags"].values())
        topics = list(group["topics"].values())
        tags.sort(key=lambda item: item["score"], reverse=True)
        topics.sort(key=lambda item: item["score"], reverse=True)
        snippets = sorted(group["snippets"], key=lambda item: item["score"], reverse=True)
        results.append(
            {
                "source_key": source_key,
                "document_id": group["document_id"],
                "title": group["title"],
                "url": group["url"],
                "domain": group["domain"],
                "visited_at": group["visited_at"],
                "score": round(group["score"], 4),
                "chunk_count": group["chunk_count"],
                "snippet": snippets[0]["text"] if snippets else "",
                "snippets": snippets,
                "topics": topics,
                "tags": tags,
            }
        )

    results.sort(key=lambda item: item["score"], reverse=True)
    return results[: max(1, limit)]


def build_related_concepts(
    source_results: list[dict[str, Any]],
    limit: int = 20,
) -> list[dict[str, Any]]:
    """Group concepts by the number of distinct sources that contain them."""
    concepts: dict[tuple[str, str], dict[str, Any]] = {}
    for source in source_results:
        seen: set[tuple[str, str]] = set()
        for kind, values in (("topic", source.get("topics", [])), ("tag", source.get("tags", []))):
            for value in values:
                label = str(value.get("label", "")).strip()
                if not label:
                    continue
                key = (kind, label.casefold())
                if key in seen:
                    continue
                seen.add(key)
                concept = concepts.get(key)
                if concept is None:
                    concept = {
                        "label": label,
                        "kind": kind,
                        "source_count": 0,
                        "sources": [],
                    }
                    concepts[key] = concept
                concept["source_count"] += 1
                concept["sources"].append(
                    {
                        "document_id": source.get("document_id") or "",
                        "title": source.get("title") or "Saved memory",
                        "url": source.get("url") or "",
                    }
                )

    results = list(concepts.values())
    results.sort(
        key=lambda item: (item["source_count"], len(item["sources"])),
        reverse=True,
    )
    return results[: max(1, limit)]


def _parse_json_list(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    try:
        parsed = json.loads(value or "[]")
        return [item for item in parsed if isinstance(item, dict)] if isinstance(parsed, list) else []
    except (TypeError, json.JSONDecodeError):
        return []


def _parse_topics(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    """Read controlled topics, including compatibility with older Chroma data."""
    topics = _parse_json_list(metadata.get("topics"))
    if topics:
        return topics

    # Older versions stored DistilBERT labels in the `tags` field. Recover those
    # as topics instead of treating them as dynamic concepts.
    legacy = _parse_json_list(metadata.get("tags"))
    try:
        from ml import config as ml_config
        valid = set(ml_config.LABELS)
    except Exception:
        valid = set()

    recovered = [
        item for item in legacy
        if str(item.get("label", "")) in valid
    ]
    return recovered


def _parse_dynamic_tags(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    """Read Grok tags; legacy records return no dynamic tags."""
    if metadata.get("topics"):
        return _parse_json_list(metadata.get("tags"))
    return []


def _meaningful_tokens(text: str) -> set[str]:
    tokens = re.findall(r"[a-z0-9]+", (text or "").casefold())
    return {_singularize(token) for token in tokens if token not in STOPWORDS and len(token) > 1}


def _singularize(token: str) -> str:
    """Tiny normalization for common plural technical terms."""
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 4 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def _normalized_phrase(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", (text or "").casefold()))


def _levenshtein_distance(s1: str, s2: str) -> int:
    """Compute Levenshtein edit distance between two strings."""
    if s1 == s2:
        return 0
    if len(s1) < len(s2):
        s1, s2 = s2, s1
    if not s2:
        return len(s1)
    prev = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr = [i + 1] * (len(s2) + 1)
        for j, c2 in enumerate(s2):
            insertions = prev[j + 1] + 1
            deletions = curr[j] + 1
            substitutions = prev[j] + (0 if c1 == c2 else 1)
            curr[j + 1] = min(insertions, deletions, substitutions)
        prev = curr
    return prev[-1]


_vocab_cache: set[str] | None = None


def get_vocab_cache() -> set[str]:
    """In-memory cache of entity/title keywords extracted from saved memories."""
    global _vocab_cache
    if _vocab_cache is None:
        try:
            with get_fts_connection() as conn:
                title_rows = conn.execute("SELECT DISTINCT source_title FROM chunk_records").fetchall()
            vocab = set()
            for r in title_rows:
                t = r["source_title"] or ""
                for tok in re.findall(r"[a-z0-9]+", t.lower()):
                    if len(tok) >= 3 and tok not in STOPWORDS:
                        vocab.add(_singularize(tok))
            _vocab_cache = vocab
        except Exception:
            _vocab_cache = set()
    return _vocab_cache


def invalidate_vocab_cache() -> None:
    """Invalidate cached vocabulary whenever memories are added, updated, or wiped."""
    global _vocab_cache
    _vocab_cache = None


def expand_query_typos(query: str) -> tuple[str, list[str]]:
    """Detect single-character typos against saved title/entity keywords.

    Returns (normalized_query, list_of_expanded_terms).
    For example: 'elon must companies' -> ('elon musk companies', ['elon', 'must', 'musk', 'companies'])
    """
    vocab = get_vocab_cache()
    tokens = re.findall(r"[a-z0-9]+", (query or "").lower())
    if not tokens or not vocab:
        return query, tokens

    corrected_tokens = []
    expanded_terms = list(tokens)

    for tok in tokens:
        sing = _singularize(tok)
        if sing in vocab or len(tok) < 4:
            corrected_tokens.append(tok)
            continue

        match = None
        for candidate in vocab:
            if abs(len(candidate) - len(sing)) <= 1:
                if _levenshtein_distance(sing, candidate) == 1:
                    match = candidate
                    break

        if match:
            corrected_tokens.append(match)
            if match not in expanded_terms:
                expanded_terms.append(match)
        else:
            corrected_tokens.append(tok)

    normalized_query = " ".join(corrected_tokens)
    return normalized_query, expanded_terms


def _keyword_score(query: str, memory_text: str, title: str = "", normalized_query: str = "") -> float:
    """Score exact phrase/token overlap between the query and memory text with typo-tolerance."""
    query_norm = _normalized_phrase(query)
    searchable = _normalized_phrase(f"{title} {memory_text}")
    if not query_norm or not searchable:
        return 0.0

    padded_searchable = f" {searchable} "
    if f" {query_norm} " in padded_searchable:
        return 1.0
    if normalized_query:
        norm_q = _normalized_phrase(normalized_query)
        if norm_q and f" {norm_q} " in padded_searchable:
            return 1.0

    query_tokens = _meaningful_tokens(query)
    if not query_tokens:
        return 0.0
    text_tokens = _meaningful_tokens(f"{title} {memory_text}")
    overlap = query_tokens & text_tokens

    # Check fuzzy matches for unmatched tokens (edit distance <= 1 on >= 4 char tokens)
    unmatched_query = query_tokens - overlap
    unmatched_text = text_tokens - overlap
    fuzzy_matches = 0
    for q_tok in unmatched_query:
        if len(q_tok) >= 4:
            for t_tok in unmatched_text:
                if abs(len(q_tok) - len(t_tok)) <= 1 and _levenshtein_distance(q_tok, t_tok) == 1:
                    fuzzy_matches += 1
                    break

    score = (len(overlap) + 0.85 * fuzzy_matches) / len(query_tokens)
    return round(min(1.0, score), 4)


def _topic_score(query_topics: list[dict[str, Any]], memory_topics: list[dict[str, Any]]) -> float:
    query_labels = {
        str(item.get("label", "")).casefold() for item in query_topics if item.get("label")
    }
    memory_labels = {
        str(item.get("label", "")).casefold() for item in memory_topics if item.get("label")
    }
    if not query_labels or not memory_labels:
        return 0.0
    return round(len(query_labels & memory_labels) / len(query_labels), 4)


def _tag_score(query: str, memory_tags: list[dict[str, Any]]) -> float:
    """Score overlap between query concepts and Grok-generated memory tags."""
    query_norm = _normalized_phrase(query)
    query_tokens = _meaningful_tokens(query)
    if not query_norm or not query_tokens:
        return 0.0

    best = 0.0
    for item in memory_tags:
        label = str(item.get("label", "")).strip()
        label_norm = _normalized_phrase(label)
        if not label_norm:
            continue
        if f" {label_norm} " in f" {query_norm} " or f" {query_norm} " in f" {label_norm} ":
            best = max(best, 1.0)
            continue

        tag_tokens = _meaningful_tokens(label)
        if not tag_tokens:
            continue
        overlap = len(query_tokens & tag_tokens)
        if overlap:
            coverage = overlap / len(tag_tokens)
            best = max(best, 1.0 if coverage == 1.0 else min(0.9, 0.45 + 0.55 * coverage))
    return round(best, 4)


def _hybrid_score(
    semantic_score: float,
    keyword_score: float,
    topic_score: float,
    tag_score: float,
) -> float:
    """Combine all four retrieval signals into one ranking score."""
    total_weight = (
        config.HYBRID_SEMANTIC_WEIGHT
        + config.HYBRID_KEYWORD_WEIGHT
        + config.HYBRID_TOPIC_WEIGHT
        + config.HYBRID_TAG_WEIGHT
    )
    if total_weight <= 0:
        raise ValueError("Hybrid retrieval weights must sum to a positive value.")

    score = (
        semantic_score * config.HYBRID_SEMANTIC_WEIGHT
        + keyword_score * config.HYBRID_KEYWORD_WEIGHT
        + topic_score * config.HYBRID_TOPIC_WEIGHT
        + tag_score * config.HYBRID_TAG_WEIGHT
    ) / total_weight
    return round(score, 4)


def _query_topics(query: str) -> list[dict[str, Any]]:
    """Classify the question with the same local DistilBERT taxonomy."""
    if not config.CLASSIFIER_ENABLED:
        return []
    try:
        from . import pipeline

        classifier = pipeline.get_topic_classifier()
        predictions = classifier.predict(query)
        return [
            item for item in predictions
            if float(item.get("score", 0.0)) >= classifier.threshold
        ]
    except Exception:
        # Retrieval must remain usable when the optional classifier artifact
        # is missing or unavailable.
        return []



def get_fts_db_path() -> Path:
    config.CHROMA_PATH.mkdir(parents=True, exist_ok=True)
    return config.CHROMA_PATH / "lexical_fts.sqlite3"


def get_fts_connection() -> sqlite3.Connection:
    db_path = get_fts_db_path()
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    _init_fts_schema(conn)
    return conn


def _init_fts_schema(conn: sqlite3.Connection) -> None:
    with conn:
        try:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
        except Exception:
            pass
        conn.execute("""
            CREATE TABLE IF NOT EXISTS chunk_records (
                id TEXT PRIMARY KEY,
                document_id TEXT,
                chunk_index INTEGER,
                source_url TEXT,
                source_title TEXT,
                source_domain TEXT,
                created_at TEXT,
                timestamp TEXT,
                visited_at TEXT,
                raw_text TEXT,
                topics TEXT,
                tags TEXT,
                metadata TEXT,
                text TEXT
            );
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_chunk_records_doc_id ON chunk_records(document_id);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_chunk_records_url ON chunk_records(source_url);")
        conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS chunk_fts USING fts5(
                id UNINDEXED,
                source_title,
                text,
                tokenize='unicode61'
            );
        """)


def sync_lexical_index(collection: Collection | None = None) -> int:
    """Incremental, idempotent sync from ChromaDB into SQLite FTS5.

    Only fetches chunks that are missing or stale in the FTS index.
    Never scans or modifies the existing Chroma collection unnecessarily.
    """
    if collection is None:
        collection = get_collection()
    count = collection.count()
    if count == 0:
        return 0

    with get_fts_connection() as conn:
        indexed_ids = {row[0] for row in conn.execute("SELECT id FROM chunk_records").fetchall()}
        chroma_ids = set(collection.get(include=[])["ids"])

        missing_ids = list(chroma_ids - indexed_ids)
        stale_ids = list(indexed_ids - chroma_ids)

        if stale_ids:
            for s_id in stale_ids:
                conn.execute("DELETE FROM chunk_records WHERE id = ?", (s_id,))
                conn.execute("DELETE FROM chunk_fts WHERE id = ?", (s_id,))

        if missing_ids:
            batch_size = 100
            for i in range(0, len(missing_ids), batch_size):
                batch = missing_ids[i : i + batch_size]
                fetched = collection.get(ids=batch, include=["documents", "metadatas"])
                for chunk_id, doc, meta in zip(fetched["ids"], fetched["documents"], fetched["metadatas"]):
                    meta = meta or {}
                    conn.execute("""
                        INSERT OR REPLACE INTO chunk_records
                        (id, document_id, chunk_index, source_url, source_title, source_domain,
                         created_at, timestamp, visited_at, raw_text, topics, tags, metadata, text)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        chunk_id,
                        meta.get("document_id", ""),
                        int(meta.get("chunk_index", 0) or 0),
                        meta.get("source_url", ""),
                        meta.get("source_title", ""),
                        meta.get("source_domain", ""),
                        meta.get("created_at", ""),
                        meta.get("timestamp", ""),
                        meta.get("visited_at", ""),
                        meta.get("raw_text", ""),
                        meta.get("topics", "[]"),
                        meta.get("tags", "[]"),
                        meta.get("metadata", "{}"),
                        doc or "",
                    ))
                    conn.execute("DELETE FROM chunk_fts WHERE id = ?", (chunk_id,))
                    conn.execute(
                        "INSERT INTO chunk_fts(id, source_title, text) VALUES (?, ?, ?)",
                        (chunk_id, str(meta.get("source_title", "")), doc or ""),
                    )
            return len(missing_ids)
    return 0


def rebuild_lexical_index() -> int:
    """Force rebuild of the SQLite FTS5 index from ChromaDB."""
    collection = get_collection()
    with get_fts_connection() as conn:
        conn.execute("DELETE FROM chunk_records;")
        conn.execute("DELETE FROM chunk_fts;")
    return sync_lexical_index(collection)


def search_lexical_fts(
    query: str,
    limit: int = config.HYBRID_SEMANTIC_CANDIDATES,
    expanded_terms: list[str] | None = None,
) -> dict[str, float]:
    """Retrieve top candidate chunk IDs with normalized BM25 scores using SQLite FTS5."""
    clean_words = expanded_terms or re.findall(r"[A-Za-z0-9_]+", query)
    if not clean_words:
        return {}

    phrase = " ".join(re.findall(r"[A-Za-z0-9_]+", query))
    tokens_query = " OR ".join(f'"{w}"*' for w in clean_words)
    fts_expr = f'("{phrase}") OR ({tokens_query})' if phrase else tokens_query

    results: dict[str, float] = {}
    with get_fts_connection() as conn:
        try:
            rows = conn.execute(
                "SELECT id, bm25(chunk_fts) as rank FROM chunk_fts WHERE chunk_fts MATCH ? ORDER BY rank ASC LIMIT ?",
                (fts_expr, max(1, limit)),
            ).fetchall()
        except sqlite3.OperationalError:
            try:
                rows = conn.execute(
                    "SELECT id, bm25(chunk_fts) as rank FROM chunk_fts WHERE chunk_fts MATCH ? ORDER BY rank ASC LIMIT ?",
                    (" OR ".join(clean_words), max(1, limit)),
                ).fetchall()
            except sqlite3.OperationalError:
                return {}

        for row in rows:
            chunk_id = row["id"]
            rank = float(row["rank"])
            score = max(0.1, min(1.0, round(-rank / (1.0 - rank), 4))) if rank < 0 else 0.1
            results[chunk_id] = score

    return results


def search_memories(
    query: str,
    top_k: int = config.TOP_K,
    min_score: float | None = None,
) -> list[dict[str, Any]]:
    """Hybrid retrieval with parallel semantic/lexical candidate pooling and typo-tolerance."""
    t_search_start = time.perf_counter()
    if min_score is None:
        min_score = config.HYBRID_MIN_SCORE

    collection = get_collection()
    count = collection.count()
    if count == 0:
        return []

    # Detect typos and expand search query (e.g. 'elon must' -> 'elon musk')
    normalized_query, expanded_terms = expand_query_typos(query)

    # 1) Semantic candidate vectors: embed original and typo-normalized queries
    queries_to_embed = [query]
    if normalized_query != query:
        queries_to_embed.append(normalized_query)
    query_vectors = embed_texts(queries_to_embed)

    # 2) Parallel semantic (ChromaDB) and lexical (SQLite FTS5) retrieval
    def _run_semantic():
        sem_res = collection.query(
            query_embeddings=query_vectors,
            n_results=min(config.HYBRID_SEMANTIC_CANDIDATES, count),
            include=["documents", "metadatas", "distances"],
        )
        scores: dict[str, float] = {}
        records: dict[str, dict[str, Any]] = {}
        for sub_ids, sub_docs, sub_metas, sub_dists in zip(
            sem_res["ids"], sem_res["documents"], sem_res["metadatas"], sem_res["distances"]
        ):
            for m_id, doc, meta, dist in zip(sub_ids, sub_docs, sub_metas, sub_dists):
                s = max(0.0, min(1.0, 1.0 - float(dist)))
                if m_id not in scores or s > scores[m_id]:
                    scores[m_id] = round(s, 4)
                    records[m_id] = {
                        "id": m_id,
                        "text": doc,
                        "metadata": meta or {},
                    }
        return scores, records

    def _run_lexical():
        return search_lexical_fts(query, limit=config.HYBRID_SEMANTIC_CANDIDATES, expanded_terms=expanded_terms)

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        fut_semantic = executor.submit(_run_semantic)
        fut_lexical = executor.submit(_run_lexical)
        semantic_scores, semantic_records = fut_semantic.result()
        lexical_candidates = fut_lexical.result()

    # 3) Merge candidate pool: candidates retrieved by semantic OR lexical search
    candidate_ids = list(set(semantic_scores.keys()) | set(lexical_candidates.keys()))
    if not candidate_ids:
        return []

    candidates: dict[str, dict[str, Any]] = {}
    with get_fts_connection() as conn:
        placeholders = ",".join("?" for _ in candidate_ids)
        rows = conn.execute(
            f"SELECT * FROM chunk_records WHERE id IN ({placeholders})",
            candidate_ids,
        ).fetchall()
        for row in rows:
            chunk_id = row["id"]
            candidates[chunk_id] = {
                "id": chunk_id,
                "text": row["text"],
                "metadata": {
                    "document_id": row["document_id"],
                    "chunk_index": row["chunk_index"],
                    "source_url": row["source_url"],
                    "source_title": row["source_title"],
                    "source_domain": row["source_domain"],
                    "created_at": row["created_at"],
                    "timestamp": row["timestamp"],
                    "visited_at": row["visited_at"],
                    "raw_text": row["raw_text"],
                    "topics": row["topics"],
                    "tags": row["tags"],
                    "metadata": row["metadata"],
                },
            }

    # For any candidate in semantic_records not yet in FTS table:
    for c_id, s_rec in semantic_records.items():
        if c_id not in candidates:
            candidates[c_id] = s_rec

    query_topics = _query_topics(query)
    matches = []
    for memory_id, record in candidates.items():
        metadata = record["metadata"] or {}
        semantic = semantic_scores.get(memory_id, 0.0)
        lexical_score = lexical_candidates.get(memory_id, 0.0)
        keyword = max(
            lexical_score,
            _keyword_score(
                query,
                record["text"],
                str(metadata.get("source_title", "")),
                normalized_query=normalized_query if normalized_query != query else "",
            ),
        )
        topics = _parse_topics(metadata)
        tags = _parse_dynamic_tags(metadata)
        topic = _topic_score(query_topics, topics)
        tag = _tag_score(query, tags)
        if normalized_query != query and tag == 0.0:
            tag = _tag_score(normalized_query, tags)
        hybrid = _hybrid_score(semantic, keyword, topic, tag)

        if hybrid < min_score:
            continue

        matches.append(
            {
                "id": memory_id,
                "text": record["text"],
                "score": hybrid,
                "semantic_score": semantic,
                "keyword_score": keyword,
                "topic_score": topic,
                "tag_score": tag,
                "source_url": metadata.get("source_url", ""),
                "source_title": metadata.get("source_title", ""),
                "source_domain": metadata.get("source_domain", ""),
                "created_at": metadata.get("created_at", ""),
                "timestamp": metadata.get("timestamp", metadata.get("created_at", "")),
                "visited_at": metadata.get("visited_at", metadata.get("timestamp", metadata.get("created_at", ""))),
                "document_id": metadata.get("document_id", ""),
                "chunk_index": int(metadata.get("chunk_index", 0) or 0),
                "topics": topics,
                "tags": tags,
            }
        )

    matches.sort(
        key=lambda item: (
            item["score"],
            item["semantic_score"],
            item["tag_score"],
            item["keyword_score"],
        ),
        reverse=True,
    )

    # Document-level ranking with dynamic chunk allowance
    source_rows = build_source_results(matches, limit=max(1, top_k))
    selected: list[dict[str, Any]] = []
    max_chunks = max(config.CHUNKS_PER_SOURCE, 4 if len(source_rows) <= 2 else config.CHUNKS_PER_SOURCE)
    for source in source_rows:
        source_key = source["source_key"]
        source_matches = [
            match for match in matches if _source_key(match) == source_key
        ]
        source_matches.sort(
            key=lambda item: (
                item["score"],
                item["semantic_score"],
                item["keyword_score"],
            ),
            reverse=True,
        )
        selected.extend(source_matches[:max_chunks])

    total_search_ms = (time.perf_counter() - t_search_start) * 1000
    logger.info(
        "Structured Timing: search_memories [query_len=%d, dense_candidates=%d, lexical_candidates=%d, merged_candidates=%d, returned_chunks=%d, search_ms=%.2f]",
        len(query), len(semantic_scores), len(lexical_candidates), len(matches), len(selected), total_search_ms,
    )
    return selected


def list_memories() -> list[dict[str, Any]]:
    """Return every stored memory, newest first."""
    collection = get_collection()
    sync_lexical_index(collection)
    with get_fts_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM chunk_records ORDER BY created_at DESC"
        ).fetchall()
        memories = []
        for row in rows:
            meta = {
                "topics": row["topics"],
                "tags": row["tags"],
                "metadata": row["metadata"],
            }
            memories.append(
                {
                    "id": row["id"],
                    "text": row["text"],
                    "source_url": row["source_url"],
                    "source_title": row["source_title"],
                    "source_domain": row["source_domain"],
                    "created_at": row["created_at"],
                    "timestamp": row["timestamp"] or row["created_at"],
                    "visited_at": row["visited_at"] or row["timestamp"] or row["created_at"],
                    "document_id": row["document_id"],
                    "chunk_index": int(row["chunk_index"] or 0),
                    "topics": _parse_topics(meta),
                    "tags": _parse_dynamic_tags(meta),
                }
            )
        if memories:
            return memories

    # Fallback directly to Chroma if SQLite table is empty
    result = collection.get(include=["documents", "metadatas"])
    memories = []
    for memory_id, document, metadata in zip(
        result["ids"],
        result["documents"],
        result["metadatas"],
    ):
        metadata = metadata or {}
        memories.append(
            {
                "id": memory_id,
                "text": document,
                "source_url": metadata.get("source_url", ""),
                "source_title": metadata.get("source_title", ""),
                "source_domain": metadata.get("source_domain", ""),
                "created_at": metadata.get("created_at", ""),
                "timestamp": metadata.get("timestamp", metadata.get("created_at", "")),
                "visited_at": metadata.get("visited_at", metadata.get("timestamp", metadata.get("created_at", ""))),
                "document_id": metadata.get("document_id", ""),
                "chunk_index": int(metadata.get("chunk_index", 0) or 0),
                "topics": _parse_topics(metadata),
                "tags": _parse_dynamic_tags(metadata),
            }
        )
    memories.sort(key=lambda item: item["created_at"], reverse=True)
    return memories


def count_memories() -> int:
    return get_collection().count()


def delete_all_memories() -> int:
    """Wipe the database. Only used by the prototype's --reset flag."""
    collection = get_collection()
    ids = collection.get(include=[])["ids"]
    if ids:
        collection.delete(ids=ids)
    with get_fts_connection() as conn:
        conn.execute("DELETE FROM chunk_records;")
        conn.execute("DELETE FROM chunk_fts;")
    invalidate_vocab_cache()
    return len(ids)

