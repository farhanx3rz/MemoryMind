"""Backfill Grok dynamic tags onto stored documents that are missing them.

Handles two categories of documents:
  1. Legacy records without a `topics` field (old format before classifier was added).
  2. Modern records that have `topics` but still have empty tags (`tags = '[]'` or '')
     because the background LLM job failed, timed out, or was never submitted.

This means every saved webpage with missing dynamic tags gets them filled in,
not just the oldest format records.

Usage:
    python -m scripts.backfill_dynamic_tags
    python -m scripts.backfill_dynamic_tags --dry-run
    python -m scripts.backfill_dynamic_tags --limit 20
    python -m scripts.backfill_dynamic_tags --force   # re-tag even docs that already have tags

The script uses memory_store.update_document_tags() to write tags to SQLite first
(authoritative) and then patches ChromaDB, matching the production pipeline.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict

from app import config, llm, memory_store


def _tags_empty(tags_value: str | None) -> bool:
    """Return True when a tags field value represents an empty tag list."""
    raw = (tags_value or "").strip()
    if not raw or raw == "[]":
        return True
    try:
        parsed = json.loads(raw)
        return not (isinstance(parsed, list) and parsed)
    except Exception:
        return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backfill dynamic Grok tags for all documents with empty tags"
    )
    parser.add_argument("--dry-run", action="store_true", help="Show tags without writing them")
    parser.add_argument("--limit", type=int, default=0, help="Max parent documents to process")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-tag documents that already have tags (overwrite existing)",
    )
    parser.add_argument(
        "--text-limit",
        type=int,
        default=20000,
        help="Max characters of document text to send to the LLM (default: 20000)",
    )
    args = parser.parse_args()

    if config.LLM_PROVIDER == "none":
        raise SystemExit(
            "LLM_PROVIDER is set to 'none'. Configure a real LLM provider before backfilling tags."
        )

    # ── Step 1: Gather documents from SQLite (the authoritative store) ──────
    with memory_store.get_fts_connection() as conn:
        rows = conn.execute(
            "SELECT id, document_id, source_url, source_title, topics, tags, text "
            "FROM chunk_records ORDER BY document_id, chunk_index"
        ).fetchall()

    if not rows:
        print("No chunk_records found. Nothing to do.")
        return

    # Group chunks by document_id
    grouped: dict[str, list] = defaultdict(list)
    doc_info: dict[str, dict] = {}

    for row in rows:
        doc_id = row["document_id"]
        grouped[doc_id].append(row)
        if doc_id not in doc_info:
            doc_info[doc_id] = {
                "url": row["source_url"],
                "title": row["source_title"],
                "tags": row["tags"],
                "topics": row["topics"],
            }

    # ── Step 2: Filter to documents that need tags ───────────────────────────
    needs_tags: list[str] = []
    for doc_id, info in doc_info.items():
        if args.force or _tags_empty(info["tags"]):
            needs_tags.append(doc_id)

    if args.limit > 0:
        needs_tags = needs_tags[: args.limit]

    total = len(needs_tags)
    already_tagged = len(doc_info) - total
    print(
        f"Documents in database: {len(doc_info)}  |  "
        f"Already tagged: {already_tagged}  |  "
        f"Need backfill: {total}"
    )

    if total == 0:
        print("All documents already have dynamic tags. Use --force to re-tag them.")
        return

    # ── Step 3: Extract and write tags for each document ────────────────────
    success = 0
    failed = 0

    for index, doc_id in enumerate(needs_tags, start=1):
        chunks = grouped[doc_id]
        # Concatenate chunk texts in order to form a representative article view.
        # Cap at args.text_limit chars to avoid extremely long prompts.
        text = " ".join(chunk["text"] or "" for chunk in chunks)
        text = text[: args.text_limit]

        info = doc_info[doc_id]
        title = info["title"] or ""
        url = info["url"] or ""

        print(
            f"[{index}/{total}] {doc_id[:8]}...  {title[:50]}  ({url[:60]})",
            end="  ",
            flush=True,
        )

        try:
            tags = llm.extract_dynamic_tags(text)
        except Exception as exc:
            print(f"FAILED: {exc}")
            failed += 1
            continue

        if not tags:
            print("→ No tags returned (text too short or LLM returned empty)")
            failed += 1
            continue

        labels = [t["label"] for t in tags]
        print(f"-> {labels}")

        if not args.dry_run:
            # update_document_tags writes to SQLite (authoritative) and patches ChromaDB.
            updated = memory_store.update_document_tags(doc_id, tags)
            if updated == 0:
                # Document exists in SQLite but update matched no rows — unusual.
                print(f"  WARNING: update_document_tags wrote 0 rows for {doc_id}")
            success += 1
        else:
            success += 1  # count as success in dry-run mode

    # ── Step 4: Report ───────────────────────────────────────────────────────
    mode = "[DRY RUN] " if args.dry_run else ""
    print(
        f"\n{mode}Backfill complete: {success} succeeded, {failed} failed "
        f"out of {total} documents."
    )
    if args.dry_run:
        print("Run without --dry-run to write the tags.")


if __name__ == "__main__":
    main()
