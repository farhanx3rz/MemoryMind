"""Live phase-1/phase-2 webpage validation for RECALL.

This script deliberately keeps source URLs in the validation layer only.
Production extraction, classification, embeddings, retrieval, and QA remain
source-agnostic.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass

import httpx

from app import config, llm, memory_store, pipeline
from app.webpage import from_html


@dataclass(frozen=True)
class SourceCase:
    name: str
    url: str


PHASE_1_WIKIPEDIA = [
    SourceCase("technology", "https://en.wikipedia.org/wiki/Computer"),
    SourceCase("science", "https://en.wikipedia.org/wiki/DNA"),
    SourceCase("history", "https://en.wikipedia.org/wiki/Industrial_Revolution"),
    SourceCase("programming", "https://en.wikipedia.org/wiki/Python_(programming_language)"),
]

PHASE_2_NON_WIKIPEDIA = [
    SourceCase("non-wikipedia programming", "https://docs.python.org/3/tutorial/"),
]


def fetch(case: SourceCase) -> str:
    response = httpx.get(
        case.url,
        timeout=30,
        follow_redirects=True,
        headers={
            "User-Agent": "MemoryMindValidator/1.0 (https://github.com/memorymind; validator@memorymind.org) httpx/0.28"
        },
    )
    response.raise_for_status()
    response.encoding = response.encoding or "utf-8"
    return response.text


def run(cases: list[SourceCase], persist: bool, question: str | None) -> int:
    stored_ids: list[str] = []
    for case in cases:
        print(f"\n[{case.name}] {case.url}")
        try:
            html = fetch(case)
            document = from_html(url=case.url, html=html)
            print(f"  extracted chars : {len(document.extracted_text):,}")
            print(f"  chunks          : {len(document.chunks)}")
            print(f"  domain          : {document.source}")

            document = pipeline.classify_document(document)
            print(f"  topics          : {[tag['label'] for tag in document.topics]}")

            # Embedding verification without changing the persistent store.
            vector = memory_store.embed_texts([document.chunks[0]])[0]
            print(f"  embedding dims  : {len(vector)}")

            if persist:
                memory_id = pipeline.store_document(document)
                stored_ids.append(memory_id)
                print(f"  stored document : {memory_id}")
        except Exception as error:
            print(f"  FAILED          : {type(error).__name__}: {error}")
            return 1

    if persist and question:
        print(f"\nRetrieval/QA check: {question}")
        matches = memory_store.search_memories(question)
        print(f"  retrieved chunks: {len(matches)}")
        for index, match in enumerate(matches, 1):
            print(
                f"  {index}. {match['source_domain']} "
                f"score={match['score']} chunk={match['chunk_index']}"
            )
            print(f"     URL: {match['source_url']}")
        if matches:
            answer = llm.generate_answer(question, matches)
            print(f"  answer          : {answer}")
        else:
            print("  answer          : no relevant stored memory")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the generic RECALL webpage pipeline.")
    parser.add_argument("--all", action="store_true", help="Run Wikipedia phase 1 and non-Wikipedia phase 2.")
    parser.add_argument("--persist", action="store_true", help="Store validated documents in ChromaDB.")
    parser.add_argument("--question", type=str, default=None, help="Run retrieval + QA after --persist.")
    args = parser.parse_args()

    cases = list(PHASE_1_WIKIPEDIA)
    if args.all:
        cases.extend(PHASE_2_NON_WIKIPEDIA)

    if args.question and not args.persist:
        parser.error("--question requires --persist")
    return run(cases, persist=args.persist, question=args.question)


if __name__ == "__main__":
    sys.exit(main())
