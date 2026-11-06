"""Benchmark baseline timings for Remember-Webpage and Answer-Question workflows."""

import time
import json
from pathlib import Path
from app import webpage, pipeline, memory_store, llm, config

def benchmark_remember(url: str, html: str | None = None, extracted_text: str | None = None):
    timings = {}
    t_start = time.perf_counter()

    # 1. URL validation
    t0 = time.perf_counter()
    valid_url = webpage.validate_source_url(url)
    timings["url_validation_ms"] = round((time.perf_counter() - t0) * 1000, 2)

    # 2. Content extraction & chunking
    t0 = time.perf_counter()
    if html:
        doc = webpage.from_html(url=valid_url, html=html)
    else:
        doc = webpage.from_extracted_text(url=valid_url, text=extracted_text or "Sample text " * 100)
    timings["extraction_chunking_ms"] = round((time.perf_counter() - t0) * 1000, 2)
    timings["chunk_count"] = len(doc.chunks)
    timings["char_count"] = len(doc.extracted_text)

    # 3. Topic classification
    t0 = time.perf_counter()
    doc = pipeline.classify_document(doc)
    timings["classification_ms"] = round((time.perf_counter() - t0) * 1000, 2)

    # 4. Dynamic tags (LLM)
    t0 = time.perf_counter()
    # Mock or time dynamic tags
    if config.DYNAMIC_TAGS_ENABLED and config.LLM_PROVIDER != "none":
        try:
            doc = pipeline.extract_dynamic_tags(doc)
        except Exception:
            pass
    timings["dynamic_tags_ms"] = round((time.perf_counter() - t0) * 1000, 2)

    # 5. Embedding generation
    t0 = time.perf_counter()
    vectors = memory_store.embed_texts(doc.chunks)
    timings["embedding_generation_ms"] = round((time.perf_counter() - t0) * 1000, 2)

    # 6. ChromaDB + SQLite FTS storage
    t0 = time.perf_counter()
    memory_id = pipeline.store_document(doc)
    timings["storage_ms"] = round((time.perf_counter() - t0) * 1000, 2)

    timings["total_remember_ms"] = round((time.perf_counter() - t_start) * 1000, 2)
    return timings, memory_id

def benchmark_ask(question: str):
    timings = {}
    t_start = time.perf_counter()

    # 1. Query normalization & embedding
    t0 = time.perf_counter()
    q_vec = memory_store.embed_texts([question])[0]
    timings["query_embedding_ms"] = round((time.perf_counter() - t0) * 1000, 2)

    # 2. Semantic retrieval
    t0 = time.perf_counter()
    coll = memory_store.get_collection()
    count = coll.count()
    sem_res = coll.query(
        query_embeddings=[q_vec],
        n_results=min(config.HYBRID_SEMANTIC_CANDIDATES, count),
        include=["documents", "metadatas", "distances"],
    )
    timings["semantic_retrieval_ms"] = round((time.perf_counter() - t0) * 1000, 2)
    timings["semantic_candidates_count"] = len(sem_res["ids"][0]) if sem_res["ids"] else 0

    # 3. SQLite FTS sync & retrieval
    t0 = time.perf_counter()
    memory_store.sync_lexical_index(coll)
    timings["fts_sync_ms"] = round((time.perf_counter() - t0) * 1000, 2)

    t0 = time.perf_counter()
    lex_res = memory_store.search_lexical_fts(question)
    timings["lexical_retrieval_ms"] = round((time.perf_counter() - t0) * 1000, 2)
    timings["lexical_candidates_count"] = len(lex_res)

    # 4. Full search_memories end-to-end
    t0 = time.perf_counter()
    matches = memory_store.search_memories(question)
    timings["hybrid_search_total_ms"] = round((time.perf_counter() - t0) * 1000, 2)
    timings["matches_count"] = len(matches)

    # 5. Prompt construction
    t0 = time.perf_counter()
    user_prompt = llm.build_user_prompt(question, matches)
    timings["prompt_construction_ms"] = round((time.perf_counter() - t0) * 1000, 2)
    timings["prompt_char_count"] = len(user_prompt)

    timings["total_ask_pre_llm_ms"] = round((time.perf_counter() - t_start) * 1000, 2)
    return timings

if __name__ == "__main__":
    import httpx
    print("Running baseline benchmarks...")
    
    # Measure Ask on existing memories
    q = "elon must companies"
    ask_timings = benchmark_ask(q)
    print("\n--- BASELINE ASK TIMINGS ---")
    for k, v in ask_timings.items():
        print(f"  {k:30}: {v}")

    # Warm repeated runs for Ask
    ask_runs = []
    for _ in range(5):
        t = benchmark_ask("elon must companies")
        ask_runs.append(t["hybrid_search_total_ms"])

    ask_runs.sort()
    med_ask = ask_runs[len(ask_runs) // 2]
    p95_ask = ask_runs[int(len(ask_runs) * 0.95)]
    # Measure Remember on a standard text sample
    sample_text = (
        "Elon Musk is the CEO of Tesla and SpaceX. He founded The Boring Company, xAI, and Neuralink. "
        "He previously co-founded PayPal and Zip2. In 2022, he acquired Twitter for 44 billion dollars. "
    ) * 10
    
    # We test with persist=False inside benchmark or measure stages directly
    rem_timings, _ = benchmark_remember("https://en.wikipedia.org/wiki/Elon_Musk", extracted_text=sample_text)
    print("\n--- BASELINE REMEMBER TIMINGS ---")
    for k, v in rem_timings.items():
        print(f"  {k:30}: {v}")
