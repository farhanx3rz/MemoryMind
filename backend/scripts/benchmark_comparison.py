"""Benchmark comparisons: Before vs After optimizations for MemoryMind RECALL."""
from __future__ import annotations

import sys
import time
import statistics
from unittest.mock import patch

from app import config, llm, memory_store, pipeline, webpage

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def run_ask_benchmark(question: str, num_runs: int = 10):
    print(f"\n--- BENCHMARKING ASK WORKFLOW: '{question}' ({num_runs} retrieval runs, 3 full LLM runs) ---")
    
    # 1. Warm search runs
    search_latencies = []
    matches = []
    for i in range(num_runs):
        t0 = time.perf_counter()
        matches = memory_store.search_memories(question, top_k=config.TOP_K)
        search_latencies.append((time.perf_counter() - t0) * 1000)

    search_latencies.sort()
    med_search = statistics.median(search_latencies)
    idx_p95 = min(int(len(search_latencies) * 0.95), len(search_latencies) - 1)
    p95_search = search_latencies[idx_p95]

    print(f"  Warm Retrieval Latency ({num_runs} runs):")
    print(f"    Min:    {min(search_latencies):.2f} ms")
    print(f"    Median: {med_search:.2f} ms")
    print(f"    p95:    {p95_search:.2f} ms")

    # 2. Time to First Token (TTFT)
    ttft_ms = 0.0
    tokens_streamed = []
    t_stream_start = time.perf_counter()
    for token in llm.stream_answer_tokens(question, matches):
        if not tokens_streamed:
            ttft_ms = (time.perf_counter() - t_stream_start) * 1000
        tokens_streamed.append(token)
    stream_total_ms = (time.perf_counter() - t_stream_start) * 1000
    print(f"  Streaming Time-To-First-Token (TTFT): {ttft_ms:.2f} ms")
    print(f"  Streaming Total Duration:             {stream_total_ms:.2f} ms")

    # 3. Repeated full LLM runs
    llm_latencies = []
    full_latencies = []
    last_answer = ""
    last_citations = []
    for _ in range(3):
        t_req_start = time.perf_counter()
        m = memory_store.search_memories(question, top_k=config.TOP_K)
        t_search = (time.perf_counter() - t_req_start) * 1000
        
        t_llm_start = time.perf_counter()
        last_answer, last_citations = llm.generate_answer_with_citations(question, m)
        t_llm = (time.perf_counter() - t_llm_start) * 1000
        
        llm_latencies.append(t_llm)
        full_latencies.append(t_search + t_llm)

    llm_latencies.sort()
    full_latencies.sort()
    med_llm = statistics.median(llm_latencies)
    p95_llm = full_latencies[min(int(len(full_latencies) * 0.95), len(full_latencies) - 1)]
    med_full = statistics.median(full_latencies)

    print(f"  LLM Generation Latency (separate from retrieval):")
    print(f"    Median: {med_llm:.2f} ms")
    print(f"  Full Answer Request Latency (Retrieval + LLM):")
    print(f"    Median: {med_full:.2f} ms")
    print(f"    p95:    {p95_llm:.2f} ms")
    print(f"  Candidates Retrieved: {len(matches)}")
    print(f"  Cited Sources:        {last_citations}")
    print(f"\nAnswer Output Preview:\n{last_answer[:280]}...\n")

    return {
        "question": question,
        "warm_search_median_ms": round(med_search, 2),
        "warm_search_p95_ms": round(p95_search, 2),
        "ttft_ms": round(ttft_ms, 2),
        "stream_total_ms": round(stream_total_ms, 2),
        "llm_median_ms": round(med_llm, 2),
        "full_answer_median_ms": round(med_full, 2),
        "full_answer_p95_ms": round(p95_llm, 2),
        "citations": last_citations,
    }


def run_remember_benchmark(num_runs: int = 5):
    print(f"\n--- BENCHMARKING REMEMBER WORKFLOW ({num_runs} runs) ---")
    
    sample_text = (
        "Elon Musk is a technology entrepreneur, investor, and engineer. "
        "He is the founder, CEO, and chief engineer of SpaceX; angel investor, CEO and product architect of Tesla, Inc.; "
        "founder of The Boring Company; co-founder of Neuralink and OpenAI; and founder of xAI. "
        "Musk previously co-founded Zip2 and PayPal (formerly X.com). "
    ) * 8

    # Stage breakdowns
    t0 = time.perf_counter()
    valid_url = webpage.validate_source_url("https://en.wikipedia.org/wiki/Elon_Musk")
    url_val_ms = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    doc = webpage.from_extracted_text(url=valid_url, title="Elon Musk", text=sample_text)
    extract_chunk_ms = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    doc = pipeline.classify_document(doc)
    classify_ms = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    vectors = memory_store.embed_texts(doc.chunks)
    embed_ms = (time.perf_counter() - t0) * 1000

    # Measure concurrent pipeline (classify + tags)
    pipeline_runs = []
    for _ in range(num_runs):
        test_doc = webpage.from_extracted_text(url=valid_url, title="Elon Musk", text=sample_text)
        t0 = time.perf_counter()
        with patch.object(pipeline, "store_document", return_value="test_id"):
            ingested, _ = pipeline.ingest_document(test_doc)
        pipeline_runs.append((time.perf_counter() - t0) * 1000)

    pipeline_runs.sort()
    med_pipeline = statistics.median(pipeline_runs)
    idx_p95 = min(int(len(pipeline_runs) * 0.95), len(pipeline_runs) - 1)
    p95_pipeline = pipeline_runs[idx_p95]

    print(f"  URL Validation:            {url_val_ms:.2f} ms")
    print(f"  Extraction & Chunking:     {extract_chunk_ms:.2f} ms ({len(doc.chunks)} chunks, {len(doc.extracted_text)} chars)")
    print(f"  Topic Classification:      {classify_ms:.2f} ms ({len(doc.topics)} topics)")
    print(f"  Embedding Generation:      {embed_ms:.2f} ms")
    print(f"  Concurrent Ingest Pipeline (classify + tag extraction):")
    print(f"    Min:    {min(pipeline_runs):.2f} ms")
    print(f"    Median: {med_pipeline:.2f} ms")
    print(f"    p95:    {p95_pipeline:.2f} ms")

    return {
        "url_validation_ms": round(url_val_ms, 2),
        "extraction_chunking_ms": round(extract_chunk_ms, 2),
        "classification_ms": round(classify_ms, 2),
        "embedding_ms": round(embed_ms, 2),
        "pipeline_median_ms": round(med_pipeline, 2),
        "pipeline_p95_ms": round(p95_pipeline, 2),
    }


if __name__ == "__main__":
    print("=" * 60)
    print("MEMORYMIND OPTIMIZATION BENCHMARK REPORT")
    print("=" * 60)
    
    ask_results = run_ask_benchmark("elon must companies", num_runs=10)
    rem_results = run_remember_benchmark(num_runs=5)

    print("\n" + "=" * 60)
    print("BENCHMARK COMPLETED SUCCESSFULLY")
    print("=" * 60)
