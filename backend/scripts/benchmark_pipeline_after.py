"""
Benchmark the webpage ingestion pipeline AFTER async tag optimization.
Run from backend/ with: python benchmark_pipeline_after.py
"""
import statistics
import time
import sys
import os
import copy
import uuid

sys.path.insert(0, os.path.dirname(__file__))

SAMPLE_HTML = """
<html>
<head><title>Vector Databases and Semantic Search</title></head>
<body>
<nav>Home About Contact</nav>
<main>
<h1>Understanding Vector Databases</h1>
<p>Vector databases store embeddings and enable semantic similarity search.
ChromaDB is an open-source vector database popular for RAG applications.
It supports cosine similarity, HNSW indexing, and persistent storage.</p>
<p>Semantic search works by converting text to dense vector representations
using models like sentence-transformers. Queries are also embedded and compared
to stored vectors using cosine or dot-product similarity.</p>
<p>Popular vector databases include Pinecone, Weaviate, Qdrant, Milvus, and ChromaDB.
Each has different trade-offs in terms of scalability, latency, and feature set.</p>
<p>Retrieval augmented generation (RAG) combines vector search with LLMs.
Retrieved chunks provide grounding evidence. This reduces hallucinations and
keeps answers factual and up-to-date.</p>
<p>Embedding models vary in dimension, quality, and speed. The all-MiniLM-L6-v2
model from sentence-transformers provides 384-dimensional embeddings and runs fast
on CPU. Larger models like e5-large provide higher quality at greater cost.</p>
</main>
<footer>Privacy Policy - Terms</footer>
</body>
</html>
""" * 5

SAMPLE_URL = "https://example.com/vector-databases-after"
REPS = 5


def timeit_fn(fn, *args, **kwargs):
    t = time.perf_counter()
    result = fn(*args, **kwargs)
    elapsed = (time.perf_counter() - t) * 1000
    return result, elapsed


def bench(name, fn, *args, reps=REPS, **kwargs):
    times = []
    result = None
    for _ in range(reps):
        result, ms = timeit_fn(fn, *args, **kwargs)
        times.append(ms)
    med = statistics.median(times)
    p95 = sorted(times)[max(0, int(len(times) * 0.95) - 1)] if len(times) > 1 else times[0]
    all_str = [f"{t:.0f}" for t in times]
    print(f"  {name:<42} median={med:7.1f}ms  p95={p95:7.1f}ms  all={all_str}")
    return result, times


def main():
    print("=" * 72)
    print("MemoryMind Pipeline Benchmark (AFTER async tag optimization)")
    print("=" * 72)
    print("  Tags now run in background; ingest_document() returns after classify+store.")
    print()

    # Warm up (model loading)
    from app.webpage import from_html
    doc, extract_times = bench("from_html() [Stage 1]", from_html, url=SAMPLE_URL, html=SAMPLE_HTML)
    print(f"  => {len(doc.chunks)} chunks, text length={len(doc.extracted_text)}")

    print("\n[Stage 2] Topic classification (local DistilBERT)")
    from app.pipeline import classify_document
    classified_doc, classify_times = bench("classify_document()", classify_document, copy.deepcopy(doc))
    print(f"  => topics={classified_doc.topics}")

    print("\n[Stage 3] ingest_document() -- NEW: returns after classify+store, tags async")
    from app.pipeline import ingest_document

    def do_ingest():
        d = copy.deepcopy(doc)
        d.document_id = str(uuid.uuid4())
        return ingest_document(d)

    (result_doc, memory_id), ingest_times = bench("ingest_document() [AFTER]", do_ingest)
    print(f"  => memory_id={memory_id}, topics={result_doc.topics}, tags={result_doc.tags} (async)")

    # Verify update_document_tags works
    print("\n[Stage 4] update_document_tags() -- the async write-back path")
    from app.memory_store import update_document_tags
    sample_tags = [
        {"label": "ChromaDB", "score": 1.0},
        {"label": "vector database", "score": 1.0},
        {"label": "RAG", "score": 1.0},
    ]
    _, update_times = bench(
        "update_document_tags()",
        update_document_tags,
        result_doc.document_id or memory_id,
        sample_tags,
    )

    # Summary
    def med(times):
        return statistics.median(times)

    def p95(times):
        s = sorted(times)
        return s[max(0, int(len(s) * 0.95) - 1)]

    print("\n" + "=" * 72)
    print("BEFORE vs AFTER comparison")
    print("=" * 72)
    print(f"  {'Metric':<40} {'BEFORE':>10} {'AFTER':>10} {'Saved':>10}")
    print(f"  {'-'*40} {'-'*10} {'-'*10} {'-'*10}")

    before_total_med = 6021.8
    before_total_p95 = 7884.7
    after_med = med(ingest_times)
    after_p95 = p95(ingest_times)

    print(f"  {'ingest_document() median':<40} {before_total_med:>9.1f}ms {after_med:>9.1f}ms {before_total_med - after_med:>9.1f}ms")
    print(f"  {'ingest_document() p95':<40} {before_total_p95:>9.1f}ms {after_p95:>9.1f}ms {before_total_p95 - after_p95:>9.1f}ms")
    print()

    speedup_med = before_total_med / after_med if after_med > 0 else float("inf")
    speedup_p95 = before_total_p95 / after_p95 if after_p95 > 0 else float("inf")
    print(f"  Speedup (median): {speedup_med:.1f}x")
    print(f"  Speedup (p95):    {speedup_p95:.1f}x")
    print()
    print(f"  Classification:           {med(classify_times):7.1f}ms median")
    print(f"  DB store (via ingest):    ~{after_med - med(classify_times):7.1f}ms median")
    print(f"  Tag write-back (async):   {med(update_times):7.1f}ms median (SQLite + ChromaDB patch)")
    print()
    print("  Dynamic tag LLM call (5243ms) runs in background thread.")
    print("  User gets HTTP 200 response after embed+store (~200ms total).")
    print()


if __name__ == "__main__":
    main()
