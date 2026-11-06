"""
Benchmark the webpage ingestion pipeline stage-by-stage.
Run from backend/ with: python benchmark_pipeline.py
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

SAMPLE_URL = "https://example.com/vector-databases"
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
    print("MemoryMind Pipeline Benchmark (BEFORE optimizations)")
    print("=" * 72)

    # Stage 1: HTML extraction + chunking
    print("\n[Stage 1] HTML extraction + chunking")
    from app.webpage import from_html
    doc, extract_times = bench("from_html()", from_html, url=SAMPLE_URL, html=SAMPLE_HTML)
    print(f"  => {len(doc.chunks)} chunks produced, text length={len(doc.extracted_text)}")

    # Stage 2: Topic classification (local DistilBERT)
    print("\n[Stage 2] Topic classification (local DistilBERT)")
    from app.pipeline import classify_document
    classified_doc, classify_times = bench("classify_document()", classify_document, copy.deepcopy(doc))
    print(f"  => topics={classified_doc.topics}")

    # Stage 3: Dynamic tag extraction (external LLM)
    print("\n[Stage 3] Dynamic tag extraction (external LLM)")
    from app import config
    if config.LLM_PROVIDER == "none":
        print("  [SKIP] LLM_PROVIDER=none -- tags empty, cost=0ms")
        tag_times = [0.0] * REPS
    else:
        from app.pipeline import extract_dynamic_tags
        tagged_doc, tag_times = bench("extract_dynamic_tags()", extract_dynamic_tags, copy.deepcopy(doc))
        print(f"  => tags={[t['label'] for t in tagged_doc.tags]}")

    # Stage 4: Embedding generation
    print("\n[Stage 4] Embedding generation")
    from app.memory_store import embed_texts
    _, embed_times = bench(f"embed_texts({len(doc.chunks)} chunks)", embed_texts, doc.chunks)

    # Stage 5: ChromaDB + FTS store (each call uses a fresh uuid to avoid dedup)
    print("\n[Stage 5] ChromaDB + FTS store")
    from app.memory_store import add_web_document

    def do_store():
        d = copy.deepcopy(doc)
        d.document_id = str(uuid.uuid4())
        d.topics = classified_doc.topics
        d.tags = []
        return add_web_document(d)

    _, store_times = bench("add_web_document()", do_store)

    # Stage 6: Full ingest_document (current: classify || tags concurrently, then store serially)
    print("\n[Stage 6] Full ingest_document() -- current implementation")
    from app.pipeline import ingest_document

    def do_ingest():
        d = copy.deepcopy(doc)
        d.document_id = str(uuid.uuid4())
        return ingest_document(d)

    _, ingest_times = bench("ingest_document()", do_ingest)

    # --- Summary ---
    def med(times):
        return statistics.median(times)

    def p95(times):
        s = sorted(times)
        return s[max(0, int(len(s) * 0.95) - 1)]

    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"  {'Stage':<38} {'Median':>10} {'P95':>10}")
    print(f"  {'-'*38} {'-'*10} {'-'*10}")
    print(f"  {'1. Extraction + chunking':<38} {med(extract_times):>9.1f}ms {p95(extract_times):>9.1f}ms")
    print(f"  {'2. Classification (DistilBERT)':<38} {med(classify_times):>9.1f}ms {p95(classify_times):>9.1f}ms")
    print(f"  {'3. Dynamic tags (LLM)':<38} {med(tag_times):>9.1f}ms {p95(tag_times):>9.1f}ms")
    print(f"  {'4. Embedding':<38} {med(embed_times):>9.1f}ms {p95(embed_times):>9.1f}ms")
    print(f"  {'5. DB store (Chroma + FTS)':<38} {med(store_times):>9.1f}ms {p95(store_times):>9.1f}ms")
    print(f"  {'6. TOTAL ingest_document()':<38} {med(ingest_times):>9.1f}ms {p95(ingest_times):>9.1f}ms")
    print()

    breakdown = {
        "Classification": med(classify_times),
        "Dynamic tags (LLM)": med(tag_times),
        "Embedding": med(embed_times),
        "DB store": med(store_times),
    }
    bottleneck = max(breakdown, key=breakdown.get)
    print(f"  Dominant bottleneck: {bottleneck} ({breakdown[bottleneck]:.0f}ms median)")
    print()


if __name__ == "__main__":
    main()
