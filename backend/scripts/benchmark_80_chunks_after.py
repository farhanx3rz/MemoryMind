"""
Benchmark script measuring timings AFTER optimizations on the exact 80-chunk Al Pacino input.
Run from backend/ with: python scripts/benchmark_80_chunks_after.py
"""
import sys, os
sys.path.insert(0, os.getcwd())
import sqlite3
import time
import copy
import uuid
from pathlib import Path

# Load 80 chunks from the database
db_path = Path("data/chroma/lexical_fts.sqlite3")
conn = sqlite3.connect(db_path)
c = conn.cursor()
c.execute("SELECT chunk_index, text FROM chunk_records WHERE document_id = ? ORDER BY chunk_index", ('a56146b3-697e-44be-b555-64c12cd010f9',))
rows = c.fetchall()
chunks = [r[1] for r in rows]
conn.close()

if len(chunks) != 80:
    raise RuntimeError(f"Expected 80 chunks, got {len(chunks)}")

print(f"Loaded {len(chunks)} chunks from document a56146b3-697e-44be-b555-64c12cd010f9")

from app.webpage import WebDocument

def make_doc():
    return WebDocument(
        url="https://en.wikipedia.org/wiki/Al_Pacino",
        title="Al Pacino - Wikipedia",
        source="en.wikipedia.org",
        extracted_text=" ".join(chunks),
        chunks=list(chunks),
        document_id=str(uuid.uuid4()),
        timestamp="2026-10-26T03:00:00Z",
    )

from ml.classifier import TopicClassifier
from app import memory_store, pipeline

print("\n--- CLASSIFIER BENCHMARK (AFTER OPTIMIZATION) ---")
classifier = TopicClassifier()
t0 = time.perf_counter()
classifier._load()
load_ms = (time.perf_counter() - t0) * 1000

t0 = time.perf_counter()
warm_tags = classifier.predict_chunks_to_tags(chunks)
warm_infer_ms = (time.perf_counter() - t0) * 1000
print(f"Warm Classifier Infer (sampled): {warm_infer_ms:7.2f} ms  -> {warm_tags}")

print("\n--- EMBEDDING BENCHMARK (AFTER OPTIMIZATION) ---")
# 1. Warm inference on fresh chunks (using batch_size=64)
# Clear cache first to test fresh 80 chunks
memory_store._embedding_cache.clear()
t0 = time.perf_counter()
vecs_fresh = memory_store.embed_texts(chunks)
fresh_embed_ms = (time.perf_counter() - t0) * 1000
print(f"Warm Fresh 80 Chunks Embed:     {fresh_embed_ms:7.2f} ms")

# 2. Warm cached inference (repeated chunks/page)
t0 = time.perf_counter()
vecs_cached = memory_store.embed_texts(chunks)
cached_embed_ms = (time.perf_counter() - t0) * 1000
print(f"Warm Cached 80 Chunks Embed:    {cached_embed_ms:7.2f} ms")

print("\n--- FULL PIPELINE RUN (classify + embed + store) ---")
# Run 1: with cached embeddings
t0 = time.perf_counter()
test_doc = make_doc()
doc, mem_id = pipeline.ingest_document(test_doc)
pipe_cached_ms = (time.perf_counter() - t0) * 1000
print(f"Pipeline ingest_document() (cached): {pipe_cached_ms:7.2f} ms (topics={doc.topics})")

# Run 2: with fresh embeddings (clear cache)
memory_store._embedding_cache.clear()
t0 = time.perf_counter()
test_doc = make_doc()
doc, mem_id = pipeline.ingest_document(test_doc)
pipe_fresh_ms = (time.perf_counter() - t0) * 1000
print(f"Pipeline ingest_document() (fresh):  {pipe_fresh_ms:7.2f} ms (topics={doc.topics})")
