"""
Benchmark script measuring cold and warm runs on the exact 80-chunk Al Pacino input.
Run from backend/ with: python scripts/benchmark_80_chunks.py
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

print("\n--- CLASSIFIER BENCHMARK (80 chunks) ---")
# Cold run
cold_classifier = TopicClassifier()
t0 = time.perf_counter()
cold_classifier._load()
cold_load_ms = (time.perf_counter() - t0) * 1000

t0 = time.perf_counter()
cold_tags = cold_classifier.predict_chunks_to_tags(chunks)
cold_infer_ms = (time.perf_counter() - t0) * 1000
print(f"Cold Classifier Load:     {cold_load_ms:7.2f} ms")
print(f"Cold Classifier Infer:    {cold_infer_ms:7.2f} ms")
print(f"Cold Total Classify:      {cold_load_ms + cold_infer_ms:7.2f} ms  -> {cold_tags}")

# Warm run
t0 = time.perf_counter()
warm_tags = cold_classifier.predict_chunks_to_tags(chunks)
warm_infer_ms = (time.perf_counter() - t0) * 1000
print(f"Warm Classifier Infer:    {warm_infer_ms:7.2f} ms  -> {warm_tags}")

print("\n--- EMBEDDING BENCHMARK (80 chunks) ---")
from sentence_transformers import SentenceTransformer
from app import config

# Cold run
t0 = time.perf_counter()
cold_embed_model = SentenceTransformer(config.EMBEDDING_MODEL)
cold_embed_load_ms = (time.perf_counter() - t0) * 1000

t0 = time.perf_counter()
cold_vectors = cold_embed_model.encode(chunks, batch_size=32, show_progress_bar=False)
cold_embed_infer_ms = (time.perf_counter() - t0) * 1000
print(f"Cold Embedding Model Load: {cold_embed_load_ms:7.2f} ms")
print(f"Cold Embedding Infer:      {cold_embed_infer_ms:7.2f} ms")
print(f"Cold Total Embed:          {cold_embed_load_ms + cold_embed_infer_ms:7.2f} ms")

# Warm run
t0 = time.perf_counter()
warm_vectors = cold_embed_model.encode(chunks, batch_size=32, show_progress_bar=False)
warm_embed_infer_ms = (time.perf_counter() - t0) * 1000
print(f"Warm Embedding Infer:      {warm_embed_infer_ms:7.2f} ms")

print("\n--- FULL PIPELINE WARM RUN (classify + embed + store) ---")
t0 = time.perf_counter()
test_doc = make_doc()
doc, mem_id = pipeline.ingest_document(test_doc)
pipe_ms = (time.perf_counter() - t0) * 1000
print(f"Pipeline ingest_document() took: {pipe_ms:7.2f} ms (topics={doc.topics})")
