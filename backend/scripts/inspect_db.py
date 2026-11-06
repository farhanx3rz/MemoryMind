import sqlite3

conn = sqlite3.connect('backend/data/chroma/lexical_fts.sqlite3')
conn.row_factory = sqlite3.Row
rows = conn.execute("SELECT id, document_id, source_url, source_title, tags FROM chunk_records").fetchall()
print(f"Total chunks: {len(rows)}")
docs = {}
for r in rows:
    doc_id = r["document_id"]
    if doc_id not in docs:
        docs[doc_id] = {
            "doc_id": doc_id,
            "url": r["source_url"],
            "title": r["source_title"],
            "tags": r["tags"],
            "chunk_count": 0,
        }
    docs[doc_id]["chunk_count"] += 1

for doc_id, info in docs.items():
    print(info)
