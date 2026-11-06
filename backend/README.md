# MemoryMind backend

## Generic webpage pipeline

RECALL is designed around a source-neutral webpage abstraction. Wikipedia is
only the first controlled validation source; it is not a special case.

```text
Webpage
  -> Content Extraction
  -> WebDocument
  -> Clean Article/Text
  -> Chunking
  -> Topic Classification (DistilBERT)
  -> Dynamic Tags (Configured LLM, 3-8 concepts)
  -> Local Embeddings
  -> ChromaDB
  -> Hybrid Retrieval (semantic + keyword + topic + tag)
  -> Document/source grouping
  -> Configured LLM source-grouped RAG
  -> Answer + source URL(s) + read times
```

`app/webpage.py` defines the core `WebDocument` representation:

- `url`
- `title`
- `source` / domain
- `extracted_text`
- `chunks`
- `timestamp`
- `topics`
- `metadata`

Content extraction is independent from the ML models. Trafilatura is the
primary extractor, with a small BeautifulSoup fallback. Trafilatura's standard
extraction cascade is designed to identify main text while reducing boilerplate
and can use readability/jusText fallbacks.

The API accepts either complete HTML or already-extracted text:

```text
POST /webpage
{
  "url": "https://example.com/article",
  "title": "Article title",
  "html": "<html>...</html>"
}
```

or:

```json
{
  "url": "https://example.com/article",
  "title": "Article title",
  "extracted_text": "Main article text..."
}
```

The same pipeline handles Wikipedia, blogs, documentation, news articles,
tutorials, technical articles, educational sites, and ordinary webpages.
No classifier, embedding, vector-store, retrieval, or QA code receives a
Wikipedia-specific branch.

## Store semantics

Full-page ingestion stores one Chroma record per chunk. Every chunk carries:

- source URL
- page title
- source domain
- timestamp
- parent document id
- chunk index
- DistilBERT topics
- Dynamic tags (via configured LLM)

Retrieval searches all stored chunks regardless of domain. Source metadata is
returned with matches so an answer can be traced back to the original page.



## Dynamic tags

Each selected memory or full webpage receives one metadata extraction call
when `DYNAMIC_TAGS_ENABLED=true` using the configured LLM provider (defaults
to Grok `grok-4.7`). The LLM returns 3-8 concise concepts such as
`ChromaDB`, `vector database`, `embeddings`, or `Docker`. These are stored
separately from the fixed DistilBERT topic taxonomy.

## Hybrid retrieval

`POST /ask` now combines four signals:

```text
semantic similarity  55%
keyword overlap      20%
topic overlap        10%
dynamic-tag match    15%
```

The final score is configurable through `.env`. Semantic candidates come from
ChromaDB; lexical/topic/tag signals are then used to rescue exact matches that
pure vector search can miss. The API returns the component scores with each
memory for debugging and evaluation.

After chunk ranking, results are grouped by `document_id` into source-level
records. Each source keeps its strongest relevance score, read timestamp,
topics, dynamic tags, and up to `CHUNKS_PER_SOURCE` relevant snippets. This
prevents one long article from dominating every result while preserving the
best evidence from that article.

## Backfilling existing memories

Memories created before dynamic-tag support can be enriched with Grok without
re-saving the source pages:

```powershell
python -m scripts.backfill_dynamic_tags
```

Use `--dry-run` to inspect the generated tags first, or `--limit N` to process
only a small number of parent documents while testing.

## Grounded RAG synthesis

The configured LLM (defaults to Grok `grok-4.7`, or Gemini, OpenAI, Claude, Ollama, none)
receives only the top hybrid-ranked memories plus their source title/URL.
The system prompt tells the LLM to answer only from those memories and cite the
retrieved source numbers inline as `[1]`, `[2]`, etc. The Chrome extension also
renders the source links from the returned memory metadata.

## Endpoints

| Endpoint | Method | Purpose |
|---|---|---|
| `/health` | `GET` | Health check & service readiness check. |
| `/memory` | `POST` | Save a selected text snippet into memory store. |
| `/webpage` | `POST` | Extract, chunk, classify, embed, and store a complete webpage. |
| `/memory/pdf` | `POST` | Extract, chunk, and index an uploaded multi-page PDF document. |
| `/tags` | `GET` | Retrieve cached DistilBERT topics and concept tags for a URL. |
| `/memory/{doc_id}/tags` | `GET` | Retrieve topics and dynamic tags by parent document ID. |
| `/ask` | `POST` | Retrieve relevant chunks using hybrid fusion and synthesize answer. |
| `/ask/stream` | `POST` | Server-Sent Events (SSE) streaming answers with stage updates & token events. |
| `/jobs/create` | `POST` | Create asynchronous multi-source research summarization job. |
| `/jobs/{job_id}/status` | `GET` | Poll research job processing status and progress percentage. |
| `/jobs/{job_id}/pdf` | `GET` | Download compiled ReportLab PDF research report. |
| `/provider/status` | `GET` | Inspect active AI provider status and credential masking. |
| `/provider/test` | `POST` | Test connectivity with a selected AI provider credentials. |
| `/provider/configure` | `POST` | Transactionally configure and save AI provider credentials. |
| `/provider/remove` | `POST` | Disconnect and clear credentials for an AI provider. |

## Phase validation

Run local tests:

```powershell
cd backend
python -m unittest discover -s app/tests -p "test_*.py"
python -m unittest discover -s ml/tests -p "test_*.py"
```

Run the live Wikipedia validation set:

```powershell
python -m scripts.validate_web_sources
```

Run Wikipedia plus a non-Wikipedia source:

```powershell
python -m scripts.validate_web_sources --all
```

Run the complete pipeline through retrieval and configured QA, persisting the
validation pages in ChromaDB:

```powershell
python -m scripts.validate_web_sources --all --persist --question "What did I read about Python?"
```

The live validation requires internet access and whatever local models are
needed for embeddings/classification. It reports failures rather than
inventing successful results.

## Custom ML topic classifier

The classifier is a genuinely fine-tuned Hugging Face Transformer with a
multi-label sigmoid head. It classifies content, not websites or domains.
External LLMs are not used to produce topic tags; classification is performed entirely by the local DistilBERT model. Dynamic concept tags (separate from topics) are optionally extracted using the configured LLM provider.

See `ml/README.md` for the training and evaluation pipeline.


## Document/source history

Each ingestion receives a stable `document_id` that is shared by all chunks
from that page or selected-memory event. Provenance fields include
`title`, `url`, `domain`, `visited_at`, `topics`, and `tags`.

`POST /ask` returns document-level `sources` in addition to chunk-level
`memories`. Each source contains:

- `document_id`
- `title`, `url`, `domain`
- `visited_at`
- hybrid relevance `score`
- `chunk_count`
- `snippet` / `snippets`
- `topics`
- `tags`

The same information appearing on two different websites remains two distinct
sources, with separate identities and reading times.

The response also returns `related_concepts`, grouping topics and dynamic tags
by the number of distinct sources in which they appear.
