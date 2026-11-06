# MemoryMind Technical Architecture

MemoryMind is an intelligent personal memory and research synthesis platform. It bridges a modern **Manifest V3 Chrome Extension** with a **FastAPI asynchronous backend**, integrating hybrid vector-lexical retrieval, fine-tuned transformer topic classification, multi-source document ingestion, and multi-provider LLM synthesis.

---

## 1. High-Level Architecture

```mermaid
flowchart TD
    subgraph Client["Chrome Extension (Manifest V3)"]
        Popup["Popup UI: Ask / Remember / Settings"]
        ResUI["Research UI: Multi-Source Summarizer"]
        SW["Background Service Worker"]
        APIClient["Client API Layer (api.js)"]
        Popup --> APIClient
        ResUI --> APIClient
        SW --> APIClient
    end

    subgraph Backend["FastAPI Backend (Port 8000)"]
        Router["REST & SSE Streaming Endpoints (/ask, /webpage, /memory, /jobs)"]
        Auth["Security & Credential Masking"]

        subgraph IngestionEngine["Web & PDF Ingestion"]
            Extractor["Content Extraction (Trafilatura & PyPDF)"]
            Chunker["Word-Window Chunking (220 words, 40 overlap)"]
            Extractor --> Chunker
        end

        subgraph MLSubsystem["Machine Learning Pipeline"]
            Classifier["DistilBERT Topic Classifier (17 Taxonomy Domains)"]
            Embedder["Sentence Transformers (all-MiniLM-L6-v2)"]
        end

        subgraph HybridSearch["Hybrid Retrieval Engine (memory_store.py)"]
            DenseScore["Dense Cosine Similarity (55%)"]
            LexScore["SQLite BM25 Keyword Search (20%)"]
            TopicScore["DistilBERT Topic Match (10%)"]
            TagScore["Dynamic Concept Tag Match (15%)"]
            ScoreFusion["Score Fusion & Source-Level Grouping"]
            DenseScore --> ScoreFusion
            LexScore --> ScoreFusion
            TopicScore --> ScoreFusion
            TagScore --> ScoreFusion
        end

        subgraph MultiSourceSynthesis["Multi-Source Summarization & Synthesis"]
            Collector["Source Collector (URLs, Tabs, PDFs)"]
            JobWorker["Background Async Job Manager"]
            PDFGen["ReportLab PDF Generator"]
            Collector --> JobWorker
            JobWorker --> PDFGen
        end

        subgraph LLMGateway["Multi-Provider LLM Gateway"]
            Sanitizer["Prompt Injection Defense (<untrusted_evidence>)"]
            ProviderRouter["Provider Gateway (Grok, Gemini, OpenAI, Claude, Ollama, Offline)"]
            Sanitizer --> ProviderRouter
        end
    end

    subgraph Storage["Databases & Caches"]
        Chroma[("ChromaDB Vector Store")]
        FTS[("SQLite FTS5 Lexical Index")]
        JobData[("Job Store & Generated PDFs")]
    end

    subgraph External["External Services"]
        HFHub["Hugging Face Model Hub (farhanx3rz/memorymind-classifier)"]
        AIProviders["External AI Provider APIs"]
    end

    %% Connections
    APIClient <--> Router
    Router --> Auth
    Auth --> IngestionEngine
    Auth --> HybridSearch
    Auth --> MultiSourceSynthesis

    %% Ingestion flow
    Chunker --> Classifier
    Chunker --> Embedder
    Classifier -.->|Auto-Download on Demand| HFHub
    Embedder --> Chroma
    Chunker --> FTS

    %% Retrieval & Synthesis flow
    HybridSearch <--> Chroma
    HybridSearch <--> FTS
    HybridSearch --> LLMGateway
    LLMGateway --> AIProviders
    AIProviders --> LLMGateway
    LLMGateway --> Router

    %% Summarization flow
    JobWorker --> HybridSearch
    JobWorker --> LLMGateway
    PDFGen --> JobData
```

---

## 2. Component Breakdown

### 2.1 Chrome Extension (`extension/`)
- **Manifest V3 Architecture:** Uses a service worker (`background/service-worker.js`) for context menus and tab lifecycle monitoring.
- **Popup Dashboard (`popup/`):**
  - **First-Run Onboarding:** Intercepts unconfigured states and prompts the user to activate an AI provider.
  - **Interactive Question Answering:** Text input supporting instant streaming answers with inline clickable citation badges `[1]`, `[2]`.
  - **One-Click Page & Selection Memory:** Sends page URL or highlighted text to the backend for automatic extraction and tagging.
  - **AI Provider Management:** Dedicated Settings view to inspect active provider status (ACTIVE / NEEDS ATTENTION / NOT CONFIGURED), view masked credentials, live-test connections, or transactionally switch providers.
- **Unified Summarizer (`pdfsummarizer/` & `research/`):**
  - Multi-source research interface supporting mixed inputs (webpages, uploaded PDFs, active tab).
  - Configurable synthesis parameters (style, topic focus, length).
  - Background job polling with visual progress bar and downloadable PDF report generation.
- **API Service Layer (`api.js`):**
  - Manages `chrome.storage.local` persistence with graceful fallbacks.
  - Handles server health checking, streaming Server-Sent Events (SSE), and offline error recovery.

---

### 2.2 FastAPI Backend (`backend/app/`)
- **Main Application (`main.py`):**
  - Asynchronous lifespan preloading for ML models and ChromaDB clients.
  - Routes for `/health`, `/ask` (standard and SSE streaming), `/remember`, `/page`, `/jobs`, and `/provider/*`.
- **Security & Provider Management (`config.py`):**
  - Server-side credential isolation: Provider API keys are never stored in the extension or returned in plain text via API responses.
  - Key masking (`sk-â€¢â€¢â€¢â€¢â€¢â€¢â€¢â€¢abcd`) and transactional provider switching (test before commit).
- **Hybrid Retrieval Engine (`memory_store.py`):**
  - Combines four distinct signal channels into a unified relevance score:
    $$\text{Final Score} = 0.55 \cdot S_{\text{semantic}} + 0.20 \cdot S_{\text{keyword}} + 0.10 \cdot S_{\text{topic}} + 0.15 \cdot S_{\text{tag}}$$
  - **Semantic Dense Search:** ChromaDB cosine similarity using `sentence-transformers/all-MiniLM-L6-v2`.
  - **Lexical Keyword Search:** SQLite FTS5 full-text indexing for exact phrase matching.
  - **Topic Classification:** Fine-tuned DistilBERT multi-label taxonomy scoring.
  - **Concept Tags:** LLM-extracted dynamic tags for nuanced domain indexing.

---

### 2.3 Machine Learning Subsystem (`backend/ml/`)
- **Model:** Fine-tuned DistilBERT multi-label topic classifier trained across 17 technical domains (AI, RAG, Databases, Cloud, Mathematics, etc.).
- **Hugging Face Hub Integration (`hub.py`):**
  - Model weights (`model.safetensors`, 268 MB) are decoupled from the Git repository and hosted on [Hugging Face (`farhanx3rz/memorymind-classifier`)](https://huggingface.co/farhanx3rz/memorymind-classifier).
  - Uses `snapshot_download` to fetch artifacts into an isolated staging directory on demand.
  - Validates configuration, weights, and tokenizer files before promoting to the active model directory.
  - Supports configurable revisions (`HF_MODEL_REVISION`) and offline fallback.
- **Inference Pipeline (`classifier.py`):**
  - Efficient representative chunk evaluation: for long documents, samples head and distributed body chunks to prevent redundant inference while maintaining high classification accuracy.

---

### 2.4 Document Processing & Summarization (`backend/app/`)
- **Web Extraction (`webpage.py`):** Uses Trafilatura and BeautifulSoup to sanitize arbitrary webpage HTML, strip navigation boilerplate, and extract clean markdown content.
- **PDF Extraction (`pdf_handler.py`):** Uses `pypdf` to extract text from multi-page PDFs, mapping content to individual page numbers for citation accuracy.
- **Synthesis Engine (`synthesis.py`):** Formulates structured research summaries with citation verification.
- **ReportLab PDF Generator (`pdf_generator.py`):** Compiles multi-source research into branded, publication-ready PDF reports with headers, table of contents, summaries, and formal citation bibliographies.

---

## 3. Data Flow Workflows

### 3.1 Memory Capture & Indexing
```mermaid
sequenceDiagram
    autonumber
    actor User
    participant Ext as Chrome Extension
    participant API as FastAPI Backend
    participant ML as DistilBERT Classifier
    participant LLM as Active LLM
    participant DB as ChromaDB & SQLite FTS

    User->>Ext: Click "Remember current page"
    Ext->>API: POST /page {url, html}
    API->>API: Extract clean text & title (Trafilatura)
    API->>API: Chunk into word windows (220 words, 40 overlap)
    par Topic Classification
        API->>ML: Predict topic tags across chunks
        ML-->>API: Topics (e.g., ["RAG", "Databases"])
    and Dynamic Concept Extraction
        API->>LLM: Extract 3-8 key concepts
        LLM-->>API: Tags (e.g., ["vector search", "indexing"])
    end
    API->>DB: Store embeddings (ChromaDB) & text (SQLite FTS5)
    DB-->>API: Document ID confirmed
    API-->>Ext: HTTP 200 {document_id, topics, tags}
    Ext-->>User: Show "âœ“ Page saved Â· RAG Â· Databases"
```

---

### 3.2 Question Answering with Citations (Streaming RAG)
```mermaid
sequenceDiagram
    autonumber
    actor User
    participant Ext as Chrome Extension
    participant API as FastAPI Backend
    participant DB as Hybrid Retrieval Store
    participant LLM as Active LLM

    User->>Ext: Enters question & clicks "Ask"
    Ext->>API: GET /ask/stream?question=... (SSE)
    API->>DB: Query Hybrid Retrieval (Dense + FTS + Topics + Tags)
    DB-->>API: Top-K source passages with metadata
    API->>LLM: Stream prompt with context passages & citation guidelines
    loop Token Streaming
        LLM-->>API: Token chunk
        API-->>Ext: SSE data: {"token": "..."}
        Ext->>User: Renders incremental Markdown & inline [1], [2] badges
    end
    API-->>Ext: SSE event: done {"sources": [...], "related_concepts": [...]}
    Ext->>User: Displays source cards with relevance scores & deep links
```

---

### 3.3 Multi-Source Research Summarization & PDF Generation
```mermaid
sequenceDiagram
    autonumber
    actor User
    participant Ext as Research Synthesizer UI
    participant API as FastAPI Backend
    participant Collector as Source Collector
    participant Worker as Background Job Manager
    participant LLM as Active LLM Provider
    participant PDF as ReportLab PDF Engine

    User->>Ext: Selects active tabs, URLs & uploads PDFs
    Ext->>API: POST /jobs/create {sources, style, topic, length}
    API->>Collector: Ingest and parse all selected sources
    Collector-->>API: Unified extracted passages
    API->>Worker: Enqueue background job (UUID)
    API-->>Ext: HTTP 200 {job_id, status: "pending"}

    loop Polling Status
        Ext->>API: GET /jobs/{job_id}/status
        API-->>Ext: {status: "processing", progress: 60}
    end

    Worker->>LLM: Synthesize executive summary & structured sections
    LLM-->>Worker: Grounded multi-source synthesis text
    Worker->>PDF: Generate formatted PDF report with bibliography
    PDF-->>Worker: Compiled PDF artifact (backend/data/jobs/{id}.pdf)
    Worker-->>API: Job marked "completed"

    Ext->>API: GET /jobs/{job_id}/status -> "completed"
    User->>Ext: Click "Download Research Report (PDF)"
    Ext->>API: GET /jobs/{job_id}/pdf
    API-->>Ext: application/pdf stream
    Ext-->>User: Saved PDF report download
```

