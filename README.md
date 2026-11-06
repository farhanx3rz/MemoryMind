# MemoryMind

> **Your Personal AI Web Memory & Multi-Source Research Synthesis Platform**
> Capture knowledge from arbitrary webpages, perform hybrid semantic-lexical recall, categorize content using fine-tuned transformer topic classification, and synthesize grounded research with verifiable citations.

[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688.svg?style=flat&logo=fastapi)](https://fastapi.tiangolo.com)
[![Chrome Extension](https://img.shields.io/badge/Chrome_Extension-Manifest_V3-4285F4.svg?style=flat&logo=googlechrome)](https://developer.chrome.com/docs/extensions/mv3/intro/)
[![ChromaDB](https://img.shields.io/badge/Vector_DB-ChromaDB-fc521f.svg?style=flat)](https://www.trychroma.com/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97_Model_Hub-farhanx3rz%2Fmemorymind--classifier-ffcc00.svg?style=flat)](https://huggingface.co/farhanx3rz/memorymind-classifier)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python: 3.10--3.12](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-3776AB.svg?style=flat&logo=python)](https://www.python.org/)

---

## Application Showcase

| 1. Main Interface & QA | 2. AI Provider Settings | 3. Webpage Memory Saved |
|:---:|:---:|:---:|
| <img src="docs/screenshots/01-main-interface.png" alt="MemoryMind Main Interface and Question Answering Popup" width="280"> | <img src="docs/screenshots/02-ai-provider-settings.png" alt="MemoryMind AI Provider Configuration and Settings" width="280"> | <img src="docs/screenshots/03-webpage-saved.png" alt="MemoryMind Webpage Saved Confirmation with Topic Tags" width="280"> |
| Ask questions grounded in saved memories with streaming answers and citations | Transactionally configure and test Grok, Gemini, OpenAI, Claude, or Ollama with masked keys | One-click webpage capture with automatic DistilBERT topic classification and dynamic tags |

---

## Quick Start

Follow these steps to set up and run MemoryMind locally:

```powershell
# 1. Clone repository & install dependencies
git clone https://github.com/farhanx3rz/MemoryMind.git
cd MemoryMind
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt

# 2. Configure environment variables
# Copy the template configuration to backend/.env:
copy .env.example backend\.env
```

> **LLM Provider Defaults & Offline Mode:**
> - **With `.env.example`:** Copying `.env.example` pre-selects `LLM_PROVIDER=grok`. If you use this configuration, add your `XAI_API_KEY=your_key_here` to `backend/.env`.
> - **Without `.env` or Unset:** If `LLM_PROVIDER` is not set in the environment, `backend/app/config.py` safely defaults to `LLM_PROVIDER=none` (Offline / Echo Mode).
> - **Explicit Offline / Echo Mode:** Set `LLM_PROVIDER=none` in `backend/.env` to test hybrid vector retrieval with zero API keys by returning top matching memory passages directly with relevance scores. LLM answer synthesis and dynamic concept tags require a configured provider key.
> - Supported providers: `grok` (`grok-4.7`), `gemini` (`gemini-3.5-flash`), `openai` (`gpt-4o-mini`), `anthropic` (`claude-3-5-haiku-latest`), `ollama` (`llama3.1`), or `none`.

```powershell
# 3. Start the FastAPI backend server
python backend/run_server.py
```

> **First-Launch Model Downloads:**
> On initial startup, MemoryMind may download the Sentence Transformers embedding model (`all-MiniLM-L6-v2`, ~90 MB) and fetch the fine-tuned DistilBERT weights from Hugging Face Hub ([`farhanx3rz/memorymind-classifier`](https://huggingface.co/farhanx3rz/memorymind-classifier), ~268 MB) if they are not already cached locally. Download and initialization times vary depending on your network connection speed and hardware. Once downloaded, models are cached locally so subsequent launches avoid re-downloading.

4. **Load the Chrome Extension:**
   - Open Google Chrome and navigate to `chrome://extensions/`.
   - Enable **Developer mode** using the toggle in the upper-right corner.
   - Click **Load unpacked** and select the `extension/` directory (`MemoryMind/extension`).
   - Pin MemoryMind to your browser toolbar.

> **Key Project Links:**
> - [System Architecture Documentation](docs/architecture.md)
> - [Backend Pipeline & Store Documentation](backend/README.md)
> - [Local Development & Debugging Guide](docs/development.md)
> - [Fine-Tuned Topic Classifier on Hugging Face Hub](https://huggingface.co/farhanx3rz/memorymind-classifier)

---

## 1. Overview & Problem Solved

Every day, software engineers, researchers, and students consume hundreds of technical articles, documentation pages, tutorials, and PDF papers. Most of this information is lost to forgotten browser bookmarks or buried history tabs.

**MemoryMind** solves this cognitive bottleneck:
- **Instant Web Memory:** One click captures highlighted passages or entire articles with boilerplate navigation stripped.
- **Hybrid Retrieval (Dense + Lexical + ML Taxonomy):** Merges ChromaDB semantic embeddings with SQLite FTS5 BM25 text search and DistilBERT topic tags, avoiding both vector drift and keyword brittleness.
- **Verifiable RAG Synthesis:** Streams natural language answers grounded strictly in your saved memories, complete with interactive clickable citation badges (`[1]`, `[2]`).
- **Multi-Source Research Synthesizer:** Aggregates arbitrary combinations of live webpages, PDF files, and active tabs into unified research summaries with downloadable, formatted PDF reports.
- **Hugging Face Hub Decoupling:** Uses a fine-tuned DistilBERT classifier hosted on Hugging Face that downloads and caches automatically on demand, eliminating 268 MB of binary weights from the Git repository.

---

## 2. System Architecture

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

    subgraph Backend["FastAPI Backend (127.0.0.1:8000)"]
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

## 3. Key Features

### Grounded Question Answering & Memory Retrieval
- **Hybrid Search Fusion:** Scores candidates using a 4-signal weighted algorithm:
  $$\text{Final Score} = 0.55 \cdot S_{\text{semantic}} + 0.20 \cdot S_{\text{keyword}} + 0.10 \cdot S_{\text{topic}} + 0.15 \cdot S_{\text{tag}}$$
- **Streaming Responses:** Real-time token streaming via Server-Sent Events (SSE) with stage feedback (*"Searching memories..."* $\rightarrow$ *"Synthesizing answer..."*).
- **Interactive Citations:** Answers embed interactive badges (`[1]`, `[2]`) that smoothly scroll to and highlight the corresponding source provenance cards with metadata and deep links.
- **Prompt Injection Defense:** Webpage content is framed in strictly untrusted boundaries (`<untrusted_evidence>`), sanitizing embedded prompt overrides or hallucinated citations.

### Multi-Source Research & PDF Synthesis
- **Flexible Ingestion:** Ingest any combination of active browser tabs, raw URLs, and local PDF documents.
- **Customizable Synthesis:** Configure synthesis style (Academic, Executive, Technical), length, and topic focus.
- **ReportLab PDF Generation:** Asynchronous background worker compiles structured summaries into branded PDF reports complete with headers, executive summaries, key findings, and formal bibliographies.

### Fine-Tuned Transformer Topic Classifier
- **17 Domain Taxonomy:** Fine-tuned `distilbert-base-uncased` across: *Artificial Intelligence, Machine Learning, Deep Learning, Large Language Models, RAG, Information Retrieval, Data Science, Programming, Web Development, Databases, Cloud Computing, Cybersecurity, Mathematics, Science, Business, History, Other*.
- **Hugging Face Hub Integration:** Decoupled from Git. Model artifacts are downloaded from [farhanx3rz/memorymind-classifier](https://huggingface.co/farhanx3rz/memorymind-classifier) into local cache on demand.
- **Evaluation on Held-Out Test Split (741 samples):**
  - **Subset Accuracy (Exact Match):** 91.23%
  - **Hamming Accuracy:** 99.31% (Hamming Loss: 0.69%)
  - **Micro F1:** 95.29% | **Macro F1:** 94.92% | **Weighted F1:** 95.32%
  - **Micro Precision:** 94.73% | **Micro Recall:** 95.86%
  - **Data Leakage:** 0 overlap across train, validation, and test splits (verified leak-free)
  - Detailed per-class metrics and confusion matrices are recorded in [`backend/ml/data/evaluation_report.json`](backend/ml/data/evaluation_report.json).

### First-Run Activation & Provider Management
- **First-Run Onboarding:** Clean onboarding experience on first extension launch if no AI provider is configured.
- **Safe Provider Switching:** Transactional validation (tests credentials against provider API before committing).
- **Multi-Provider Support:**
  - **xAI Grok** (`grok-4.7`)
  - **Google Gemini** (`gemini-3.5-flash`)
  - **OpenAI** (`gpt-4o-mini`)
  - **Anthropic Claude** (`claude-3-5-haiku-latest`)
  - **Ollama** (`llama3.1` / self-hosted local models)
  - **Offline / Echo Mode** (zero API keys needed; tests retrieval by echoing top memory passages)
- **Credential Security:** Provider API keys are stored server-side in `backend/.env`. The extension only receives masked keys (`sk-••••••••abcd`) and keys are never printed in logs or error messages.

---

## 4. Technology Stack

| Category | Technologies |
|---|---|
| **Client / Extension** | Chrome Extensions API (Manifest V3), JavaScript (ES Modules), HTML5, CSS3, SVG |
| **Backend Framework** | Python 3.12, FastAPI, Uvicorn, Pydantic v2, HTTPX, Python-Dotenv |
| **Vector DB & Retrieval** | ChromaDB (Dense Cosine Similarity), SQLite FTS5 (BM25 Lexical Index) |
| **Embeddings & ML** | Sentence-Transformers (`all-MiniLM-L6-v2`), PyTorch, Hugging Face `transformers`, `huggingface-hub`, `safetensors`, Scikit-Learn |
| **Document Processing** | Trafilatura, BeautifulSoup4, PyPDF, ReportLab |
| **Quality & Testing** | Pytest, Unittest, Mock, GitHub Actions compatible |

---

## 5. Repository Structure

```text
MemoryMind/
│
├── backend/                               # FastAPI application & ML pipeline
│   ├── app/                               # Core backend services
│   │   ├── config.py                      # Server, retrieval, and provider configuration
│   │   ├── main.py                        # FastAPI routes & streaming SSE endpoints
│   │   ├── memory_store.py                # ChromaDB + SQLite FTS5 hybrid retrieval engine
│   │   ├── llm.py                         # Multi-provider LLM gateway & prompt injection defense
│   │   ├── pipeline.py                    # Webpage ingestion and classification pipeline
│   │   ├── webpage.py                     # Trafilatura HTML boilerplate cleaner & extractor
│   │   ├── pdf_handler.py                 # Multi-page PDF text extraction
│   │   ├── pdf_generator.py               # ReportLab PDF report compilation
│   │   ├── summarization_jobs.py          # Background job manager & persistent worker
│   │   └── tests/                         # Backend test suite (14 test modules)
│   │
│   ├── ml/                                # Machine Learning classifier subsystem
│   │   ├── classifier.py                  # TopicClassifier inference engine & chunk aggregator
│   │   ├── hub.py                         # Hugging Face snapshot download & caching logic
│   │   ├── config.py                      # Classifier hyperparameters & taxonomy labels
│   │   ├── train.py                       # DistilBERT fine-tuning script
│   │   ├── evaluate.py                    # Multi-label evaluation & metrics generator
│   │   ├── dataset.py                     # Stratified disjoint group splitter (leak-free)
│   │   ├── predict.py                     # Standalone CLI prediction utility
│   │   ├── data/                          # Labeled dataset (starter_dataset.jsonl, splits.json)
│   │   ├── models/                        # Local model cache directory (weights gitignored)
│   │   └── tests/                         # ML and Hugging Face integration tests
│   │
│   ├── scripts/                           # Database inspector & benchmark scripts
│   ├── prototype.py                       # Standalone CLI terminal RAG prototype
│   ├── run_server.py                      # Server runner with interactive startup check
│   ├── requirements.txt                   # Backend Python dependencies
│   ├── .env.example                       # Environment configuration template
│   └── .gitignore                         # Backend-specific ignore rules
│
├── extension/                             # Chrome Browser Extension (Manifest V3)
│   ├── manifest.json                      # Extension manifest & permissions
│   ├── api.js                             # Client HTTP, SSE streaming, and storage service
│   ├── config.js                          # Client-side endpoints & authentication
│   ├── pdf-retrieval.js                   # Active tab PDF byte extraction
│   ├── popup/                             # Main extension UI (Dashboard, Settings, Activation)
│   │   ├── popup.html
│   │   ├── popup.js
│   │   └── popup.css
│   ├── icons/                             # Extension brand icons (16, 32, 48, 128 px)
│   ├── pdfsummarizer/                     # PDF upload & summarizer UI
│   └── research/                          # Multi-source research synthesis interface
│
├── docs/                                  # Technical documentation
│   ├── architecture.md                    # Detailed architecture & sequence workflows
│   ├── development.md                     # Local setup, testing, and debugging guide
│   ├── screenshots/                       # UI showcase screenshots and capture checklist
│   └── fixes/                             # Engineering fix retrospectives
│
├── .gitignore                             # Root repository gitignore
├── .env.example                           # Root environment variable template
├── requirements.txt                       # Project-wide dependencies
├── LICENSE                                # MIT Open Source License
└── README.md                              # Project documentation
```

---

## 6. Installation & Setup

### Prerequisites
- **Python 3.10 – 3.12** installed on your system.
- **Google Chrome** (or any Chromium browser).
- **Git**.

### Step 1: Clone the Repository
```powershell
git clone https://github.com/farhanx3rz/MemoryMind.git
cd MemoryMind
```

### Step 2: Create a Virtual Environment & Install Dependencies
On Windows (PowerShell):
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

On Linux / macOS:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 3: Configure Environment Variables
Copy the template configuration into `backend/.env`:
```powershell
copy .env.example backend\.env
```

Open `backend/.env` in your editor and configure your active AI provider:
```env
# Choose provider: grok | gemini | openai | anthropic | ollama | none
LLM_PROVIDER=grok

# Provide credentials for your selected provider:
XAI_API_KEY=xai-your-api-key-here
GROK_MODEL=grok-4.7

# Hugging Face Topic Classifier Settings (Pre-configured):
HF_MODEL_REPO=farhanx3rz/memorymind-classifier
HF_MODEL_REVISION=main
HF_TOKEN=
```

> **Provider Configuration & Offline Mode:**
> - When using `.env.example`, `LLM_PROVIDER` is set to `grok` and expects an `XAI_API_KEY`.
> - If `LLM_PROVIDER` is unset or omitted from the environment, `backend/app/config.py` safely defaults to `"none"` (Offline / Echo Mode).
> - Set `LLM_PROVIDER=none` explicitly in `backend/.env` to run MemoryMind in **Offline / Echo Mode**, testing hybrid vector retrieval by returning the most relevant memory passages directly with zero external API calls.

### Step 4: Start the FastAPI Backend Server
```powershell
python backend/run_server.py
```
- The backend initializes at `http://127.0.0.1:8000`.
- **Model Downloads & Caching:** If models are not present locally, MemoryMind downloads the Sentence Transformers embeddings and automatically fetches the fine-tuned DistilBERT weights into `backend/ml/models/recall-topic-classifier/`. Download times depend on network bandwidth. Subsequent launches load cached weights locally.
- Interactive Swagger API docs are available at `http://127.0.0.1:8000/docs`.

### Step 5: Install the Chrome Extension
1. Open Chrome and navigate to `chrome://extensions/`.
2. Toggle on **Developer mode** in the upper-right corner.
3. Click **Load unpacked**.
4. Browse to and select the `extension/` folder within this repository (`MemoryMind/extension`).
5. Pin MemoryMind to your browser toolbar.

---

## 7. Hugging Face Model Integration

The multi-label topic classifier is hosted separately on the Hugging Face Model Hub:

**Model Repository:** [`farhanx3rz/memorymind-classifier`](https://huggingface.co/farhanx3rz/memorymind-classifier)

### Why Decouple the Model?
- **Lightweight Repository:** Standard Git repositories are designed for code, not multi-hundred megabyte binaries. Decoupling saves 268 MB of binary bloat.
- **Automatic Caching:** Download occurs only once during initial setup; subsequent server restarts reuse the cached files locally without re-downloading over the network.
- **Resilient Fallback:** If internet access is unavailable, MemoryMind gracefully reuses the existing cached model without throwing errors.
- **Configurable Revisions:** Pin to specific commits or release tags simply by setting `HF_MODEL_REVISION` in `.env`.

---

## 8. Automated Testing & Verification

MemoryMind maintains an automated test suite covering security, authentication, hybrid retrieval, PDF processing, machine learning classification, and Hugging Face integration:

```powershell
$env:PYTHONPATH="backend"
pytest backend/ml/tests/ backend/app/tests/ -v
```

### Verified Test Results: **167 Passed, 2 Skipped, 117 Subtests Passed**
- **Machine Learning Subsystem (`backend/ml/tests/`): 20 passed**
  - `test_hf_hub.py`: **9 passed** (Caching, offline fallback, mock download, rollback, validation)
  - `test_ml.py`: **11 passed** (Leak-free stratified splitting, multilabel encoding, deduplication)
- **FastAPI Core & Security Suite (`backend/app/tests/`): 147 passed, 2 skipped, 117 subtests passed**
  - `test_auth_routes.py`: **3 passed** (API authentication, header verification)
  - `test_citation_schema.py`: **11 passed** (Citation schema validation, report compatibility)
  - `test_e2e_regression.py`: **6 passed** (Remember text/webpage/PDF, ask, stream, job flow)
  - `test_hybrid_retrieval.py`: **10 passed** (4-signal fusion, source grouping, concept aggregation)
  - `test_integration_workflow.py`: **2 skipped** (`test_job_workflow_with_citations` and `test_citation_schema_in_api` are live integration tests that require an actively running backend server on port 8000; when run against offline unit suites, they skip cleanly via `httpx.ConnectError`)
  - `test_optimization_and_reproduction.py`: **14 passed** (Typo tolerance, chunk sampling, caching)
  - `test_pdf_formatting.py`: **12 passed** (ReportLab layout, bibliography generation)
  - `test_pdf_summarization.py`: **12 passed** (Multi-page extraction, section structuring)
  - `test_pdf_upload_validation.py`: **11 passed** (Size limits, MIME type guards, corrupted file rejection)
  - `test_provider_management.py`: **11 passed** (Transactional provider switching, env persistence)
  - `test_provider_security.py`: **11 passed** (Credential masking, sensitive data redaction)
  - `test_release_hygiene.py`: **6 passed** (Distribution exclusion rules, zero secret leakage)
  - `test_ssrf_protection.py`: **14 passed** (SSRF URL guards, private IP address blocking)
  - `test_synthesis_timeout_retry.py`: **13 passed** (Exponential backoff, timeout handling)
  - `test_webpage_pipeline.py`: **14 passed** (Trafilatura boilerplate cleanup, chunk preservation)

> **Test Scope & Qualification:** These automated tests verify algorithmic logic, regression safety, input boundary sanitization, and interface contracts under local development conditions. They do not constitute an evaluation of multi-tenant load or distributed production deployment.

---

## 9. Limitations

- **Local Backend Dependency:** MemoryMind currently requires the local FastAPI Python backend running on your machine to perform embeddings, vector search, and model inference.
- **First-Run Model Downloads:** Initial launch may require downloading model weights (~90 MB embedding model and ~268 MB classifier from Hugging Face Hub); download duration depends on network connection speed and hardware. Once cached locally, subsequent startups bypass downloading.
- **Single-User Architecture:** Designed for personal developer/researcher productivity on local storage; not intended as a multi-tenant enterprise SaaS in its current state.

---

## 10. Future Roadmap

- [ ] **Cross-Device Sync:** Optional end-to-end encrypted sync for memories across multiple desktop instances.
- [ ] **Local LLM Streaming Acceleration:** Enhanced WebGPU or llama.cpp native binding for 100% in-browser zero-server execution.
- [ ] **Automated Flashcard Generation:** Spaced repetition flashcards generated from captured webpage concepts.
- [ ] **Knowledge Graph Visualization:** Interactive 3D graph view of interconnected memories and topic clusters.

---

## 11. License & Attribution

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

- **Author:** Farhan ([@farhanx3rz](https://huggingface.co/farhanx3rz))
- **Topic Classifier:** [`farhanx3rz/memorymind-classifier`](https://huggingface.co/farhanx3rz/memorymind-classifier)
