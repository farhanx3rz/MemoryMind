# MemoryMind Development & Contribution Guide

This guide provides step-by-step instructions for setting up, developing, testing, and debugging MemoryMind locally on Windows, macOS, or Linux.

---

## 1. Prerequisites

- **Python:** 3.10 to 3.12 (Python 3.12 recommended).
- **Google Chrome:** Any modern release supporting Manifest V3 extensions.
- **Git:** Standard Git CLI.
- **Hardware:**
  - 8 GB+ RAM recommended for running local Sentence Transformers and DistilBERT inference.
  - No GPU is required (CPU inference runs in ~250–500 ms).

---

## 2. Local Environment Setup

### 2.1 Clone the Repository
```bash
git clone https://github.com/farhanx3rz/MemoryMind.git
cd MemoryMind
```

### 2.2 Create a Python Virtual Environment
On Windows (PowerShell):
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

On macOS / Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2.3 Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 3. Environment Configuration

1. Copy the template `.env.example` to `backend/.env`:
   ```powershell
   copy .env.example backend\.env
   ```

2. Open `backend/.env` and configure your preferred AI provider:
   ```env
   # Select your active provider: grok | gemini | openai | anthropic | ollama | none
   # .env.example pre-configures grok. If LLM_PROVIDER is unset or empty,
   # backend/app/config.py safely defaults to "none" (Offline / Echo Mode).
   LLM_PROVIDER=grok

   # Enter credentials for your selected provider:
   XAI_API_KEY=xai-your-api-key-here
   GROK_MODEL=grok-4.7

   # Hugging Face Topic Classifier Settings (Pre-configured):
   HF_MODEL_REPO=farhanx3rz/memorymind-classifier
   HF_MODEL_REVISION=main
   HF_TOKEN=
   ```

> **Note on Provider Defaults & Offline Testing:**
> - When using `.env.example`, `LLM_PROVIDER` is set to `grok` and requires a valid `XAI_API_KEY`.
> - If `LLM_PROVIDER` is unset or omitted from the environment, `backend/app/config.py` defaults to `none` (Offline / Echo Mode).
> - Setting `LLM_PROVIDER=none` explicitly in `backend/.env` runs MemoryMind in **Offline / Echo Mode**, returning the best matching memory excerpt directly without requiring an external LLM API key.

---

## 4. Running the Application

### 4.1 Start the FastAPI Backend
From the project root:
```powershell
python backend/run_server.py
```
The server will bind to `http://127.0.0.1:8000`.

- On first launch, MemoryMind automatically checks for the DistilBERT classifier. If the 268 MB weights are not present locally, it automatically downloads and caches them from Hugging Face into `backend/ml/models/recall-topic-classifier/`.
- Interactive API documentation is available at `http://127.0.0.1:8000/docs`.

### 4.2 Load the Extension in Chrome
1. Open Google Chrome and navigate to `chrome://extensions/`.
2. Enable **Developer mode** using the toggle in the top-right corner.
3. Click **Load unpacked**.
4. Select the `extension/` directory inside the `MemoryMind` project folder (`c:\path\to\MemoryMind\extension`).
5. Pin MemoryMind to your toolbar. The purple neural-brain logo will appear.

---

## 5. Running the Automated Test Suite

MemoryMind maintains a comprehensive test suite across backend API routes, security guards, PDF processing, machine learning inference, and Hugging Face integration.

### Run All Tests:
```powershell
$env:PYTHONPATH="backend"
pytest backend/ml/tests/ backend/app/tests/ -v
```

### Run Specific Test Modules:
- **Hugging Face Hub & Caching Tests:**
  ```powershell
  $env:PYTHONPATH="backend"
  pytest backend/ml/tests/test_hf_hub.py -v
  ```
- **ML Dataset & Topic Classifier Tests:**
  ```powershell
  $env:PYTHONPATH="backend"
  pytest backend/ml/tests/test_ml.py -v
  ```
- **Provider Switching & Credential Masking Tests:**
  ```powershell
  $env:PYTHONPATH="backend"
  pytest backend/app/tests/test_provider_management.py -v
  ```
- **Release Hygiene & Secret Leak Prevention Tests:**
  ```powershell
  $env:PYTHONPATH="backend"
  pytest backend/app/tests/test_release_hygiene.py -v
  ```

---

## 6. Standalone CLI Tools

MemoryMind includes utility scripts in `backend/` and `backend/scripts/`:

### 6.1 CLI RAG Prototype (`prototype.py`)
Interact with the retrieval system directly from your terminal:
```powershell
cd backend
python prototype.py --ask "Why am I learning Python?"
python prototype.py --remember "Python is widely used for machine learning."
```

### 6.2 Standalone ML Topic Prediction (`ml.predict`)
Run the trained classifier on arbitrary text:
```powershell
$env:PYTHONPATH="backend"
python -m ml.predict "Vector databases store dense embeddings for semantic search."
```

### 6.3 Database Inspection (`backend/scripts/inspect_db.py`)
Inspect the current contents, document counts, and embeddings stored in ChromaDB:
```powershell
cd backend
python scripts/inspect_db.py
```

---

## 7. Hugging Face Model Workflow

### 7.1 Retraining the Classifier
If you wish to retrain the topic classifier on expanded data:
1. Update `backend/ml/data/starter_dataset.jsonl`.
2. Run training:
   ```powershell
   $env:PYTHONPATH="backend"
   python -m ml.train
   ```
3. Evaluate metrics:
   ```powershell
   $env:PYTHONPATH="backend"
   python -m ml.evaluate
   ```

### 7.2 Publishing New Weights to Hugging Face
```bash
# Log in to Hugging Face
huggingface-cli login

# Upload updated checkpoint files
huggingface-cli upload farhanx3rz/memorymind-classifier ./backend/ml/models/recall-topic-classifier/ . --commit-message "Release v2.0 weights"
```

To pin your deployment to a specific commit or release tag, set `HF_MODEL_REVISION` in `backend/.env`.

---

## 8. Troubleshooting

| Issue | Cause | Resolution |
|---|---|---|
| `Backend server unreachable` in extension | Backend is not running on port 8000 | Run `python backend/run_server.py` in your terminal. |
| `Topic classifier unavailable` | Missing ML dependencies or download interrupted | Run `pip install -r requirements.txt`. Ensure your machine has internet access to download weights from Hugging Face on the first run. |
| Extension icon not showing | Extension unpacked from wrong directory | In `chrome://extensions`, ensure you selected the `extension/` subfolder (which contains `manifest.json`), not the repository root. |
| Port 8000 already in use | Stale Python process running | In PowerShell, run `Get-Process -Name python \| Stop-Process -Force`. |

