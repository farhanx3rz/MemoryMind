"""All configuration lives here, read from environment variables / .env."""

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

BACKEND_DIR = Path(__file__).resolve().parent.parent

# backend/.env is loaded first so it wins over a project-root .env.
# load_dotenv never overwrites a variable that is already set, so the
# order below defines the priority.
load_dotenv(BACKEND_DIR / ".env")
load_dotenv(PROJECT_ROOT / ".env")

# Embedding model runs locally, so the first run downloads it (~90 MB).
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# ChromaDB folder. Relative paths are resolved against the "backend" folder
# so the server and the prototype always use the same database.
_chroma_path = Path(os.getenv("CHROMA_PATH", "./data/chroma"))
if not _chroma_path.is_absolute():
    _chroma_path = BACKEND_DIR / _chroma_path
CHROMA_PATH = _chroma_path

# How many memories to retrieve for a question.
TOP_K = int(os.getenv("TOP_K", "8"))

# A retrieved memory must score at least this high to be treated as relevant.
# Cosine similarity with all-MiniLM-L6-v2 lands around 0.5-0.7 for a real
# match and near 0.1 or below for unrelated text, so 0.3 keeps real matches
# and discards noise. This is what stops the system confidently answering
# questions it has no memory for.
MIN_SCORE = float(os.getenv("MIN_SCORE", "0.3"))

# LLM provider: "gemini", "grok", "openai", "anthropic", "ollama" or "none".
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "none").strip().lower()

# --- Google Gemini ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")

# --- xAI Grok ---
XAI_API_KEY = os.getenv("XAI_API_KEY", "")
GROK_MODEL = os.getenv("GROK_MODEL", "grok-4.7")
XAI_BASE_URL = os.getenv("XAI_BASE_URL", "https://api.x.ai/v1")


def mask_api_key(key: str | None) -> str | None:
    """Mask an API key for safe frontend display, showing only prefix and suffix."""
    if not key:
        return None
    cleaned = key.strip()
    if not cleaned:
        return None
    if len(cleaned) <= 8:
        return "••••••••"
    return f"{cleaned[:4]}••••••••{cleaned[-4:]}"


def is_active() -> bool:
    """Return whether the current LLM provider has a configured API key/endpoint."""
    provider = LLM_PROVIDER.lower().strip()
    if provider == "grok":
        return bool(XAI_API_KEY.strip())
    if provider == "gemini":
        return bool(GEMINI_API_KEY.strip())
    if provider == "openai":
        return bool(OPENAI_API_KEY.strip())
    if provider == "anthropic":
        return bool(ANTHROPIC_API_KEY.strip())
    if provider == "ollama":
        return bool(OLLAMA_BASE_URL.strip())
    if provider == "none":
        return True
    return False


def get_provider_details(provider_name: str) -> dict:
    """Return structured details for a supported provider without exposing raw secret keys."""
    p = provider_name.lower().strip()
    if p == "grok":
        return {
            "name": "xAI Grok",
            "provider": "grok",
            "model": GROK_MODEL,
            "requires_key": True,
            "is_configured": bool(XAI_API_KEY.strip()),
            "masked_key": mask_api_key(XAI_API_KEY),
            "base_url": XAI_BASE_URL,
        }
    elif p == "gemini":
        return {
            "name": "Google Gemini",
            "provider": "gemini",
            "model": GEMINI_MODEL,
            "requires_key": True,
            "is_configured": bool(GEMINI_API_KEY.strip()),
            "masked_key": mask_api_key(GEMINI_API_KEY),
        }
    elif p == "openai":
        return {
            "name": "OpenAI",
            "provider": "openai",
            "model": OPENAI_MODEL,
            "requires_key": True,
            "is_configured": bool(OPENAI_API_KEY.strip()),
            "masked_key": mask_api_key(OPENAI_API_KEY),
        }
    elif p == "anthropic":
        return {
            "name": "Anthropic Claude",
            "provider": "anthropic",
            "model": ANTHROPIC_MODEL,
            "requires_key": True,
            "is_configured": bool(ANTHROPIC_API_KEY.strip()),
            "masked_key": mask_api_key(ANTHROPIC_API_KEY),
        }
    elif p == "ollama":
        return {
            "name": "Ollama (Local)",
            "provider": "ollama",
            "model": OLLAMA_MODEL,
            "requires_key": False,
            "is_configured": bool(OLLAMA_BASE_URL.strip()),
            "base_url": OLLAMA_BASE_URL,
        }
    elif p == "none":
        return {
            "name": "Offline / Echo Mode",
            "provider": "none",
            "model": "none",
            "requires_key": False,
            "is_configured": True,
            "masked_key": None,
        }
    else:
        return {
            "name": p.capitalize(),
            "provider": p,
            "model": "",
            "requires_key": True,
            "is_configured": False,
            "masked_key": None,
        }


def get_all_providers_status() -> dict:
    """Return all provider details and the overall active status."""
    supported = ["grok", "openai", "gemini", "anthropic", "ollama", "none"]
    providers_info = {p: get_provider_details(p) for p in supported}

    active_provider = LLM_PROVIDER.lower().strip()
    active_is_configured = is_active()

    status_str = "ACTIVE" if active_is_configured else "NOT_CONFIGURED"

    return {
        "status": "ok",
        "active_provider": active_provider,
        "is_activated": active_is_configured,
        "connection_status": status_str,
        "providers": providers_info,
    }


def _update_env_file(updates: dict[str, str]) -> None:
    """Atomically update key-value pairs in backend/.env preserving other configuration lines."""
    env_file = BACKEND_DIR / ".env"
    lines: list[str] = []
    if env_file.exists():
        lines = env_file.read_text(encoding="utf-8").splitlines()

    remaining = dict(updates)
    for i, line in enumerate(lines):
        line_clean = line.strip()
        if not line_clean or line_clean.startswith("#") or "=" not in line:
            continue
        key_name = line.split("=", 1)[0].strip()
        if key_name in remaining:
            lines[i] = f"{key_name}={remaining.pop(key_name)}"

    for key_name, val in remaining.items():
        lines.append(f"{key_name}={val}")

    env_file.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_provider_config(
    provider: str,
    api_key: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
    set_active: bool = True,
) -> None:
    """Update runtime provider configuration and persist safely to backend/.env."""
    global LLM_PROVIDER, XAI_API_KEY, GROK_MODEL, XAI_BASE_URL
    global GEMINI_API_KEY, GEMINI_MODEL
    global OPENAI_API_KEY, OPENAI_MODEL
    global ANTHROPIC_API_KEY, ANTHROPIC_MODEL
    global OLLAMA_BASE_URL, OLLAMA_MODEL

    p = provider.lower().strip()
    env_updates: dict[str, str] = {}

    if set_active:
        LLM_PROVIDER = p
        env_updates["LLM_PROVIDER"] = p

    if base_url is not None:
        from .provider_security import validate_provider_url
        is_valid_url, url_err = validate_provider_url(p, base_url)
        if not is_valid_url:
            raise ValueError(f"Provider URL security rejection: {url_err}")

    if p == "grok":
        if api_key is not None:
            XAI_API_KEY = api_key.strip()
            env_updates["XAI_API_KEY"] = XAI_API_KEY
        if model is not None:
            GROK_MODEL = model.strip()
            env_updates["GROK_MODEL"] = GROK_MODEL
        if base_url is not None:
            XAI_BASE_URL = base_url.strip()
            env_updates["XAI_BASE_URL"] = XAI_BASE_URL
    elif p == "gemini":
        if api_key is not None:
            GEMINI_API_KEY = api_key.strip()
            env_updates["GEMINI_API_KEY"] = GEMINI_API_KEY
        if model is not None:
            GEMINI_MODEL = model.strip()
            env_updates["GEMINI_MODEL"] = GEMINI_MODEL
    elif p == "openai":
        if api_key is not None:
            OPENAI_API_KEY = api_key.strip()
            env_updates["OPENAI_API_KEY"] = OPENAI_API_KEY
        if model is not None:
            OPENAI_MODEL = model.strip()
            env_updates["OPENAI_MODEL"] = OPENAI_MODEL
    elif p == "anthropic":
        if api_key is not None:
            ANTHROPIC_API_KEY = api_key.strip()
            env_updates["ANTHROPIC_API_KEY"] = ANTHROPIC_API_KEY
        if model is not None:
            ANTHROPIC_MODEL = model.strip()
            env_updates["ANTHROPIC_MODEL"] = ANTHROPIC_MODEL
    elif p == "ollama":
        if base_url is not None:
            OLLAMA_BASE_URL = base_url.strip()
            env_updates["OLLAMA_BASE_URL"] = OLLAMA_BASE_URL
        if model is not None:
            OLLAMA_MODEL = model.strip()
            env_updates["OLLAMA_MODEL"] = OLLAMA_MODEL

    if env_updates:
        _update_env_file(env_updates)


def remove_provider_credentials(provider: str) -> dict:
    """Clear credentials for a specific provider and switch active provider if necessary."""
    global LLM_PROVIDER, XAI_API_KEY, GEMINI_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY
    p = provider.lower().strip()
    env_updates: dict[str, str] = {}

    if p == "grok":
        XAI_API_KEY = ""
        env_updates["XAI_API_KEY"] = ""
    elif p == "gemini":
        GEMINI_API_KEY = ""
        env_updates["GEMINI_API_KEY"] = ""
    elif p == "openai":
        OPENAI_API_KEY = ""
        env_updates["OPENAI_API_KEY"] = ""
    elif p == "anthropic":
        ANTHROPIC_API_KEY = ""
        env_updates["ANTHROPIC_API_KEY"] = ""

    # If the removed provider was currently active, fall back
    if LLM_PROVIDER.lower().strip() == p:
        # Check if another provider has credentials
        for candidate in ["grok", "openai", "gemini", "anthropic"]:
            if candidate != p:
                details = get_provider_details(candidate)
                if details["is_configured"]:
                    LLM_PROVIDER = candidate
                    env_updates["LLM_PROVIDER"] = candidate
                    break
        else:
            # No other provider configured
            LLM_PROVIDER = "none"
            env_updates["LLM_PROVIDER"] = "none"

    if env_updates:
        _update_env_file(env_updates)

    return get_all_providers_status()


def set_provider_key(key: str, provider: str | None = None) -> None:
    """Backward compatibility wrapper for setting provider key."""
    target_provider = (provider or LLM_PROVIDER).strip().lower()
    save_provider_config(target_provider, api_key=key, set_active=True)



# Whether to condense selected webpage text with the LLM before storing it.
# Each "Remember" costs one extra API call, which matters on the Gemini
# free tier. Set to false to store the selected text as-is.
MEMORY_EXTRACTION = os.getenv("MEMORY_EXTRACTION", "true").strip().lower() in (
    "1",
    "true",
    "yes",
)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-3-5-haiku-latest")

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1")

# --- Provider URL Security & Custom Endpoints ---
# By default, external provider URLs are restricted to official provider domains
# (e.g. api.x.ai, api.openai.com) and Ollama is restricted to local loopback.
ALLOW_CUSTOM_PROVIDER_URLS = os.getenv("ALLOW_CUSTOM_PROVIDER_URLS", "false").strip().lower() in ("1", "true", "yes")
_custom_hosts = os.getenv("ALLOWED_PROVIDER_HOSTS", "").strip()
ALLOWED_PROVIDER_HOSTS = {h.strip().lower() for h in _custom_hosts.split(",") if h.strip()}

BACKEND_HOST = os.getenv("BACKEND_HOST", "127.0.0.1")
BACKEND_PORT = int(os.getenv("BACKEND_PORT", "8000"))

# Custom RECALL topic classifier. It is intentionally separate from embeddings
# and the Gemini answer generator. Missing model artifacts never prevent the
# existing RAG memory pipeline from working.
CLASSIFIER_ENABLED = os.getenv("CLASSIFIER_ENABLED", "true").strip().lower() in ("1", "true", "yes")
CLASSIFIER_THRESHOLD = float(os.getenv("RECALL_CLASSIFIER_THRESHOLD", "0.5"))

# Hugging Face model hosting configuration
HF_MODEL_REPO = os.getenv("HF_MODEL_REPO", "farhanx3rz/memorymind-classifier").strip()
HF_MODEL_REVISION = os.getenv("HF_MODEL_REVISION", "main").strip()
HF_TOKEN = os.getenv("HF_TOKEN", "").strip() or None
# Fine-grained concept extraction is intentionally separate from the trained
# DistilBERT topic taxonomy. Grok produces a small set of reusable concepts.
DYNAMIC_TAGS_ENABLED = os.getenv("DYNAMIC_TAGS_ENABLED", "true").strip().lower() in ("1", "true", "yes")
DYNAMIC_TAG_MIN = int(os.getenv("DYNAMIC_TAG_MIN", "3"))
DYNAMIC_TAG_MAX = int(os.getenv("DYNAMIC_TAG_MAX", "8"))

# Hybrid retrieval combines local semantic similarity with lexical matches and
# the two levels of topic metadata. The weights are intentionally configurable.
HYBRID_SEMANTIC_WEIGHT = float(os.getenv("HYBRID_SEMANTIC_WEIGHT", "0.55"))
HYBRID_KEYWORD_WEIGHT = float(os.getenv("HYBRID_KEYWORD_WEIGHT", "0.20"))
HYBRID_TOPIC_WEIGHT = float(os.getenv("HYBRID_TOPIC_WEIGHT", "0.10"))
HYBRID_TAG_WEIGHT = float(os.getenv("HYBRID_TAG_WEIGHT", "0.15"))
HYBRID_MIN_SCORE = float(os.getenv("HYBRID_MIN_SCORE", "0.25"))
HYBRID_SEMANTIC_CANDIDATES = int(os.getenv("HYBRID_SEMANTIC_CANDIDATES", "48"))
# Document-level results prevent a long article from crowding every other
# source out of the answer. Each source contributes at most this many evidence
# chunks to Grok and the extension.
CHUNKS_PER_SOURCE = int(os.getenv("CHUNKS_PER_SOURCE", "2"))
SOURCE_TOP_K = int(os.getenv("SOURCE_TOP_K", str(TOP_K)))

# Generic webpage ingestion. These values are source-neutral; no website or
# domain receives special treatment in extraction, classification, or search.
WEBPAGE_CHUNK_SIZE = int(os.getenv("WEBPAGE_CHUNK_SIZE", "220"))
WEBPAGE_CHUNK_OVERLAP = int(os.getenv("WEBPAGE_CHUNK_OVERLAP", "40"))
MIN_EXTRACTED_CHARS = int(os.getenv("MIN_EXTRACTED_CHARS", "400"))
WEBPAGE_MAX_HTML_CHARS = int(os.getenv("WEBPAGE_MAX_HTML_CHARS", "10000000"))

# Synthesis timeout and retry configuration
SYNTHESIS_TIMEOUT = float(os.getenv("SYNTHESIS_TIMEOUT", "180.0"))
SYNTHESIS_MAX_RETRIES = int(os.getenv("SYNTHESIS_MAX_RETRIES", "3"))
SYNTHESIS_MIN_RETRY_DELAY = float(os.getenv("SYNTHESIS_MIN_RETRY_DELAY", "1.0"))
SYNTHESIS_MAX_RETRY_DELAY = float(os.getenv("SYNTHESIS_MAX_RETRY_DELAY", "30.0"))
# Maximum input size before triggering chunked synthesis
SYNTHESIS_MAX_INPUT_CHARS = int(os.getenv("SYNTHESIS_MAX_INPUT_CHARS", "50000"))
# Output token limits by style
SYNTHESIS_MAX_TOKENS_QUICK = int(os.getenv("SYNTHESIS_MAX_TOKENS_QUICK", "500"))
SYNTHESIS_MAX_TOKENS_DETAILED = int(os.getenv("SYNTHESIS_MAX_TOKENS_DETAILED", "2000"))
SYNTHESIS_MAX_TOKENS_NOTES = int(os.getenv("SYNTHESIS_MAX_TOKENS_NOTES", "1000"))
# Fallback provider if primary fails (grok -> openai -> gemini)
SYNTHESIS_FALLBACK_ENABLED = os.getenv("SYNTHESIS_FALLBACK_ENABLED", "true").strip().lower() in ("1", "true", "yes")

# Security and CORS restrictions
# Explicit list of permitted Chrome extension IDs. If set, only these IDs can access via CORS.
ALLOWED_EXTENSION_IDS = [
    x.strip() for x in os.getenv("ALLOWED_EXTENSION_IDS", "").split(",") if x.strip()
]
# In development, allows matching unpacked chrome-extension:// origins if ALLOWED_EXTENSION_IDS is empty
ALLOW_DEV_EXTENSIONS = os.getenv("ALLOW_DEV_EXTENSIONS", "true").strip().lower() in (
    "1",
    "true",
    "yes",
)

# Optional local authentication token for the backend API.
# If configured (e.g. via MEMORYMIND_API_KEY or API_AUTH_TOKEN in the environment),
# requests to /memory, /webpage, and /ask require an X-API-Key or Bearer token header.
# If unset, local loopback requests do not require authentication.
API_AUTH_TOKEN = os.getenv("MEMORYMIND_API_KEY", os.getenv("API_AUTH_TOKEN", "")).strip()
