"""MemoryMind API: the backend the Chrome extension talks to.

Run it with:
    uvicorn app.main:app --reload

The LLM API key is read from the environment on this server only.
It is never included in any response.
"""

import asyncio
import hmac
import json
import logging
import re
import time
from typing import Any

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Query, Security, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.security import APIKeyHeader
from contextlib import asynccontextmanager
from pydantic import BaseModel, Field, field_validator

from . import config, llm, memory_store, pipeline
from .webpage import from_extracted_text, from_html, prune_html_bloat, validate_source_url

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("memorymind")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Pre-warm ML models during server startup so incoming user requests never pay cold-load penalty
    try:
        pipeline.get_topic_classifier()._load()
        logger.info("Topic classifier preloaded successfully.")
    except Exception as exc:
        logger.warning("Topic classifier preloading skipped or failed: %s", exc)

    try:
        memory_store.get_embedding_model()
        logger.info("Embedding model preloaded successfully.")
    except Exception as exc:
        logger.warning("Embedding model preloading failed: %s", exc)

    # Periodic background cleanup of expired PDF sessions and research jobs
    cleanup_stop_event = asyncio.Event()

    async def _periodic_cleanup():
        while not cleanup_stop_event.is_set():
            try:
                await asyncio.sleep(600)  # 10 minutes
                if cleanup_stop_event.is_set():
                    break
                from . import pdf_routes, summarization_jobs
                pdf_routes.cleanup_expired_sessions()
                summarization_jobs.get_job_manager().cleanup_expired()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.warning("Periodic cleanup error: %s", exc)

    cleanup_task = asyncio.create_task(_periodic_cleanup())

    yield

    cleanup_stop_event.set()
    cleanup_task.cancel()
    try:
        await cleanup_task
    except (asyncio.CancelledError, Exception):
        pass


app = FastAPI(
    title="MemoryMind",
    description="Stores text as memories and answers questions about them.",
    version="3.0.0",
    lifespan=lifespan,
)


def get_cors_origin_regex() -> str:
    """Build CORS origin regex based on configured extension IDs and loopback hosts."""
    dev_pattern = r"https?://(localhost|127\.0\.0\.1)(:\d+)?"
    if config.ALLOWED_EXTENSION_IDS:
        ext_ids = "|".join(re.escape(i) for i in config.ALLOWED_EXTENSION_IDS)
        return rf"^(chrome-extension://({ext_ids})|{dev_pattern})$"
    if config.ALLOW_DEV_EXTENSIONS:
        logger.debug(
            "CORS allows development chrome-extension origins. "
            "Configure ALLOWED_EXTENSION_IDS for production."
        )
        return rf"^(chrome-extension://[a-p]{{32}}|{dev_pattern})$"
    return rf"^{dev_pattern}$"


# The Chrome extension runs on a chrome-extension:// origin and local web
# tooling is served from http://127.0.0.1:8000 or localhost.
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=get_cors_origin_regex(),
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def verify_local_auth(
    x_api_key: str | None = Security(api_key_header),
    authorization: str | None = Header(default=None),
) -> None:
    """Lightweight local authentication mechanism.

    If API_AUTH_TOKEN is configured in the environment, verifies X-API-Key
    or Authorization: Bearer <token> headers. Query-parameter tokens are
    intentionally NOT accepted to prevent credential leakage in logs,
    referrers, and browser history.
    If unset, local loopback requests proceed.
    """
    expected = config.API_AUTH_TOKEN
    if not expected:
        return

    # Coerce FastAPI Query/Header descriptor objects to None when called
    # directly in tests (outside FastAPI's dependency-injection framework).
    _x_api_key = x_api_key if isinstance(x_api_key, str) else None
    _authorization = authorization if isinstance(authorization, str) else None

    if _x_api_key and hmac.compare_digest(_x_api_key.strip(), expected):
        return

    if _authorization:
        parts = _authorization.strip().split()
        if len(parts) == 2 and parts[0].lower() == "bearer":
            if hmac.compare_digest(parts[1].strip(), expected):
                return

    raise HTTPException(
        status_code=401,
        detail="Unauthorized. A valid X-API-Key or Bearer token is required.",
    )


NO_MEMORY_ANSWER = "I don't have a memory about that."

# --- Request and response shapes ------------------------------------------


class MemoryRequest(BaseModel):
    text: str = Field(..., description="The text to remember.")
    source_url: str | None = Field(default=None, description="Where the text came from.")
    source_title: str | None = Field(default=None, description="Title of that page.")
    visited_at: str | None = Field(
        default=None,
        description="When this page/memory was read or captured, as an ISO-8601 timestamp.",
    )

    @field_validator("text")
    @classmethod
    def text_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("text must not be empty")
        return value

    @field_validator("source_url")
    @classmethod
    def validate_memory_source_url(cls, value: str | None) -> str | None:
        val = (value or "").strip()
        if not val:
            return None
        return validate_source_url(val, allow_none=True)

    @field_validator("source_title", "visited_at")
    @classmethod
    def blank_becomes_none(cls, value: str | None) -> str | None:
        value = (value or "").strip()
        return value or None


class AskRequest(BaseModel):
    question: str = Field(..., description="The question to answer.")

    @field_validator("question")
    @classmethod
    def question_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("question must not be empty")
        return value


class WebpageRequest(BaseModel):
    url: str = Field(..., description="The webpage URL.")
    title: str | None = Field(default=None, description="Optional page title.")
    html: str | None = Field(default=None, description="Current webpage HTML.")
    extracted_text: str | None = Field(default=None, description="Already-extracted main text.")
    visited_at: str | None = Field(
        default=None,
        description="When this page was read or captured, as an ISO-8601 timestamp.",
    )

    @field_validator("url")
    @classmethod
    def validate_webpage_url(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("url must not be empty")
        result = validate_source_url(value, allow_none=False)
        if not result:
            raise ValueError("A valid http or https URL is required.")
        return result

    @field_validator("title", "html", "extracted_text", "visited_at")
    @classmethod
    def optional_blank_becomes_none(cls, value: str | None) -> str | None:
        value = (value or "").strip()
        return value or None

    def model_post_init(self, __context) -> None:
        if not self.html and not self.extracted_text:
            raise ValueError("Provide html or extracted_text")


class TopicTag(BaseModel):
    label: str
    score: float


class DynamicTag(BaseModel):
    label: str
    score: float = 1.0


class WebpageResponse(BaseModel):
    success: bool
    id: str
    document_id: str
    url: str
    title: str
    domain: str | None
    timestamp: str
    visited_at: str
    chunks: int
    topics: list[TopicTag] = Field(default_factory=list)
    tags: list[DynamicTag] = Field(default_factory=list)


class StoredMemory(BaseModel):
    text: str
    score: float
    semantic_score: float = 0.0
    keyword_score: float = 0.0
    topic_score: float = 0.0
    tag_score: float = 0.0
    source_url: str | None = None
    source_title: str | None = None
    source_domain: str | None = None
    timestamp: str | None = None
    visited_at: str | None = None
    document_id: str | None = None
    chunk_index: int = 0
    topics: list[TopicTag] = Field(default_factory=list)
    tags: list[DynamicTag] = Field(default_factory=list)


class MemoryResponse(BaseModel):
    success: bool
    id: str
    document_id: str
    memory: str
    source_url: str | None = None
    source_title: str | None = None
    visited_at: str
    extracted: bool = False
    topics: list[TopicTag] = Field(default_factory=list)
    tags: list[DynamicTag] = Field(default_factory=list)


class SourceSnippet(BaseModel):
    text: str
    score: float
    chunk_index: int = 0


class SourceResponse(BaseModel):
    document_id: str
    title: str
    url: str | None = None
    domain: str | None = None
    visited_at: str | None = None
    score: float
    chunk_count: int = 0
    snippet: str = ""
    snippets: list[SourceSnippet] = Field(default_factory=list)
    topics: list[TopicTag] = Field(default_factory=list)
    tags: list[DynamicTag] = Field(default_factory=list)
    source_index: int = 0
    is_cited: bool = False


class RelatedSource(BaseModel):
    document_id: str
    title: str
    url: str | None = None


class RelatedConceptResponse(BaseModel):
    label: str
    kind: str
    source_count: int
    sources: list[RelatedSource] = Field(default_factory=list)


class AskResponse(BaseModel):
    answer: str
    memories: list[StoredMemory]
    sources: list[SourceResponse] = Field(default_factory=list)
    related_concepts: list[RelatedConceptResponse] = Field(default_factory=list)
    cited_sources: list[int] = Field(default_factory=list)


class ActivationRequest(BaseModel):
    api_key: str = Field(..., min_length=1, description="API key to activate the LLM provider.")
    provider: str | None = Field(default=None, description="Optional provider override (e.g. grok, gemini, openai).")


class ProviderTestRequest(BaseModel):
    provider: str = Field(..., description="Provider name (grok, openai, gemini, anthropic, ollama, none).")
    api_key: str | None = Field(default=None, description="Optional API key to test.")
    model: str | None = Field(default=None, description="Optional model to test.")
    base_url: str | None = Field(default=None, description="Optional base URL to test.")


class ProviderConfigureRequest(BaseModel):
    provider: str = Field(..., description="Provider name to configure and set active.")
    api_key: str | None = Field(default=None, description="API key for the provider.")
    model: str | None = Field(default=None, description="Model for the provider.")
    base_url: str | None = Field(default=None, description="Base URL for the provider.")
    test_first: bool = Field(default=True, description="Whether to test connection before applying configuration.")


class ProviderRemoveRequest(BaseModel):
    provider: str = Field(..., description="Provider name to remove credentials for.")


# --- Endpoints -------------------------------------------------------------


@app.get("/health")
def health() -> dict:
    """Liveness check and provider activation status."""
    return {
        "status": "ok",
        "provider": config.LLM_PROVIDER,
        "is_activated": config.is_active(),
        "connection_status": "ACTIVE" if config.is_active() else "NOT_CONFIGURED",
    }


@app.get("/provider/status", dependencies=[Depends(verify_local_auth)])
def get_provider_status() -> dict:
    """Return all provider details, configuration states, and active connection status."""
    return config.get_all_providers_status()


@app.post("/provider/test", dependencies=[Depends(verify_local_auth)])
def test_provider(request: ProviderTestRequest) -> dict:
    """Test connection for a specific provider without modifying saved configuration."""
    success, message = llm.test_provider_connection(
        provider=request.provider,
        api_key=request.api_key,
        model=request.model,
        base_url=request.base_url,
    )
    if not success:
        raise HTTPException(status_code=400, detail=message)
    return {"success": True, "message": message}


@app.post("/provider/configure", dependencies=[Depends(verify_local_auth)])
def configure_provider(request: ProviderConfigureRequest) -> dict:
    """Configure, test, and activate a provider transactionally.

    If test_first is True and the test fails, configuration is left untouched.
    """
    p = request.provider.lower().strip()

    if request.test_first:
        success, message = llm.test_provider_connection(
            provider=p,
            api_key=request.api_key,
            model=request.model,
            base_url=request.base_url,
        )
        if not success:
            raise HTTPException(status_code=400, detail=f"Activation test failed: {message}")

    config.save_provider_config(
        provider=p,
        api_key=request.api_key,
        model=request.model,
        base_url=request.base_url,
        set_active=True,
    )
    llm.reset_clients()
    logger.info("Provider '%s' successfully configured and activated.", p)

    status_data = config.get_all_providers_status()
    return {
        "status": "ok",
        "message": f"Provider '{p}' successfully activated.",
        **status_data,
    }


@app.post("/provider/remove", dependencies=[Depends(verify_local_auth)])
def remove_provider(request: ProviderRemoveRequest) -> dict:
    """Remove credentials for a provider and reset active clients."""
    result = config.remove_provider_credentials(request.provider)
    llm.reset_clients()
    logger.info("Credentials removed for provider '%s'.", request.provider)
    return result


@app.post("/activate", dependencies=[Depends(verify_local_auth)])
def activate_provider(request: ActivationRequest) -> dict:
    """Activate the backend with an API key dynamically without manual server restarts."""
    cleaned_key = request.api_key.strip()
    if not cleaned_key:
        raise HTTPException(status_code=400, detail="API key cannot be empty.")

    target_provider = (request.provider or config.LLM_PROVIDER or "grok").strip().lower()

    config.save_provider_config(target_provider, api_key=cleaned_key, set_active=True)
    llm.reset_clients()
    logger.info("Provider '%s' successfully activated with new API key.", target_provider)

    return {
        "status": "ok",
        "message": f"Successfully activated provider '{config.LLM_PROVIDER}'.",
        "provider": config.LLM_PROVIDER,
        "is_activated": True,
    }




@app.post("/memory", response_model=MemoryResponse, dependencies=[Depends(verify_local_auth)])
def create_memory(request: MemoryRequest) -> MemoryResponse:
    """Store selected text with DistilBERT topics and Grok dynamic tags."""
    t_start = time.perf_counter()
    original_text = memory_store.clean_text(request.text)

    memory_text = original_text
    extracted = False

    t_condense_start = time.perf_counter()
    try:
        condensed = llm.extract_memory(original_text)
        if condensed:
            memory_text = condensed
            extracted = True
    except Exception as error:
        # Losing the LLM must never cost the user their memory.
        logger.warning("Memory extraction failed, storing the raw text: %s", error)
    condense_ms = (time.perf_counter() - t_condense_start) * 1000

    try:
        document = from_extracted_text(
            url=request.source_url or "",
            title=request.source_title or "Untitled webpage",
            text=memory_text,
            timestamp=request.visited_at,
            metadata={
                "source_type": "selection",
                "original_selection": original_text[:1000],
            },
        )
        # Classify and tag the user's actual selection. Dynamic tags are based
        # on the original selection even when memory condensation is enabled.
        t_ingest_start = time.perf_counter()
        document, memory_id = pipeline.ingest_document(
            document,
            dynamic_tag_text=original_text,
        )
        ingest_ms = (time.perf_counter() - t_ingest_start) * 1000

        # If dynamic tags were already persisted for this URL, reuse them immediately
        if not document.tags and request.source_url:
            existing_tags = memory_store.get_tags_for_url(request.source_url)
            if existing_tags:
                document.tags = existing_tags
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except Exception as error:
        logger.exception("Failed to store memory")
        raise HTTPException(
            status_code=500, detail="Could not store the memory. See server logs."
        ) from error

    total_ms = (time.perf_counter() - t_start) * 1000
    logger.info(
        "Structured Timing: create_memory [extracted=%s, condense_ms=%.2f, ingest_ms=%.2f, total_ms=%.2f, topics=%d, tags=%d]",
        extracted, condense_ms, ingest_ms, total_ms, len(document.topics), len(document.tags),
    )

    return MemoryResponse(
        success=True,
        id=memory_id,
        document_id=document.document_id or memory_id,
        memory=memory_text,
        source_url=request.source_url,
        source_title=request.source_title,
        visited_at=document.timestamp,
        extracted=extracted,
        topics=[TopicTag(**topic) for topic in document.topics],
        tags=[DynamicTag(**tag) for tag in document.tags],
    )


@app.post("/webpage", response_model=WebpageResponse, dependencies=[Depends(verify_local_auth)])
def create_webpage_memory(request: WebpageRequest) -> WebpageResponse:
    """Run the generic webpage pipeline for a complete current page.

    Wikipedia, blogs, docs, news sites, tutorials, and other webpages all use
    this exact endpoint and WebDocument representation. Only content extraction
    knows about HTML/boilerplate; the ML/RAG stack is source-agnostic.
    """
    t_start = time.perf_counter()
    hard_limit = max(config.WEBPAGE_MAX_HTML_CHARS * 5, 50_000_000)
    if request.html and len(request.html) > hard_limit:
        raise HTTPException(
            status_code=413,
            detail=f"Webpage HTML exceeds the absolute maximum payload limit ({hard_limit:,} characters).",
        )

    if request.html and len(request.html) > config.WEBPAGE_MAX_HTML_CHARS:
        orig_len = len(request.html)
        request.html = prune_html_bloat(request.html, max_chars=config.WEBPAGE_MAX_HTML_CHARS)
        logger.info(
            "Pruned large webpage HTML from %d to %d characters (limit=%d).",
            orig_len,
            len(request.html),
            config.WEBPAGE_MAX_HTML_CHARS,
        )

    t_extract_start = time.perf_counter()
    try:
        if request.html:
            document = from_html(
                url=request.url,
                title=request.title or "",
                html=request.html,
                timestamp=request.visited_at,
            )
        else:
            document = from_extracted_text(
                url=request.url,
                title=request.title or "Untitled webpage",
                text=request.extracted_text or "",
                timestamp=request.visited_at,
            )
        extract_ms = (time.perf_counter() - t_extract_start) * 1000

        t_ingest_start = time.perf_counter()
        document, memory_id = pipeline.ingest_document(document)
        ingest_ms = (time.perf_counter() - t_ingest_start) * 1000
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except Exception as error:
        logger.exception("Failed to ingest webpage")
        raise HTTPException(
            status_code=500, detail="Could not ingest the webpage. See server logs."
        ) from error

    # If dynamic tags were already persisted for this URL, reuse them immediately
    if not document.tags:
        existing_tags = memory_store.get_tags_for_url(document.url)
        if existing_tags:
            document.tags = existing_tags

    total_ms = (time.perf_counter() - t_start) * 1000
    logger.info(
        "Structured Timing: remember_webpage [chunks=%d, topics=%d, tags=%d, extract_ms=%.2f, ingest_ms=%.2f, total_ms=%.2f]",
        len(document.chunks), len(document.topics), len(document.tags), extract_ms, ingest_ms, total_ms,
    )

    return WebpageResponse(
        success=True,
        id=memory_id,
        document_id=document.document_id or memory_id,
        url=document.url,
        title=document.title,
        domain=document.source or None,
        timestamp=document.timestamp,
        visited_at=document.timestamp,
        chunks=len(document.chunks),
        topics=[TopicTag(**topic) for topic in document.topics],
        tags=[DynamicTag(**tag) for tag in document.tags],
    )


@app.post("/memory/pdf", dependencies=[Depends(verify_local_auth)])
async def create_pdf_memory(
    file: UploadFile = File(..., description="PDF file to remember"),
    source_url: str | None = Form(default=None),
    source_title: str | None = Form(default=None),
) -> dict[str, Any]:
    """Ingest a PDF file into memory store with topics and dynamic tags."""
    from .pdf_handler import read_and_validate_pdf_upload, extract_pdf

    file_bytes = await read_and_validate_pdf_upload(file)
    filename = file.filename or "uploaded.pdf"
    extracted_pdf = await asyncio.to_thread(extract_pdf, file_bytes, filename)
    if not extracted_pdf.success:
        raise HTTPException(
            status_code=422,
            detail=extracted_pdf.error or "Failed to extract text from PDF document",
        )

    full_text = extracted_pdf.get_full_text()
    if not full_text.strip():
        raise HTTPException(status_code=422, detail="No readable text found in PDF.")

    doc_title = (source_title or filename).strip()
    url_val = (source_url or "").strip()

    document = from_extracted_text(
        url=url_val,
        title=doc_title,
        text=full_text,
    )
    document.metadata["source_type"] = "pdf"
    document.metadata["filename"] = filename
    document.metadata["page_count"] = extracted_pdf.page_count

    ingested_doc, memory_id = await asyncio.to_thread(pipeline.ingest_document, document)
    return {
        "success": True,
        "id": memory_id,
        "document_id": ingested_doc.document_id or memory_id,
        "url": url_val,
        "title": doc_title,
        "page_count": extracted_pdf.page_count,
        "chunks": len(ingested_doc.chunks),
        "topics": [TopicTag(**topic) for topic in ingested_doc.topics],
        "tags": [DynamicTag(**tag) for tag in ingested_doc.tags],
    }


@app.get("/memory/{document_id}/tags", dependencies=[Depends(verify_local_auth)])
def get_memory_tags(document_id: str) -> dict[str, Any]:
    """Retrieve dynamic tags and topics for an ingested document or URL."""
    info = memory_store.get_document_info(document_id=document_id)
    if info:
        return {
            "document_id": info["document_id"],
            "url": info["url"],
            "title": info["title"],
            "topics": [TopicTag(**t) for t in info["topics"] if isinstance(t, dict) and "label" in t],
            "tags": [DynamicTag(**t) for t in info["tags"] if isinstance(t, dict) and "label" in t],
            "ready": info["ready"],
            "saved": True,
        }
    return {
        "document_id": document_id,
        "url": "",
        "title": "",
        "topics": [],
        "tags": [],
        "ready": False,
        "saved": False,
    }


@app.get("/tags", dependencies=[Depends(verify_local_auth)])
def get_tags(
    url: str | None = None,
    document_id: str | None = None,
) -> dict[str, Any]:
    """Retrieve dynamic tags and topics by URL or document_id."""
    info = memory_store.get_document_info(document_id=document_id, url=url)
    if info:
        return {
            "document_id": info["document_id"],
            "url": info["url"],
            "title": info["title"],
            "topics": [TopicTag(**t) for t in info["topics"] if isinstance(t, dict) and "label" in t],
            "tags": [DynamicTag(**t) for t in info["tags"] if isinstance(t, dict) and "label" in t],
            "ready": info["ready"],
            "saved": True,
        }
    return {
        "document_id": document_id or "",
        "url": url or "",
        "title": "",
        "topics": [],
        "tags": [],
        "ready": False,
        "saved": False,
    }


@app.post("/ask", response_model=AskResponse, dependencies=[Depends(verify_local_auth)])
def ask(request: AskRequest) -> AskResponse:
    """Hybrid-retrieve memories, then generate the grounded Grok RAG answer."""
    if not config.is_active():
        raise HTTPException(
            status_code=403,
            detail=f"Provider '{config.LLM_PROVIDER}' is not activated. Please configure your API key to activate it.",
        )

    t_start = time.perf_counter()
    try:
        t_search_start = time.perf_counter()
        matches = memory_store.search_memories(request.question)
        search_ms = (time.perf_counter() - t_search_start) * 1000
    except Exception as error:
        logger.exception("Failed to search memories")
        raise HTTPException(
            status_code=500, detail="Could not search memories. See server logs."
        ) from error

    # Nothing relevant, so do not ask the LLM to invent an answer.
    if not matches:
        total_ms = (time.perf_counter() - t_start) * 1000
        logger.info(
            "Structured Timing: ask [matches=0, search_ms=%.2f, total_ms=%.2f]",
            search_ms, total_ms,
        )
        return AskResponse(answer=NO_MEMORY_ANSWER, memories=[])

    t_llm_start = time.perf_counter()
    try:
        answer, cited_indices = llm.generate_answer_with_citations(request.question, matches)
        llm_ms = (time.perf_counter() - t_llm_start) * 1000
    except Exception as error:
        if isinstance(error, TimeoutError) or "timeout" in str(error).lower():
            logger.warning("LLM request timed out: %s", error)
            raise HTTPException(
                status_code=504,
                detail="The request to the language model timed out. Try again later.",
            ) from error
        if llm.is_quota_error(error):
            logger.warning("LLM quota exhausted: %s", type(error).__name__)
            raise HTTPException(
                status_code=429,
                detail=(
                    "The configured LLM provider has rejected the request due to "
                    "quota or rate limits. Check your provider account and limits."
                ),
            ) from error
        logger.exception("LLM call failed")
        raise HTTPException(
            status_code=502,
            detail="The language model could not be reached. See server logs.",
        ) from error

    stored_memories = [
        StoredMemory(
            text=match["text"],
            score=match["score"],
            semantic_score=match.get("semantic_score", 0.0),
            keyword_score=match.get("keyword_score", 0.0),
            topic_score=match.get("topic_score", 0.0),
            tag_score=match.get("tag_score", 0.0),
            # ChromaDB stores "" for a missing value; the API uses null.
            source_url=match["source_url"] or None,
            source_title=match["source_title"] or None,
            source_domain=match.get("source_domain") or None,
            timestamp=match.get("timestamp") or None,
            visited_at=match.get("visited_at") or match.get("timestamp") or None,
            document_id=match.get("document_id") or None,
            chunk_index=int(match.get("chunk_index", 0)),
            topics=[TopicTag(**topic) for topic in match.get("topics", [])],
            tags=[DynamicTag(**tag) for tag in match.get("tags", [])],
        )
        for match in matches
    ]

    source_rows = memory_store.build_source_results(
        matches,
        limit=config.SOURCE_TOP_K,
    )
    related_rows = memory_store.build_related_concepts(source_rows)

    sources: list[SourceResponse] = []
    for idx, row in enumerate(source_rows, start=1):
        sources.append(
            SourceResponse(
                document_id=row["document_id"],
                title=row["title"],
                url=row["url"] or None,
                domain=row["domain"] or None,
                visited_at=row["visited_at"] or None,
                score=row["score"],
                chunk_count=row["chunk_count"],
                snippet=row["snippet"],
                snippets=[SourceSnippet(**snippet) for snippet in row["snippets"]],
                topics=[TopicTag(**topic) for topic in row["topics"]],
                tags=[DynamicTag(**tag) for tag in row["tags"]],
                source_index=idx,
                is_cited=(idx in cited_indices),
            )
        )

    related_concepts = [
        RelatedConceptResponse(
            label=row["label"],
            kind=row["kind"],
            source_count=row["source_count"],
            sources=[RelatedSource(**source) for source in row["sources"]],
        )
        for row in related_rows
    ]

    total_ms = (time.perf_counter() - t_start) * 1000
    logger.info(
        "Structured Timing: ask [matches=%d, cited=%d, search_ms=%.2f, llm_ms=%.2f, total_ms=%.2f]",
        len(matches), len(cited_indices), search_ms, llm_ms, total_ms,
    )

    return AskResponse(
        answer=answer,
        memories=stored_memories,
        sources=sources,
        related_concepts=related_concepts,
        cited_sources=cited_indices,
    )


@app.post("/ask/stream", dependencies=[Depends(verify_local_auth)])
def ask_stream(request: AskRequest):
    """Stream answer tokens followed by sources metadata event."""
    if not config.is_active():
        raise HTTPException(
            status_code=403,
            detail=f"Provider '{config.LLM_PROVIDER}' is not activated. Please configure your API key to activate it.",
        )

    t_start = time.perf_counter()
    try:
        matches = memory_store.search_memories(request.question)
    except Exception as error:
        logger.exception("Failed to search memories")
        raise HTTPException(status_code=500, detail="Could not search memories.") from error

    if not matches:
        def empty_generator():
            yield json.dumps({"type": "token", "token": NO_MEMORY_ANSWER}) + "\n"
            yield json.dumps({"type": "done", "sources": [], "related_concepts": [], "cited_sources": []}) + "\n"
        return StreamingResponse(empty_generator(), media_type="application/x-ndjson")

    source_rows = memory_store.build_source_results(matches, limit=config.SOURCE_TOP_K)
    related_rows = memory_store.build_related_concepts(source_rows)

    def event_stream():
        full_text = []
        for token in llm.stream_answer_tokens(request.question, matches):
            full_text.append(token)
            yield json.dumps({"type": "token", "token": token}) + "\n"

        complete_answer = "".join(full_text)
        cleaned_answer, cited_indices = llm.sanitize_citations(complete_answer, len(source_rows))

        sources_payload = [
            {
                "document_id": row["document_id"],
                "title": row["title"],
                "url": row["url"] or None,
                "domain": row["domain"] or None,
                "visited_at": row["visited_at"] or None,
                "score": row["score"],
                "chunk_count": row["chunk_count"],
                "snippet": row["snippet"],
                "snippets": row["snippets"],
                "topics": row["topics"],
                "tags": row["tags"],
                "source_index": idx,
                "is_cited": (idx in cited_indices),
            }
            for idx, row in enumerate(source_rows, start=1)
        ]
        related_payload = [
            {
                "label": row["label"],
                "kind": row["kind"],
                "source_count": row["source_count"],
                "sources": row["sources"],
            }
            for row in related_rows
        ]
        total_ms = (time.perf_counter() - t_start) * 1000
        logger.info(
            "Structured Timing: ask_stream [matches=%d, cited=%d, total_ms=%.2f]",
            len(matches), len(cited_indices), total_ms,
        )
        yield json.dumps({
            "type": "done",
            "sources": sources_payload,
            "related_concepts": related_payload,
            "cited_sources": cited_indices,
            "cleaned_answer": cleaned_answer,
        }) + "\n"

    return StreamingResponse(event_stream(), media_type="application/x-ndjson")


# Register PDF summarization routes
from . import pdf_routes
pdf_routes.register_routes(app)

# Register job routes
from . import job_routes
job_routes.register_routes(app)


