"""Job lifecycle management for multi-source summarization.

Handles job creation, progress tracking, synthesis orchestration, and cleanup.
"""
import asyncio
import json
import logging
from pathlib import Path
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from . import config
from .source_collector import SourceCollection, ExtractedSource
from .synthesis import synthesize_documents, extract_citations
from .pdf_generator import generate_pdf_report

logger = logging.getLogger("memorymind.summarization_jobs")


class JobStatus(Enum):
    CREATED = "created"
    EXTRACTING = "extracting"
    ANALYZING = "analyzing"
    SYNTHESIZING = "synthesizing"
    GENERATING_PDF = "generating_pdf"
    COMPLETED = "completed"
    FAILED = "failed"
    EXPIRED = "expired"


class SummaryStyle(Enum):
    QUICK = "quick"
    DETAILED = "detailed"
    NOTES = "notes"


@dataclass
class JobConfig:
    """Configuration for a summarization job."""
    summary_style: SummaryStyle = SummaryStyle.DETAILED
    topic: str = ""
    max_length: int = 2000
    include_citations: bool = True
    include_page_numbers: bool = True


@dataclass
class JobProgress:
    """Progress tracking for a job."""
    stage: JobStatus = JobStatus.CREATED
    message: str = ""
    progress: float = 0.0
    sources_processed: int = 0
    sources_total: int = 0
    started_at: float = 0.0
    completed_at: float = 0.0
    
    def update(self, stage: JobStatus, message: str = "", progress: float = 0.0):
        """Update progress."""
        self.stage = stage
        self.message = message
        if progress > 0:
            self.progress = min(1.0, progress)


@dataclass
class SummarizationJob:
    """Complete summarization job."""
    job_id: str
    created_at: float
    sources: SourceCollection
    config: JobConfig
    progress: JobProgress = field(default_factory=JobProgress)
    synthesis_text: str = ""
    citations: list[dict] = field(default_factory=list)
    pdf_bytes: bytes = b""
    error: str = ""
    
    def is_expired(self, max_age: float = 3600.0) -> bool:
        """Check if job has expired."""
        return (time.time() - self.created_at) > max_age

    def get_style_display_name(self) -> str:
        style = self.config.summary_style
        if style == SummaryStyle.QUICK:
            return "Quick Overview"
        elif style == SummaryStyle.NOTES:
            return "Study Notes"
        else:
            return "Detailed Report"

    def get_base_source_name(self) -> str:
        sources = self.sources.get_all_sources()
        if not sources:
            return "Document"
        first = sources[0].metadata
        if first.filename:
            import re
            return re.sub(r'\.pdf$', '', first.filename, flags=re.IGNORECASE).strip()
        elif first.title:
            return first.title.strip()
        elif first.url:
            return first.domain or "Webpage"
        return "Document"

    def get_report_filename(self) -> str:
        import re
        base = self.get_base_source_name()
        style = self.get_style_display_name()
        full_name = f"{base} - {style}"
        safe_name = re.sub(r'[\\/*?:"<>|]', "", full_name).strip()
        safe_name = re.sub(r'\s+', ' ', safe_name)
        if not safe_name:
            safe_name = "Summary Report"
        return f"{safe_name}.pdf"

    def get_report_title(self) -> str:
        base = self.get_base_source_name()
        style = self.get_style_display_name()
        if self.config.topic:
            return f"{base} — {self.config.topic} ({style})"
        return f"{base} — {style}"
    
    def to_dict(self) -> dict:
        """Serialize for API responses."""
        return {
            "job_id": self.job_id,
            "filename": self.get_report_filename(),
            "status": self.progress.stage.value,
            "message": self.progress.message,
            "progress": self.progress.progress,
            "sources": self.sources.to_dict(),
            "config": {
                "style": self.config.summary_style.value,
                "topic": self.config.topic,
                "max_length": self.config.max_length,
            },
            "has_pdf": len(self.pdf_bytes) > 0,
            "pdf_size": len(self.pdf_bytes),
            "error": self.error,
            "elapsed_seconds": time.time() - self.created_at,
        }


class JobManager:
    """Manages all active summarization jobs with memory and persistent storage."""

    def __init__(self):
        self.jobs: dict[str, SummarizationJob] = {}
        self.cleanup_interval = 600  # 10 minutes
        self.max_job_age = 3600  # 1 hour

    def _get_storage_dir(self) -> Path:
        p = config.BACKEND_DIR / "data" / "jobs"
        p.mkdir(parents=True, exist_ok=True)
        return p

    def _persist_job(self, job: SummarizationJob) -> None:
        try:
            storage_dir = self._get_storage_dir()
            data = job.to_dict()
            json_path = storage_dir / f"{job.job_id}.json"
            temp_path = storage_dir / f"{job.job_id}.tmp"
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(data, f)
            temp_path.replace(json_path)

            if job.pdf_bytes:
                pdf_path = storage_dir / f"{job.job_id}.pdf"
                if not pdf_path.exists() or pdf_path.stat().st_size != len(job.pdf_bytes):
                    pdf_temp = storage_dir / f"{job.job_id}.pdf.tmp"
                    with open(pdf_temp, "wb") as f:
                        f.write(job.pdf_bytes)
                    pdf_temp.replace(pdf_path)
        except Exception as exc:
            logger.warning("Failed to persist job state [%s]: %s", job.job_id, exc)

    def _load_persisted_job(self, job_id: str) -> Optional[SummarizationJob]:
        try:
            storage_dir = self._get_storage_dir()
            json_path = storage_dir / f"{job_id}.json"
            if not json_path.exists():
                return None
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            raw_stage = data.get("status", "failed")
            try:
                stage = JobStatus(raw_stage)
            except Exception:
                stage = JobStatus.FAILED

            error = data.get("error", "")
            message = data.get("message", "")
            # If the backend was restarted while the job was in an intermediate stage
            if stage not in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.EXPIRED):
                stage = JobStatus.FAILED
                error = "Job was interrupted by backend restart"
                message = "Job was interrupted by backend restart"

            config_dict = data.get("config", {})
            try:
                style = SummaryStyle(config_dict.get("style", "detailed"))
            except Exception:
                style = SummaryStyle.DETAILED

            job_cfg = JobConfig(
                summary_style=style,
                topic=config_dict.get("topic", ""),
                max_length=config_dict.get("max_length", 2000),
            )

            job = SummarizationJob(
                job_id=job_id,
                created_at=time.time() - data.get("elapsed_seconds", 0),
                sources=SourceCollection(),
                config=job_cfg,
                error=error,
            )
            job.progress.stage = stage
            job.progress.message = message
            job.progress.progress = data.get("progress", 0.0)

            pdf_path = storage_dir / f"{job_id}.pdf"
            if pdf_path.exists():
                try:
                    with open(pdf_path, "rb") as pf:
                        job.pdf_bytes = pf.read()
                except Exception:
                    pass

            self.jobs[job_id] = job
            if error == "Job was interrupted by backend restart":
                self._persist_job(job)
            return job
        except Exception as exc:
            logger.warning("Failed to load persisted job [%s]: %s", job_id, exc)
            return None

    def create_job(
        self,
        sources: SourceCollection,
        config: JobConfig,
    ) -> SummarizationJob:
        """Create a new summarization job."""
        job_id = str(uuid.uuid4())
        job = SummarizationJob(
            job_id=job_id,
            created_at=time.time(),
            sources=sources,
            config=config,
        )
        job.progress.started_at = time.time()
        self.jobs[job_id] = job
        self._persist_job(job)

        logger.info(
            "Created job [%s] with %d sources",
            job_id, len(sources.sources)
        )
        return job

    def get_job(self, job_id: str) -> Optional[SummarizationJob]:
        """Get job by ID from memory or persistent storage."""
        job = self.jobs.get(job_id)
        if job:
            return job
        return self._load_persisted_job(job_id)

    def delete_job(self, job_id: str):
        """Delete a job from memory and storage."""
        if job_id in self.jobs:
            del self.jobs[job_id]
        try:
            storage_dir = self._get_storage_dir()
            for p in (storage_dir / f"{job_id}.json", storage_dir / f"{job_id}.pdf"):
                if p.exists():
                    p.unlink()
        except Exception:
            pass
        logger.info("Deleted job [%s]", job_id)

    def cleanup_expired(self):
        """Remove expired jobs from memory and persistent disk storage."""
        now = time.time()
        expired = [
            jid for jid, job in self.jobs.items()
            if job.is_expired(self.max_job_age)
        ]
        for jid in expired:
            del self.jobs[jid]

        try:
            storage_dir = self._get_storage_dir()
            for p in storage_dir.glob("*"):
                if p.is_file():
                    mtime = p.stat().st_mtime
                    if now - mtime > self.max_job_age:
                        try:
                            p.unlink()
                        except Exception:
                            pass
        except Exception as exc:
            logger.warning("Error cleaning expired job storage: %s", exc)

        if expired:
            logger.info("Cleaned up %d expired jobs", len(expired))

    async def process_job(self, job_id: str) -> bool:
        """Process a summarization job end-to-end with persistent status updates."""
        job = self.get_job(job_id)
        if not job:
            return False

        try:
            # Stage 1: Validate sources
            job.progress.update(
                JobStatus.EXTRACTING,
                "Validating sources...",
                0.1
            )
            self._persist_job(job)

            successful_sources = job.sources.get_all_sources()
            if not successful_sources:
                job.error = "No sources successfully extracted"
                job.progress.stage = JobStatus.FAILED
                job.progress.message = "No sources successfully extracted"
                self._persist_job(job)
                return False

            job.progress.sources_total = len(successful_sources)

            # Stage 2: Analyze and chunk
            job.progress.update(
                JobStatus.ANALYZING,
                f"Analyzing {len(successful_sources)} sources...",
                0.3
            )
            self._persist_job(job)

            # Stage 3: Synthesize
            job.progress.update(
                JobStatus.SYNTHESIZING,
                "Synthesizing findings across sources...",
                0.5
            )
            self._persist_job(job)

            synthesis_text, citations = await self._synthesize_sources(job)
            job.synthesis_text = synthesis_text
            job.citations = citations

            # Stage 4: Generate PDF
            job.progress.update(
                JobStatus.GENERATING_PDF,
                "Generating PDF report...",
                0.8
            )
            self._persist_job(job)

            pdf_bytes = await self._generate_pdf(job)
            job.pdf_bytes = pdf_bytes

            # Complete
            job.progress.update(
                JobStatus.COMPLETED,
                "Report ready",
                1.0
            )
            job.progress.completed_at = time.time()
            self._persist_job(job)

            logger.info(
                "Job [%s] completed in %.2fs",
                job_id,
                job.progress.completed_at - job.progress.started_at
            )
            return True

        except Exception as e:
            logger.exception("Job [%s] failed", job_id)
            job.error = str(e)
            job.progress.stage = JobStatus.FAILED
            job.progress.message = f"Failed: {e}"
            self._persist_job(job)
            return False
    
    async def _synthesize_sources(
        self,
        job: SummarizationJob
    ) -> tuple[str, list[dict]]:
        """Run synthesis for job sources."""
        import time
        t_start = time.perf_counter()
        
        sources = job.sources.get_all_sources()
        
        # Determine max_tokens based on style and requested max_length for faster generation
        requested_chars = job.config.max_length or 2000
        # 1 token ~= 3-4 chars; give reasonable headroom while avoiding overgeneration
        tokens_from_length = max(200, int(requested_chars / 2.5))
        
        if job.config.summary_style == SummaryStyle.QUICK:
            max_tokens = min(tokens_from_length, config.SYNTHESIS_MAX_TOKENS_QUICK)
        elif job.config.summary_style == SummaryStyle.NOTES:
            max_tokens = min(tokens_from_length, config.SYNTHESIS_MAX_TOKENS_NOTES)
        else:
            max_tokens = min(tokens_from_length, config.SYNTHESIS_MAX_TOKENS_DETAILED)
        
        # Build synthesis prompt based on style
        system_prompt = self._build_system_prompt(job.config)
        user_prompt = self._build_user_prompt(sources, job.config)
        
        # Check input size and warn if large
        input_chars = len(system_prompt) + len(user_prompt)
        source_count = len(sources)
        logger.info(
            "Synthesis starting [sources=%d, input_chars=%d, style=%s, max_tokens=%d]",
            source_count, input_chars, job.config.summary_style.value, max_tokens
        )
        
        if input_chars > config.SYNTHESIS_MAX_INPUT_CHARS:
            logger.warning(
                "Large input detected [chars=%d, limit=%d]. Consider reducing source content.",
                input_chars, config.SYNTHESIS_MAX_INPUT_CHARS
            )
        
        # Call LLM
        from . import llm
        
        if config.LLM_PROVIDER == "grok":
            raw_text = await asyncio.to_thread(
                self._synthesize_with_grok,
                system_prompt,
                user_prompt,
                max_tokens
            )
        elif config.LLM_PROVIDER == "openai":
            raw_text = await asyncio.to_thread(
                self._synthesize_with_openai,
                system_prompt,
                user_prompt,
                max_tokens
            )
        elif config.LLM_PROVIDER == "gemini":
            raw_text = await asyncio.to_thread(
                self._synthesize_with_gemini,
                system_prompt,
                user_prompt,
                max_tokens
            )
        else:
            raw_text = "No LLM configured. Please configure an LLM provider."
        
        elapsed = time.perf_counter() - t_start
        logger.info(
            "Synthesis complete [elapsed=%.2fs, output_chars=%d, sources=%d]",
            elapsed, len(raw_text), source_count
        )
        
        # Extract citations
        citations = self._extract_citations(raw_text, sources)
        
        return raw_text, citations
    
    def _build_system_prompt(self, config: JobConfig) -> str:
        """Build system prompt based on summary style."""
        if config.summary_style == SummaryStyle.QUICK:
            return """You are a research synthesizer creating concise overviews.

Task: Provide a brief summary of the central ideas and key conclusions across all sources.
Be direct and concise. Start immediately with content without conversational filler.

Format: 2-4 paragraphs covering main themes, essential findings, and primary conclusions.

Citations: Use [Source: filename:page N] or [Source: title] format where applicable."""
        
        elif config.summary_style == SummaryStyle.NOTES:
            return """You are a study notes generator creating revision-friendly content.

Task: Extract key concepts, definitions, important facts, and actionable insights.
Be direct and concise. Start immediately with content without conversational filler.

Format:
- Key Concepts (bullet points)
- Important Definitions
- Essential Facts & Statistics  
- Study Tips & Takeaways

Citations: Use [Source: filename:page N] or [Source: title] format where applicable."""
        
        else:  # DETAILED
            return """You are an expert research synthesizer creating comprehensive reports.

Task: Analyze all sources and produce a detailed thematic synthesis.
Be direct and concise. Start immediately with content without conversational filler.

Required sections:
1. Executive Summary
2. Main Themes (organized by topic, not by document)
3. Cross-Source Analysis (agreements, contradictions, complementary findings)
4. Key Findings & Evidence
5. Limitations & Open Questions
6. Conclusions

Citations: Use [Source: filename:page N] or [Source: webpage_title] format.
Identify disagreements between sources and attribute claims clearly."""
    
    def _build_user_prompt(
        self,
        sources: list[ExtractedSource],
        config: JobConfig
    ) -> str:
        """Build user prompt with all source content."""
        sections = []
        
        for idx, source in enumerate(sources, 1):
            meta = source.metadata
            
            if meta.source_type.value == "webpage":
                section = f"""<source id="{idx}" type="webpage">
<title>{meta.title}</title>
<url>{meta.url}</url>
<content>
{source.text[:15000]}
</content>
</source>"""
            else:  # PDF
                section = f"""<source id="{idx}" type="pdf" filename="{meta.filename}">
<pages total="{meta.page_count}">"""
                for page_num, page_text in source.pages.items():
                    clean_page = page_text.strip()
                    if not clean_page:
                        continue
                    section += f"""
<page num="{page_num}">
{clean_page[:4000]}
</page>"""
                section += "\n</pages>\n</source>"
            
            sections.append(section)
        
        instructions = []
        if config.topic:
            instructions.append(f"Focus on: {config.topic}")
        if config.max_length:
            instructions.append(f"Target Length: approximately {config.max_length} characters. Be direct and avoid repetition.")
        
        extra_instruction = ("\n\n" + "\n".join(instructions)) if instructions else ""
        
        return f"""Analyze and synthesize the following sources:{extra_instruction}

{chr(10).join(sections)}

Generate your synthesis following the specified format."""
    
    def _synthesize_with_grok(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int
    ) -> str:
        """Synthesize using Grok with retry logic."""
        import time
        import random
        from openai import APITimeoutError, APIConnectionError, RateLimitError, APIStatusError
        from . import llm
        
        # Log input size
        input_chars = len(system_prompt) + len(user_prompt)
        input_tokens_est = input_chars // 4  # Rough estimate
        logger.info(
            "Grok synthesis starting [model=%s, input_chars=%d, input_tokens_est=%d, max_tokens=%d, timeout=%.1fs]",
            config.GROK_MODEL, input_chars, input_tokens_est, max_tokens, config.SYNTHESIS_TIMEOUT
        )
        
        client = llm._get_grok_synthesis_client()
        
        for attempt in range(config.SYNTHESIS_MAX_RETRIES):
            try:
                t_start = time.perf_counter()
                response = client.chat.completions.create(
                    model=config.GROK_MODEL,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.3,
                    max_tokens=max_tokens,
                    timeout=config.SYNTHESIS_TIMEOUT,
                )
                elapsed = time.perf_counter() - t_start
                
                result = response.choices[0].message.content.strip()
                logger.info(
                    "Grok synthesis succeeded [attempt=%d, elapsed=%.2fs, output_chars=%d]",
                    attempt + 1, elapsed, len(result)
                )
                return result
                
            except APITimeoutError as e:
                elapsed = time.perf_counter() - t_start
                logger.warning(
                    "Grok synthesis timeout [attempt=%d/%d, elapsed=%.2fs]",
                    attempt + 1, config.SYNTHESIS_MAX_RETRIES, elapsed
                )
                if attempt + 1 >= config.SYNTHESIS_MAX_RETRIES:
                    raise RuntimeError(
                        f"Synthesis request timed out after {config.SYNTHESIS_MAX_RETRIES} attempts. "
                        f"Try reducing the number of sources or document size."
                    ) from e
                    
            except (APIConnectionError, RateLimitError) as e:
                logger.warning(
                    "Grok synthesis transient error [attempt=%d/%d, error=%s]",
                    attempt + 1, config.SYNTHESIS_MAX_RETRIES, type(e).__name__
                )
                if attempt + 1 >= config.SYNTHESIS_MAX_RETRIES:
                    raise RuntimeError(
                        f"Synthesis request failed after {config.SYNTHESIS_MAX_RETRIES} attempts: {type(e).__name__}"
                    ) from e
                    
            except APIStatusError as e:
                # Check if retryable (429, 5xx)
                if e.status_code in (429, 500, 502, 503, 504):
                    logger.warning(
                        "Grok synthesis retryable error [attempt=%d/%d, status=%d]",
                        attempt + 1, config.SYNTHESIS_MAX_RETRIES, e.status_code
                    )
                    if attempt + 1 >= config.SYNTHESIS_MAX_RETRIES:
                        raise RuntimeError(
                            f"Synthesis request failed with HTTP {e.status_code} after {config.SYNTHESIS_MAX_RETRIES} attempts"
                        ) from e
                else:
                    # Permanent error (4xx except 429)
                    logger.error("Grok synthesis permanent error [status=%d]: %s", e.status_code, e.message)
                    raise RuntimeError(f"Synthesis request failed: HTTP {e.status_code}") from e
            
            # Exponential backoff with jitter
            if attempt + 1 < config.SYNTHESIS_MAX_RETRIES:
                delay = min(
                    config.SYNTHESIS_MIN_RETRY_DELAY * (2 ** attempt) + random.uniform(0, 1),
                    config.SYNTHESIS_MAX_RETRY_DELAY
                )
                logger.info("Retrying synthesis in %.2fs...", delay)
                time.sleep(delay)
        
        # Try fallback to OpenAI or Gemini if Grok failed and fallback is enabled
        if config.SYNTHESIS_FALLBACK_ENABLED:
            if config.OPENAI_API_KEY:
                logger.info("Grok synthesis failed after %d attempts, attempting fallback to OpenAI", config.SYNTHESIS_MAX_RETRIES)
                try:
                    return self._synthesize_with_openai(
                        system_prompt,
                        user_prompt,
                        max_tokens
                    )
                except Exception as fallback_error:
                    logger.error("OpenAI fallback also failed: %s", str(fallback_error))
            if config.GEMINI_API_KEY:
                logger.info("Attempting fallback to Gemini")
                try:
                    return self._synthesize_with_gemini(
                        system_prompt,
                        user_prompt,
                        max_tokens
                    )
                except Exception as fallback_error:
                    logger.error("Gemini fallback also failed: %s", str(fallback_error))
        
        raise RuntimeError(
            f"Synthesis failed after {config.SYNTHESIS_MAX_RETRIES} attempts with {config.GROK_MODEL} "
            f"(timeout={config.SYNTHESIS_TIMEOUT}s). Please try again or reduce document length."
        )
    
    def _synthesize_with_openai(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int
    ) -> str:
        """Synthesize using OpenAI with retry logic."""
        import time
        import random
        from openai import APITimeoutError, APIConnectionError, RateLimitError, APIStatusError
        from . import llm
        
        # Log input size
        input_chars = len(system_prompt) + len(user_prompt)
        input_tokens_est = input_chars // 4
        logger.info(
            "OpenAI synthesis starting [model=%s, input_chars=%d, input_tokens_est=%d, max_tokens=%d, timeout=%.1fs]",
            config.OPENAI_MODEL, input_chars, input_tokens_est, max_tokens, config.SYNTHESIS_TIMEOUT
        )
        
        client = llm._get_openai_synthesis_client()
        
        for attempt in range(config.SYNTHESIS_MAX_RETRIES):
            try:
                t_start = time.perf_counter()
                response = client.chat.completions.create(
                    model=config.OPENAI_MODEL,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.3,
                    max_tokens=max_tokens,
                    timeout=config.SYNTHESIS_TIMEOUT,
                )
                elapsed = time.perf_counter() - t_start
                
                result = response.choices[0].message.content.strip()
                logger.info(
                    "OpenAI synthesis succeeded [attempt=%d, elapsed=%.2fs, output_chars=%d]",
                    attempt + 1, elapsed, len(result)
                )
                return result
                
            except APITimeoutError as e:
                elapsed = time.perf_counter() - t_start
                logger.warning(
                    "OpenAI synthesis timeout [attempt=%d/%d, elapsed=%.2fs]",
                    attempt + 1, config.SYNTHESIS_MAX_RETRIES, elapsed
                )
                if attempt + 1 >= config.SYNTHESIS_MAX_RETRIES:
                    raise RuntimeError(
                        f"Synthesis request timed out after {config.SYNTHESIS_MAX_RETRIES} attempts. "
                        f"Try reducing the number of sources or document size."
                    ) from e
                    
            except (APIConnectionError, RateLimitError) as e:
                logger.warning(
                    "OpenAI synthesis transient error [attempt=%d/%d, error=%s]",
                    attempt + 1, config.SYNTHESIS_MAX_RETRIES, type(e).__name__
                )
                if attempt + 1 >= config.SYNTHESIS_MAX_RETRIES:
                    raise RuntimeError(
                        f"Synthesis request failed after {config.SYNTHESIS_MAX_RETRIES} attempts: {type(e).__name__}"
                    ) from e
                    
            except APIStatusError as e:
                if e.status_code in (429, 500, 502, 503, 504):
                    logger.warning(
                        "OpenAI synthesis retryable error [attempt=%d/%d, status=%d]",
                        attempt + 1, config.SYNTHESIS_MAX_RETRIES, e.status_code
                    )
                    if attempt + 1 >= config.SYNTHESIS_MAX_RETRIES:
                        raise RuntimeError(
                            f"Synthesis request failed with HTTP {e.status_code} after {config.SYNTHESIS_MAX_RETRIES} attempts"
                        ) from e
                else:
                    logger.error("OpenAI synthesis permanent error [status=%d]: %s", e.status_code, e.message)
                    raise RuntimeError(f"Synthesis request failed: HTTP {e.status_code}") from e
            
            # Exponential backoff with jitter
            if attempt + 1 < config.SYNTHESIS_MAX_RETRIES:
                delay = min(
                    config.SYNTHESIS_MIN_RETRY_DELAY * (2 ** attempt) + random.uniform(0, 1),
                    config.SYNTHESIS_MAX_RETRY_DELAY
                )
                logger.info("Retrying synthesis in %.2fs...", delay)
                time.sleep(delay)
        
        if config.SYNTHESIS_FALLBACK_ENABLED and config.GEMINI_API_KEY:
            logger.info("OpenAI synthesis failed, attempting fallback to Gemini")
            try:
                return self._synthesize_with_gemini(system_prompt, user_prompt, max_tokens)
            except Exception as e:
                logger.error("Gemini fallback also failed: %s", e)
        
        raise RuntimeError(f"Synthesis failed after {config.SYNTHESIS_MAX_RETRIES} attempts with {config.OPENAI_MODEL}")
    
    def _synthesize_with_gemini(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int
    ) -> str:
        """Synthesize using Gemini."""
        from . import llm
        from google.genai import types
        
        client = llm._get_gemini_client()
        response = client.models.generate_content(
            model=config.GEMINI_MODEL,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0.3,
                max_output_tokens=max_tokens,
            ),
        )
        return response.text.strip()
    
    def _extract_citations(
        self,
        text: str,
        sources: list[ExtractedSource]
    ) -> list[dict]:
        """Extract citations from synthesis text.
        
        Returns citations with consistent schema:
        - PDF citations: {"type": "pdf", "filename": str, "page": int, "title": str}
        - Webpage citations: {"type": "webpage", "title": str, "url": str}
        """
        import re
        citations = []
        
        # Build source mapping for validation
        source_map = {}
        for src in sources:
            if src.metadata.source_type.value == "pdf":
                source_map[src.metadata.filename.lower()] = {
                    "filename": src.metadata.filename,
                    "title": src.metadata.title,
                    "type": "pdf"
                }
            else:
                source_map[src.metadata.title.lower()] = {
                    "title": src.metadata.title,
                    "url": src.metadata.url,
                    "type": "webpage"
                }
        
        # Pattern: [Source: filename:page N] or [Source: title]
        pattern = r'\[Source:\s*([^\]]+)\]'
        matches = re.findall(pattern, text)
        
        for match in matches:
            match = match.strip()
            
            # Check if it's a PDF page reference
            if ":page" in match.lower():
                parts = match.split(":page")
                if len(parts) == 2:
                    source_name = parts[0].strip()
                    try:
                        page_num = int(parts[1].strip())
                        
                        # Validate against actual source filenames
                        source_key = source_name.lower()
                        if source_key in source_map and source_map[source_key]["type"] == "pdf":
                            citations.append({
                                "type": "pdf",
                                "filename": source_map[source_key]["filename"],
                                "page": page_num,
                                "title": source_map[source_key]["title"],
                            })
                        else:
                            # Fallback for unmatched PDF citation
                            citations.append({
                                "type": "pdf",
                                "filename": source_name,
                                "page": page_num,
                                "title": source_name,
                            })
                    except ValueError:
                        # Invalid page number, treat as text
                        citations.append({
                            "type": "webpage",
                            "title": match,
                            "url": "",
                        })
            else:
                # Webpage or title-only citation
                source_key = match.lower()
                if source_key in source_map and source_map[source_key]["type"] == "webpage":
                    citations.append({
                        "type": "webpage",
                        "title": source_map[source_key]["title"],
                        "url": source_map[source_key]["url"],
                    })
                else:
                    # Unmatched citation
                    citations.append({
                        "type": "webpage",
                        "title": match,
                        "url": "",
                    })
        
        return citations
    
    async def _generate_pdf(self, job: SummarizationJob) -> bytes:
        """Generate PDF report for job."""
        sources = job.sources.get_all_sources()
        
        # Convert ExtractedSource to format expected by pdf_generator
        pdf_sources = []
        for src in sources:
            if src.metadata.source_type.value == "pdf":
                # Mock ExtractedPDF object
                from .pdf_handler import ExtractedPDF, PDFPage
                pdf_obj = ExtractedPDF(
                    filename=src.metadata.filename,
                    total_pages=src.metadata.page_count,
                    extracted_pages=[
                        PDFPage(page_num=pnum, text=ptext)
                        for pnum, ptext in src.pages.items()
                    ],
                    extraction_status="completed"
                )
                pdf_obj.error = ""
                pdf_sources.append(pdf_obj)
        
        title = job.get_report_title()
        
        pdf_bytes = await asyncio.to_thread(
            generate_pdf_report,
            job.synthesis_text,
            pdf_sources,
            job.citations,
            title,
            sources  # Pass full source metadata
        )
        
        return pdf_bytes


# Global job manager instance
_job_manager = JobManager()


def get_job_manager() -> JobManager:
    """Get global job manager."""
    return _job_manager
