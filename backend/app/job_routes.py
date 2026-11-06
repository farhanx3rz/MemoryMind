"""FastAPI routes for unified research summarization jobs."""
import asyncio
import json
import logging
import uuid
from typing import Any

from fastapi import Depends, File, HTTPException, Request, UploadFile, Form
from starlette.datastructures import UploadFile as StarletteUploadFile
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel

from . import config
from .main import verify_local_auth
from .source_collector import SourceCollection, extract_webpage_sources, extract_pdf_sources
from .summarization_jobs import JobConfig, SummaryStyle, JobStatus, get_job_manager
from .pdf_handler import extract_pdf, validate_and_read_pdf_files

logger = logging.getLogger("memorymind.job_routes")


class CreateJobRequest(BaseModel):
    urls: list[str] = []
    summary_style: str = "detailed"
    topic: str | None = None
    max_length: int | None = None


class CreateJobResponse(BaseModel):
    job_id: str
    status: str
    message: str


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    progress: float
    message: str
    sources_summary: dict[str, Any]
    error: str | None = None
    has_pdf: bool
    pdf_size: int = 0
    filename: str = ""


async def create_job(
    request: CreateJobRequest,
    files: list[UploadFile] | None = None,
    _auth = Depends(verify_local_auth),
) -> CreateJobResponse:
    if not config.is_active():
        raise HTTPException(
            status_code=403,
            detail=f"Provider '{config.LLM_PROVIDER}' is not activated. Please configure your API key to activate it.",
        )

    manager = get_job_manager()
    manager.cleanup_expired()

    total_sources = len(request.urls)
    if files:
        total_sources += len(files)

    if total_sources == 0:
        raise HTTPException(status_code=422, detail="At least one URL or PDF file is required")

    collection = SourceCollection()

    if request.urls:
        extracted = await extract_webpage_sources(request.urls)
        for src in extracted:
            if src.metadata.success:
                collection.add_webpage(url=src.metadata.url, title=src.metadata.title, text=src.text)
            else:
                collection.add_webpage(url=src.metadata.url, error=src.metadata.error)

    if files:
        file_data = await validate_and_read_pdf_files(files)
        for file_bytes, filename in file_data:
            result = await asyncio.to_thread(extract_pdf, file_bytes, filename)
            collection.add_pdf(file_data=file_bytes, filename=filename, result=result)

    if not collection.get_all_sources():
        raise HTTPException(status_code=422, detail="No sources could be successfully extracted")

    try:
        style = SummaryStyle(request.summary_style)
    except Exception:
        style = SummaryStyle.DETAILED

    job_config = JobConfig(summary_style=style, topic=request.topic or "", max_length=request.max_length)
    job = manager.create_job(collection, job_config)
    asyncio.create_task(manager.process_job(job.job_id))

    return CreateJobResponse(job_id=job.job_id, status=job.progress.stage.value, message="Job created")


async def get_job_status(job_id: str, _auth = Depends(verify_local_auth)) -> JobStatusResponse:
    if not config.is_active():
        raise HTTPException(
            status_code=403,
            detail=f"Provider '{config.LLM_PROVIDER}' is not activated. Please configure your API key to activate it.",
        )

    manager = get_job_manager()
    job = manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return JobStatusResponse(
        job_id=job.job_id,
        status=job.progress.stage.value,
        progress=job.progress.progress,
        message=job.progress.message,
        sources_summary=job.sources.get_source_summary(),
        error=job.error,
        has_pdf=len(job.pdf_bytes) > 0,
        pdf_size=len(job.pdf_bytes),
        filename=job.get_report_filename(),
    )


async def download_job_pdf(job_id: str, _auth = Depends(verify_local_auth)) -> Response:
    if not config.is_active():
        raise HTTPException(
            status_code=403,
            detail=f"Provider '{config.LLM_PROVIDER}' is not activated. Please configure your API key to activate it.",
        )

    manager = get_job_manager()
    job = manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if not job.pdf_bytes:
        raise HTTPException(status_code=422, detail="PDF not yet generated")

    filename = job.get_report_filename()
    from urllib.parse import quote
    encoded_filename = quote(filename)
    headers = {
        "Content-Disposition": f'inline; filename="{filename}"; filename*=UTF-8\'\'{encoded_filename}'
    }
    return Response(content=job.pdf_bytes, media_type="application/pdf", headers=headers)


def register_routes(app):
    @app.post("/jobs/create", response_model=CreateJobResponse, dependencies=[Depends(verify_local_auth)])
    async def create_job_endpoint(
        raw_request: Request,
    ):
        content_type = raw_request.headers.get("content-type", "").lower()
        files: list[UploadFile] = []
        if "application/json" in content_type:
            try:
                body = await raw_request.json()
                job_request = CreateJobRequest(**body)
            except Exception as e:
                raise HTTPException(status_code=422, detail=f"Invalid request format: {e}")
        else:
            form = await raw_request.form()
            req_str = form.get("request")
            if not req_str or not isinstance(req_str, str):
                raise HTTPException(status_code=422, detail="Missing 'request' in form data")
            try:
                req_data = json.loads(req_str)
                job_request = CreateJobRequest(**req_data)
            except Exception as e:
                raise HTTPException(status_code=422, detail=f"Invalid request format: {e}")
            uploaded = form.getlist("files")
            for item in uploaded:
                if isinstance(item, (UploadFile, StarletteUploadFile)):
                    files.append(item)

        return await create_job(job_request, files)

    @app.get("/jobs/{job_id}/status", response_model=JobStatusResponse, dependencies=[Depends(verify_local_auth)])
    async def get_status_endpoint(job_id: str):
        return await get_job_status(job_id)

    @app.get("/jobs/{job_id}/pdf", dependencies=[Depends(verify_local_auth)])
    async def get_pdf_endpoint(job_id: str):
        return await download_job_pdf(job_id)

    logger.info("Job routes registered")