"""FastAPI routes for PDF summarization feature.

Provides endpoints for:
- Uploading and extracting text from PDFs
- Synthesizing findings across documents
- Generating and downloading the final PDF report
"""
import asyncio
import io
import logging
import time
import uuid
from typing import Any

from fastapi import Depends, File, HTTPException, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel, Field

from . import config
from .main import verify_local_auth
from .pdf_handler import (
    ExtractedPDF,
    extract_multiple_pdfs,
    validate_extraction_results,
    results_to_dict,
    validate_and_read_pdf_files,
)
from .synthesis import synthesize_documents, extract_citations
from .pdf_generator import generate_pdf_report

logger = logging.getLogger("memorymind.pdf_routes")

# In-memory session storage (cleared on server restart)
_pdf_sessions: dict[str, dict[str, Any]] = {}


class PDFExtractionResponse(BaseModel):
    """Response after extracting PDFs."""
    session_id: str
    pdfs: list[dict]
    total_pages: int
    total_extracted: int
    all_successful: bool


class SynthesisStatusResponse(BaseModel):
    """Response for synthesis status check."""
    session_id: str
    status: str
    progress: float
    message: str


class PDFDownloadResponse(BaseModel):
    """Response for PDF download info."""
    session_id: str
    download_url: str
    filename: str


async def extract_pdfs(
    files: list[UploadFile] = File(..., description="PDF files to extract"),
    _auth = Depends(verify_local_auth),
) -> PDFExtractionResponse:
    """Extract text from multiple uploaded PDF files."""
    cleanup_expired_sessions()
    t_start = time.perf_counter()
    session_id = str(uuid.uuid4())

    if not files:
        raise HTTPException(status_code=422, detail="No files provided")

    file_data = await validate_and_read_pdf_files(files)
    results = await asyncio.to_thread(extract_multiple_pdfs, file_data)

    is_valid, error_msg = validate_extraction_results(results)
    if not is_valid:
        logger.warning("PDF extraction validation failed: %s", error_msg)

    _pdf_sessions[session_id] = {
        "extracted_pdfs": results,
        "created_at": time.time(),
        "status": "extracted",
        "synthesis": None,
        "citations": None,
        "pdf_bytes": None,
    }

    extraction_ms = (time.perf_counter() - t_start) * 1000
    logger.info(
        "PDF extraction [session=%s, files=%d, total_pages=%d, ms=%.2f]",
        session_id, len(results), sum(r.total_pages for r in results), extraction_ms
    )

    return PDFExtractionResponse(
        session_id=session_id,
        **results_to_dict(results)
    )


async def synthesize_pdfs(
    session_id: str,
    _auth = Depends(verify_local_auth),
) -> dict[str, Any]:
    """Synthesize findings from previously extracted PDFs."""
    t_start = time.perf_counter()

    if session_id not in _pdf_sessions:
        raise HTTPException(status_code=404, detail="Session not found or expired")

    session = _pdf_sessions[session_id]
    extracted_pdfs = session.get("extracted_pdfs", [])

    if not extracted_pdfs:
        raise HTTPException(status_code=422, detail="No extracted PDFs in session")

    is_valid, error_msg = validate_extraction_results(extracted_pdfs)
    if not is_valid:
        raise HTTPException(status_code=422, detail=error_msg)

    def progress_callback(msg: str):
        session["status"] = "synthesizing"
        session["progress_message"] = msg
        logger.info("Synthesis progress [session=%s]: %s", session_id, msg)

    synthesis_text, citations = await asyncio.to_thread(
        synthesize_documents, extracted_pdfs, progress_callback
    )

    session["synthesis"] = synthesis_text
    session["citations"] = citations
    session["status"] = "synthesized"

    synthesis_ms = (time.perf_counter() - t_start) * 1000
    logger.info(
        "PDF synthesis [session=%s, citations=%d, ms=%.2f]",
        session_id, len(citations), synthesis_ms
    )

    return {
        "session_id": session_id,
        "status": "synthesized",
        "synthesis_length": len(synthesis_text),
        "citations_count": len(citations),
        "citations": citations[:20],
    }


async def generate_pdf(
    session_id: str,
    _auth = Depends(verify_local_auth),
) -> dict[str, Any]:
    """Generate PDF report from synthesis results."""
    t_start = time.perf_counter()

    if session_id not in _pdf_sessions:
        raise HTTPException(status_code=404, detail="Session not found or expired")

    session = _pdf_sessions[session_id]
    synthesis_text = session.get("synthesis")
    citations = session.get("citations", [])
    extracted_pdfs = session.get("extracted_pdfs", [])

    if not synthesis_text:
        raise HTTPException(
            status_code=422,
            detail="No synthesis found. Run /pdf/synthesize first."
        )

    try:
        pdf_bytes = await asyncio.to_thread(
            generate_pdf_report,
            synthesis_text=synthesis_text,
            extracted_pdfs=extracted_pdfs,
            citations=citations,
            title="Multi-Document Synthesis Report",
        )
    except Exception as exc:
        logger.exception("PDF generation failed [session=%s]", session_id)
        raise HTTPException(
            status_code=500,
            detail=f"PDF generation failed: {exc}"
        ) from exc

    session["pdf_bytes"] = pdf_bytes
    session["status"] = "completed"

    generation_ms = (time.perf_counter() - t_start) * 1000
    logger.info(
        "PDF generated [session=%s, size=%d bytes, ms=%.2f]",
        session_id, len(pdf_bytes), generation_ms
    )

    return {
        "session_id": session_id,
        "status": "completed",
        "download_url": f"/pdf/download/{session_id}",
        "filename": "synthesis-report.pdf",
        "file_size": len(pdf_bytes),
    }


async def download_pdf(
    session_id: str,
    _auth = Depends(verify_local_auth),
) -> Response:
    """Download the generated PDF report."""
    if session_id not in _pdf_sessions:
        raise HTTPException(status_code=404, detail="Session not found or expired")

    session = _pdf_sessions[session_id]
    pdf_bytes = session.get("pdf_bytes")

    if not pdf_bytes:
        raise HTTPException(
            status_code=422,
            detail="No PDF generated. Run /pdf/generate first."
        )

    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="synthesis-report.pdf"',
        }
    )


def cleanup_expired_sessions():
    """Remove sessions older than 1 hour."""
    current_time = time.time()
    expired = [
        sid for sid, sess in _pdf_sessions.items()
        if current_time - sess.get("created_at", 0) > 3600
    ]
    for sid in expired:
        del _pdf_sessions[sid]
    if expired:
        logger.info("Cleaned up %d expired PDF sessions", len(expired))


def register_routes(app):
    """Register PDF routes with the FastAPI app."""
    app.post("/pdf/extract", response_model=PDFExtractionResponse, dependencies=[Depends(verify_local_auth)])(extract_pdfs)
    app.post("/pdf/synthesize/{session_id}", dependencies=[Depends(verify_local_auth)])(synthesize_pdfs)
    app.post("/pdf/generate/{session_id}", dependencies=[Depends(verify_local_auth)])(generate_pdf)
    app.get("/pdf/download/{session_id}", dependencies=[Depends(verify_local_auth)])(download_pdf)
    logger.info("PDF routes registered")
