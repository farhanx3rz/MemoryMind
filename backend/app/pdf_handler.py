"""PDF extraction with page tracking and error handling.

Extracts text from PDFs while preserving document identity and page numbers.
Handles encryption, scanned PDFs, and empty documents gracefully.
"""
import io
import logging
from dataclasses import dataclass, field
from typing import Any

from fastapi import HTTPException, UploadFile

logger = logging.getLogger("memorymind.pdf_handler")

MAX_PDF_FILES = 10
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB
MAX_TOTAL_UPLOAD_SIZE = 100 * 1024 * 1024  # 100 MB


async def read_and_validate_pdf_upload(
    upload: UploadFile,
    current_total_bytes: int = 0,
) -> bytes:
    """Read an uploaded file in chunks while enforcing size limits and validating PDF format.

    Validates:
    - File extension (.pdf)
    - Declared content-type (must be application/pdf or empty)
    - Per-file and total upload size limits (enforced during chunked read)
    - PDF magic header (%PDF-) within the first 1024 bytes
    - Rejects empty files and non-PDF payloads before loading into memory
    """
    filename = upload.filename or "uploaded.pdf"
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=422,
            detail=f"Invalid file: '{filename}'. Only PDF files (.pdf) are accepted."
        )

    # Validate declared content-type if provided. Empty content-type is
    # tolerated because some browsers/extensions omit it for blob uploads.
    declared_type = (upload.content_type or "").strip().lower()
    if declared_type and declared_type not in ("application/pdf", "application/x-pdf"):
        raise HTTPException(
            status_code=422,
            detail=f"Invalid file: '{filename}'. Declared content-type '{declared_type}' is not a PDF."
        )

    # Pre-check file size if metadata available
    if upload.size is not None:
        if upload.size > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"File '{filename}' exceeds {MAX_FILE_SIZE // (1024*1024)}MB limit."
            )
        if current_total_bytes + upload.size > MAX_TOTAL_UPLOAD_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"Total upload size exceeds {MAX_TOTAL_UPLOAD_SIZE // (1024*1024)}MB limit."
            )

    chunk_size = 64 * 1024
    chunks = []
    file_bytes_read = 0
    header_checked = False

    while True:
        chunk = await upload.read(chunk_size)
        if not chunk:
            break
        file_bytes_read += len(chunk)
        if file_bytes_read > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"File '{filename}' exceeds {MAX_FILE_SIZE // (1024*1024)}MB limit."
            )
        if current_total_bytes + file_bytes_read > MAX_TOTAL_UPLOAD_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"Total upload size exceeds {MAX_TOTAL_UPLOAD_SIZE // (1024*1024)}MB limit."
            )
        chunks.append(chunk)

        # Perform streaming PDF signature check on the first chunk(s) so we
        # never buffer a large non-PDF payload before rejecting it.
        if not header_checked:
            head = b"".join(chunks)[:1024]
            if b"%PDF-" not in head:
                raise HTTPException(
                    status_code=422,
                    detail=f"Invalid file: '{filename}' is not a valid PDF document (missing %PDF- header)."
                )
            header_checked = True

    data = b"".join(chunks)
    if len(data) == 0:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid file: '{filename}' is empty."
        )

    # Final PDF magic header validation (belt and suspenders)
    if b"%PDF-" not in data[:1024]:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid file: '{filename}' is not a valid PDF document."
        )

    return data


async def validate_and_read_pdf_files(
    files: list[UploadFile],
    max_files: int = MAX_PDF_FILES,
) -> list[tuple[bytes, str]]:
    """Validate and read a batch of uploaded PDF files under size and type constraints."""
    if not files:
        return []

    if len(files) > max_files:
        raise HTTPException(
            status_code=422,
            detail=f"Maximum {max_files} PDFs allowed per request"
        )

    total_bytes = 0
    file_data: list[tuple[bytes, str]] = []
    for upload in files:
        filename = upload.filename or "uploaded.pdf"
        data = await read_and_validate_pdf_upload(upload, current_total_bytes=total_bytes)
        total_bytes += len(data)
        file_data.append((data, filename))

    return file_data


@dataclass
class PDFPage:
    """A single page from a PDF."""
    page_num: int
    text: str
    is_empty: bool = False


@dataclass
class ExtractedPDF:
    """Result of extracting a single PDF file."""
    filename: str
    total_pages: int
    extracted_pages: list[PDFPage] = field(default_factory=list)
    error: str = ""
    is_encrypted: bool = False
    is_scanned: bool = False
    extraction_status: str = "pending"

    @property
    def page_count(self) -> int:
        """Number of pages with extractable text."""
        return sum(1 for p in self.extracted_pages if not p.is_empty)

    @property
    def success(self) -> bool:
        """True if extraction succeeded."""
        return not self.error and self.page_count > 0

    def get_full_text(self) -> str:
        """Return concatenated text from all pages with page markers."""
        if not self.extracted_pages:
            return ""
        lines = []
        for page in self.extracted_pages:
            if not page.is_empty:
                lines.append(f"[Page {page.page_num}]\n{page.text}\n")
        return "\n".join(lines)


def extract_pdf(file_data: bytes, filename: str) -> ExtractedPDF:
    """Extract text from a PDF file.

    Args:
        file_data: Raw PDF bytes
        filename: Original filename for tracking

    Returns:
        ExtractedPDF with extracted text, page numbers, and error info
    """
    result = ExtractedPDF(filename=filename, total_pages=0, extraction_status="extracting")

    try:
        from pypdf import PdfReader
    except ImportError as exc:
        result.error = "PDF extraction library not installed (pypdf required)"
        result.extraction_status = "failed"
        logger.error("PDF extraction failed: %s", exc)
        return result

    try:
        pdf_file = io.BytesIO(file_data)
        reader = PdfReader(pdf_file)
        result.total_pages = len(reader.pages)

        if result.total_pages == 0:
            result.error = "PDF contains no pages"
            result.extraction_status = "failed"
            return result

    except Exception as exc:
        error_msg = str(exc).lower()
        if "encrypted" in error_msg or "password" in error_msg:
            result.is_encrypted = True
            result.error = "Cannot extract - file is password-protected"
            result.extraction_status = "failed"
        else:
            result.error = f"Failed to read PDF: {exc}"
            result.extraction_status = "failed"
        logger.warning("PDF read error [%s]: %s", filename, exc)
        return result

    extracted_text_count = 0
    for page_num, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
            text = text.strip()

            if not text:
                result.extracted_pages.append(PDFPage(page_num=page_num, text="", is_empty=True))
            else:
                extracted_text_count += 1
                result.extracted_pages.append(PDFPage(page_num=page_num, text=text, is_empty=False))

        except Exception as exc:
            logger.warning("Failed to extract page %d from %s: %s", page_num, filename, exc)
            result.extracted_pages.append(PDFPage(page_num=page_num, text="", is_empty=True))

    if extracted_text_count == 0 and result.total_pages > 0:
        result.is_scanned = True
        ocr_succeeded = False
        try:
            import pytesseract
            from pdf2image import convert_from_bytes

            images = convert_from_bytes(file_data, first_page=1, last_page=min(result.total_pages, 10))
            ocr_pages = []
            for p_num, img in enumerate(images, start=1):
                page_text = pytesseract.image_to_string(img).strip()
                if page_text:
                    ocr_pages.append(PDFPage(page_num=p_num, text=page_text, is_empty=False))
                else:
                    ocr_pages.append(PDFPage(page_num=p_num, text="", is_empty=True))

            if any(not p.is_empty for p in ocr_pages):
                result.extracted_pages = ocr_pages
                result.error = ""
                result.extraction_status = "completed"
                ocr_succeeded = True
                logger.info("OCR successfully extracted text from scanned PDF [%s]", filename)
        except Exception as ocr_exc:
            logger.debug("Optional OCR unavailable or failed for %s: %s", filename, ocr_exc)

        if not ocr_succeeded:
            result.error = "Text extraction failed - appears to be image-only (OCR unavailable)"
            result.extraction_status = "failed"
            logger.info("PDF %s appears to be scanned/image-only", filename)
            return result

    result.extraction_status = "completed"
    logger.info(
        "Extracted PDF [%s]: %d pages total, %d with text",
        filename, result.total_pages, extracted_text_count
    )
    return result


def extract_multiple_pdfs(files: list[tuple[bytes, str]]) -> list[ExtractedPDF]:
    """Extract text from multiple PDF files.

    Args:
        files: List of (file_data, filename) tuples

    Returns:
        List of ExtractedPDF results
    """
    results = []
    for file_data, filename in files:
        result = extract_pdf(file_data, filename)
        results.append(result)
    return results


def validate_extraction_results(results: list[ExtractedPDF]) -> tuple[bool, str]:
    """Validate extraction results and return (is_valid, error_message).

    At least one PDF must have extractable text.
    """
    if not results:
        return False, "No PDFs provided"

    successful = [r for r in results if r.success]
    if not successful:
        failed_reasons = []
        for r in results:
            if r.error:
                failed_reasons.append(f"{r.filename}: {r.error}")
        return False, "No text could be extracted. " + " | ".join(failed_reasons)

    return True, ""


def results_to_dict(results: list[ExtractedPDF]) -> dict[str, Any]:
    """Convert ExtractedPDF results to JSON-serializable dict."""
    return {
        "pdfs": [
            {
                "filename": r.filename,
                "total_pages": r.total_pages,
                "extracted_pages": r.page_count,
                "success": r.success,
                "error": r.error,
                "is_encrypted": r.is_encrypted,
                "is_scanned": r.is_scanned,
                "status": r.extraction_status,
            }
            for r in results
        ],
        "total_pages": sum(r.total_pages for r in results),
        "total_extracted": sum(r.page_count for r in results),
        "all_successful": all(r.success for r in results),
    }
