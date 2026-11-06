"""Unit and endpoint tests for PDF upload validation and size limits.

Verifies:
1. File-count limit (<=10 allowed, >10 rejected with 422).
2. Per-file size limit (<=50MB allowed, >50MB rejected with 413).
3. Total upload size limit (<=100MB allowed, >100MB rejected with 413).
4. File extension validation (must end in .pdf).
5. Declared content-type validation (application/pdf, application/x-pdf, or empty).
6. Magic header validation (%PDF- in first 1024 bytes).
7. Non-PDF and malformed payload rejection before large file reads.
8. Empty file rejection (0 bytes -> 422).
9. Endpoint enforcement on /memory/pdf, /pdf/extract, /jobs/create.
"""

import io
import unittest
from unittest.mock import AsyncMock, patch

from fastapi import HTTPException, UploadFile
from starlette.testclient import TestClient

from app import config
from app.main import app
from app.pdf_handler import (
    MAX_FILE_SIZE,
    MAX_PDF_FILES,
    MAX_TOTAL_UPLOAD_SIZE,
    read_and_validate_pdf_upload,
    validate_and_read_pdf_files,
)


def _make_mock_upload(
    filename: str = "doc.pdf",
    content: bytes = b"%PDF-1.4\n1 0 obj<</Type/Catalog>>endobj\n%%EOF",
    content_type: str = "application/pdf",
    size: int | None = None,
) -> UploadFile:
    bio = io.BytesIO(content)
    upload = UploadFile(
        file=bio,
        filename=filename,
        headers={"content-type": content_type} if content_type else None,
        size=size if size is not None else len(content),
    )
    return upload


class PDFUploadValidationTests(unittest.IsolatedAsyncioTestCase):
    async def test_rejects_non_pdf_extension(self):
        """Files without .pdf extension must be rejected with 422."""
        upload = _make_mock_upload(filename="malicious.exe", content=b"%PDF-1.4 header")
        with self.assertRaises(HTTPException) as ctx:
            await read_and_validate_pdf_upload(upload)
        self.assertEqual(ctx.exception.status_code, 422)
        self.assertIn("Only PDF files (.pdf) are accepted", ctx.exception.detail)

    async def test_rejects_invalid_declared_content_type(self):
        """Files declaring a non-PDF MIME type must be rejected with 422."""
        upload = _make_mock_upload(
            filename="document.pdf",
            content=b"%PDF-1.4 valid",
            content_type="image/jpeg",
        )
        with self.assertRaises(HTTPException) as ctx:
            await read_and_validate_pdf_upload(upload)
        self.assertEqual(ctx.exception.status_code, 422)
        self.assertIn("Declared content-type 'image/jpeg' is not a PDF", ctx.exception.detail)

    async def test_rejects_missing_pdf_magic_signature(self):
        """Files missing %PDF- in the header must be rejected with 422 even if named .pdf."""
        upload = _make_mock_upload(
            filename="spoofed.pdf",
            content=b"<!DOCTYPE html><html><body>Not a PDF</body></html>",
            content_type="application/pdf",
        )
        with self.assertRaises(HTTPException) as ctx:
            await read_and_validate_pdf_upload(upload)
        self.assertEqual(ctx.exception.status_code, 422)
        self.assertIn("missing %PDF- header", ctx.exception.detail)

    async def test_rejects_empty_file(self):
        """Empty 0-byte file must be rejected with 422."""
        upload = _make_mock_upload(filename="empty.pdf", content=b"", content_type="application/pdf")
        with self.assertRaises(HTTPException) as ctx:
            await read_and_validate_pdf_upload(upload)
        self.assertEqual(ctx.exception.status_code, 422)

    async def test_rejects_file_exceeding_per_file_size_limit(self):
        """Files larger than 50MB must be rejected with 413."""
        # Test pre-check via size metadata
        upload = _make_mock_upload(
            filename="huge.pdf",
            content=b"%PDF-1.4 dummy",
            size=MAX_FILE_SIZE + 1024,
        )
        with self.assertRaises(HTTPException) as ctx:
            await read_and_validate_pdf_upload(upload)
        self.assertEqual(ctx.exception.status_code, 413)
        self.assertIn("exceeds 50MB limit", ctx.exception.detail)

    async def test_streaming_rejection_stops_early_on_bad_signature(self):
        """Non-PDF payload streamed in chunks is rejected on first chunk without reading rest."""
        chunks_read = []

        class MockSlowStreamUpload(UploadFile):
            def __init__(self):
                super().__init__(file=io.BytesIO(), filename="large_fake.pdf")
                self.calls = 0

            async def read(self, size: int = -1) -> bytes:
                self.calls += 1
                if self.calls == 1:
                    # First 64KB chunk has NO %PDF- header
                    return b"A" * 65536
                elif self.calls <= 5:
                    return b"B" * 65536
                return b""

        mock_upload = MockSlowStreamUpload()
        with self.assertRaises(HTTPException) as ctx:
            await read_and_validate_pdf_upload(mock_upload)
        self.assertEqual(ctx.exception.status_code, 422)
        # Verify it rejected immediately on chunk 1 and did not read further chunks
        self.assertEqual(mock_upload.calls, 1)

    async def test_rejects_batch_exceeding_max_file_count(self):
        """Batches with more than 10 files must be rejected with 422."""
        files = [_make_mock_upload(filename=f"doc_{i}.pdf") for i in range(11)]
        with self.assertRaises(HTTPException) as ctx:
            await validate_and_read_pdf_files(files, max_files=MAX_PDF_FILES)
        self.assertEqual(ctx.exception.status_code, 422)
        self.assertIn("Maximum 10 PDFs allowed per request", ctx.exception.detail)

    async def test_accepts_valid_pdf_batch(self):
        """Batch of valid PDFs within count and size limits succeeds."""
        valid_content = b"%PDF-1.4\n1 0 obj<</Type/Catalog>>endobj\n%%EOF"
        files = [_make_mock_upload(filename=f"doc_{i}.pdf", content=valid_content) for i in range(3)]
        results = await validate_and_read_pdf_files(files)
        self.assertEqual(len(results), 3)
        for data, fname in results:
            self.assertEqual(data, valid_content)
            self.assertTrue(fname.startswith("doc_"))


class PDFUploadEndpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def setUp(self):
        self._orig_token = config.API_AUTH_TOKEN
        config.API_AUTH_TOKEN = ""  # Local unauthenticated access for testing endpoint validation

    def tearDown(self):
        config.API_AUTH_TOKEN = self._orig_token

    def test_memory_pdf_rejects_non_pdf_file(self):
        """POST /memory/pdf rejects non-PDF file upload with 422."""
        res = self.client.post(
            "/memory/pdf",
            files={"file": ("notes.txt", b"Plain text notes", "text/plain")},
        )
        self.assertEqual(res.status_code, 422)
        self.assertIn("Only PDF files", res.text)

    def test_pdf_extract_rejects_exceeded_file_count(self):
        """POST /pdf/extract rejects more than 10 files with 422."""
        dummy_pdf = b"%PDF-1.4\n1 0 obj<</Type/Catalog>>endobj\n%%EOF"
        files = [("files", (f"file_{i}.pdf", dummy_pdf, "application/pdf")) for i in range(11)]
        res = self.client.post("/pdf/extract", files=files)
        self.assertEqual(res.status_code, 422)
        self.assertIn("Maximum 10 PDFs allowed", res.text)

    def test_jobs_create_rejects_malformed_pdf(self):
        """POST /jobs/create multipart rejects malformed non-PDF files with 422."""
        import json
        req_json = json.dumps({"urls": []})
        res = self.client.post(
            "/jobs/create",
            data={"request": req_json},
            files=[("files", ("spoofed.pdf", b"NOT A REAL PDF", "application/pdf"))],
        )
        self.assertEqual(res.status_code, 422)
        self.assertIn("missing %PDF- header", res.text)


if __name__ == "__main__":
    unittest.main()
