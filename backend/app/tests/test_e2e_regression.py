"""End-to-end regression tests for MemoryMind MKVII core workflows.

Tests:
1. Remember selected text (POST /memory -> GET /memory/{id}/tags -> GET /tags)
2. Remember current page (POST /webpage)
3. Ask question (POST /ask)
4. Ask streaming (POST /ask/stream)
5. PDF memory upload (POST /memory/pdf)
6. Multi-source research job lifecycle:
   - Creation (POST /jobs/create with PDF and configuration)
   - Status polling (GET /jobs/{job_id}/status)
   - PDF report retrieval (GET /jobs/{job_id}/pdf)
"""

import io
import json
import time
import unittest
from starlette.testclient import TestClient

from app import config, memory_store
from app.main import app
from app.summarization_jobs import get_job_manager, JobConfig, SummaryStyle
from app.source_collector import SourceCollection


def _create_test_pdf(text: str = "MemoryMind integration test content") -> bytes:
    try:
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import letter
        buf = io.BytesIO()
        c = canvas.Canvas(buf, pagesize=letter)
        c.drawString(100, 750, text)
        c.showPage()
        c.save()
        return buf.getvalue()
    except ImportError:
        return (
            b"%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
            b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
            b"3 0 obj<</Type/Page/MediaBox[0 0 612 792]/Parent 2 0 R>>endobj\n"
            b"xref\n0 4\n0000000000 65535 f\n0000000009 00000 n\n"
            b"0000000058 00000 n\n0000000115 00000 n\n"
            b"trailer<</Size 4/Root 1 0 R>>\nstartxref\n180\n%%EOF"
        )


class E2ERegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def setUp(self):
        self._orig_token = config.API_AUTH_TOKEN
        config.API_AUTH_TOKEN = ""

    def tearDown(self):
        config.API_AUTH_TOKEN = self._orig_token

    def test_remember_selected_text_and_tag_retrieval(self):
        """End-to-end test: remember selected text -> check tags by doc id -> check tags by url."""
        url = "https://example.org/e2e-remember-text"
        payload = {
            "text": "Machine learning enables computers to learn from data without explicit rules.",
            "source_url": url,
            "source_title": "Machine Learning Overview",
            "visited_at": "2026-10-27T12:00:00Z",
        }
        res = self.client.post("/memory", json=payload)
        self.assertEqual(res.status_code, 200, f"Remember failed: {res.text}")
        data = res.json()
        self.assertTrue(data.get("success"))
        doc_id = data.get("document_id") or data.get("id")
        self.assertTrue(doc_id)

        # Retrieve tags by document ID
        tag_res = self.client.get(f"/memory/{doc_id}/tags")
        self.assertEqual(tag_res.status_code, 200)
        tag_data = tag_res.json()
        self.assertTrue(tag_data.get("saved"))
        self.assertEqual(tag_data.get("document_id"), doc_id)

        # Retrieve tags by URL
        url_tag_res = self.client.get(f"/tags?url={url}")
        self.assertEqual(url_tag_res.status_code, 200)

    def test_remember_current_webpage(self):
        """End-to-end test: save HTML content via /webpage endpoint."""
        html = """
        <!DOCTYPE html>
        <html>
        <head><title>Vector Embeddings Guide</title></head>
        <body>
            <main>
                <h1>Dense Vector Representations</h1>
                <p>Embeddings project high dimensional text tokens into a shared continuous semantic vector space.</p>
            </main>
        </body>
        </html>
        """
        payload = {
            "url": "https://example.org/embeddings-guide",
            "title": "Vector Embeddings Guide",
            "html": html,
        }
        res = self.client.post("/webpage", json=payload)
        self.assertEqual(res.status_code, 200, f"Webpage remember failed: {res.text}")
        data = res.json()
        self.assertTrue(data.get("success"))
        self.assertGreater(data.get("chunks", 0), 0)

    def test_ask_question_flow(self):
        """End-to-end test: query the memory store via /ask."""
        payload = {"question": "What is Python?"}
        res = self.client.post("/ask", json=payload)
        self.assertEqual(res.status_code, 200, f"Ask failed: {res.text}")
        data = res.json()
        self.assertIn("answer", data)
        self.assertIsInstance(data.get("sources"), list)

    def test_ask_stream_flow(self):
        """End-to-end test: stream answer tokens via SSE on /ask/stream."""
        payload = {"question": "What is Python?"}
        with self.client.stream("POST", "/ask/stream", json=payload) as response:
            self.assertEqual(response.status_code, 200)
            content = "".join(chunk for chunk in response.iter_text())
            self.assertTrue(len(content) > 0)
            self.assertIn('"type": "token"', content)
            self.assertIn('"type": "done"', content)

    def test_pdf_upload_to_memory_flow(self):
        """End-to-end test: upload a PDF file into the memory store via /memory/pdf."""
        pdf_bytes = _create_test_pdf("Neural networks process tensors across multiple hidden layers.")
        files = {"file": ("deep_learning.pdf", pdf_bytes, "application/pdf")}
        data = {
            "source_url": "https://example.org/dl-paper.pdf",
            "source_title": "Deep Learning Paper",
        }
        res = self.client.post("/memory/pdf", files=files, data=data)
        self.assertEqual(res.status_code, 200, f"PDF upload memory failed: {res.text}")
        body = res.json()
        self.assertTrue(body.get("success"))
        self.assertGreaterEqual(body.get("page_count", 0), 1)

    def test_research_job_lifecycle_create_poll_and_view_pdf(self):
        """End-to-end test: create research job -> poll status -> download/view PDF."""
        pdf_bytes = _create_test_pdf("Multi-source research on artificial intelligence and agent architectures.")
        files = {"files": ("research_paper.pdf", pdf_bytes, "application/pdf")}
        req_config = json.dumps({
            "urls": [],
            "summary_style": "detailed",
            "topic": "AI Research",
            "max_length": 1000,
        })
        form_data = {"request": req_config}

        # 1. Create Job
        create_res = self.client.post("/jobs/create", data=form_data, files=files)
        self.assertEqual(create_res.status_code, 200, f"Job create failed: {create_res.text}")
        job_data = create_res.json()
        job_id = job_data["job_id"]
        self.assertTrue(job_id)

        # 2. Poll Status
        status_res = self.client.get(f"/jobs/{job_id}/status")
        self.assertEqual(status_res.status_code, 200)
        status_data = status_res.json()
        self.assertEqual(status_data["job_id"], job_id)
        self.assertIn("status", status_data)
        self.assertIn("progress", status_data)

        # 3. Simulate job completion by attaching generated PDF bytes to job object
        manager = get_job_manager()
        job = manager.get_job(job_id)
        self.assertIsNotNone(job)
        job.pdf_bytes = pdf_bytes
        # In-memory dictionary in JobManager is already updated by reference

        # Verify status now reports has_pdf True
        updated_status = self.client.get(f"/jobs/{job_id}/status").json()
        self.assertTrue(updated_status["has_pdf"])
        self.assertGreater(updated_status["pdf_size"], 0)

        # 4. View / Download PDF Report
        pdf_res = self.client.get(f"/jobs/{job_id}/pdf")
        self.assertEqual(pdf_res.status_code, 200)
        self.assertEqual(pdf_res.headers["content-type"], "application/pdf")
        self.assertIn("inline", pdf_res.headers.get("content-disposition", ""))
        self.assertEqual(pdf_res.content, pdf_bytes)


if __name__ == "__main__":
    unittest.main()
