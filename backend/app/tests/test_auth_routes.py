"""Comprehensive unit and FastAPI route authentication verification tests.

Verifies:
1. Direct unit tests of verify_local_auth:
   - Missing, invalid, and valid credentials
   - Rejection of query-parameter tokens in URLs
   - Handling of Bearer authentication
   - Fallback when API_AUTH_TOKEN is unset
2. FastAPI TestClient requests across EVERY protected route:
   - 401 on missing credentials
   - 401 on invalid X-API-Key
   - 401 on invalid Bearer token
   - 401 on query parameter tokens (?token=... or ?api_key=...)
   - Non-401 (auth passed) on valid X-API-Key
   - Non-401 (auth passed) on valid Bearer token
3. Public /health route remains accessible without credentials.
"""

import io
import unittest
from fastapi import HTTPException
from starlette.testclient import TestClient

from app import config
from app.main import app, verify_local_auth


class AuthVerificationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def setUp(self):
        self._orig_token = config.API_AUTH_TOKEN

    def tearDown(self):
        config.API_AUTH_TOKEN = self._orig_token

    def test_direct_verify_local_auth_unit(self):
        """Direct unit testing of verify_local_auth function."""
        # Unset / empty token -> always passes
        config.API_AUTH_TOKEN = ""
        verify_local_auth(x_api_key=None, authorization=None)

        config.API_AUTH_TOKEN = "super-secret-auth-key-12345"

        # Missing credentials -> 401
        with self.assertRaises(HTTPException) as ctx:
            verify_local_auth(x_api_key=None, authorization=None)
        self.assertEqual(ctx.exception.status_code, 401)
        self.assertIn("Unauthorized", ctx.exception.detail)

        # Invalid X-API-Key -> 401
        with self.assertRaises(HTTPException) as ctx:
            verify_local_auth(x_api_key="wrong-key", authorization=None)
        self.assertEqual(ctx.exception.status_code, 401)

        # Invalid Bearer token -> 401
        with self.assertRaises(HTTPException) as ctx:
            verify_local_auth(x_api_key=None, authorization="Bearer wrong-key")
        self.assertEqual(ctx.exception.status_code, 401)

        # Unsupported auth scheme -> 401
        with self.assertRaises(HTTPException) as ctx:
            verify_local_auth(x_api_key=None, authorization="Basic dXNlcjpwYXNz")
        self.assertEqual(ctx.exception.status_code, 401)

        # Valid X-API-Key -> passes
        verify_local_auth(x_api_key="super-secret-auth-key-12345", authorization=None)

        # Valid Bearer token -> passes
        verify_local_auth(x_api_key=None, authorization="Bearer super-secret-auth-key-12345")
        verify_local_auth(x_api_key=None, authorization="bearer super-secret-auth-key-12345")

    def test_health_route_is_unprotected(self):
        """The /health endpoint must remain accessible without any credentials."""
        config.API_AUTH_TOKEN = "strict-token-abc"
        res = self.client.get("/health")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data.get("status"), "ok")

    def test_all_protected_routes_with_fastapi_requests(self):
        """Test missing, invalid, and valid credentials against all protected routes."""
        test_token = "valid-test-token-77777"
        config.API_AUTH_TOKEN = test_token

        minimal_pdf = b"%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj 2 0 obj<</Type/Pages/Kids[]/Count 0>>endobj\nxref\n0 3\n0000000000 65535 f\ntrailer<</Size 3/Root 1 0 R>>\nstartxref\n100\n%%EOF"

        protected_endpoints = [
            ("POST", "/memory", {"json": {"text": "A memorable sentence", "source_url": "https://example.com"}}),
            ("POST", "/webpage", {"json": {"url": "https://example.com/page", "title": "Test Page", "html": "<html><body>Hello</body></html>"}}),
            ("POST", "/memory/pdf", {"files": {"file": ("test.pdf", minimal_pdf, "application/pdf")}}),
            ("GET", "/memory/doc123/tags", {}),
            ("GET", "/tags?url=https%3A%2F%2Fexample.com", {}),
            ("POST", "/ask", {"json": {"question": "What is Python?"}}),
            ("POST", "/ask/stream", {"json": {"question": "What is Python?"}}),
            ("POST", "/pdf/extract", {"files": [("files", ("test.pdf", minimal_pdf, "application/pdf"))]}),
            ("POST", "/pdf/synthesize/dummy_session", {}),
            ("POST", "/pdf/generate/dummy_session", {}),
            ("GET", "/pdf/download/dummy_session", {}),
            ("POST", "/jobs/create", {"json": {"urls": ["https://example.com/article"]}}),
            ("GET", "/jobs/dummy_job/status", {}),
            ("GET", "/jobs/dummy_job/pdf", {}),
            ("GET", "/provider/status", {}),
            ("POST", "/provider/test", {"json": {"provider": "none"}}),
            ("POST", "/provider/configure", {"json": {"provider": "none", "test_first": False}}),
            ("POST", "/provider/remove", {"json": {"provider": "none"}}),
        ]

        for method, path, kwargs in protected_endpoints:
            # 1. Missing credentials -> MUST return 401
            res_missing = self.client.request(method, path, **kwargs)
            self.assertEqual(
                res_missing.status_code, 401,
                f"Route {method} {path} should return 401 when unauthenticated, got {res_missing.status_code}"
            )

            # 2. Invalid X-API-Key -> MUST return 401
            res_bad_key = self.client.request(method, path, headers={"X-API-Key": "invalid-secret"}, **kwargs)
            self.assertEqual(
                res_bad_key.status_code, 401,
                f"Route {method} {path} should return 401 with invalid X-API-Key, got {res_bad_key.status_code}"
            )

            # 3. Invalid Bearer token -> MUST return 401
            res_bad_bearer = self.client.request(method, path, headers={"Authorization": "Bearer invalid-secret"}, **kwargs)
            self.assertEqual(
                res_bad_bearer.status_code, 401,
                f"Route {method} {path} should return 401 with invalid Bearer, got {res_bad_bearer.status_code}"
            )

            # 4. Token in URL query param -> MUST return 401 (URL tokens disallowed)
            delim = "&" if "?" in path else "?"
            query_path = f"{path}{delim}token={test_token}&api_key={test_token}"
            res_query = self.client.request(method, query_path, **kwargs)
            self.assertEqual(
                res_query.status_code, 401,
                f"Route {method} {path} should reject query parameter tokens with 401, got {res_query.status_code}"
            )

            # 5. Valid X-API-Key -> MUST NOT return 401 (auth passed)
            res_valid_key = self.client.request(method, path, headers={"X-API-Key": test_token}, **kwargs)
            self.assertNotEqual(
                res_valid_key.status_code, 401,
                f"Route {method} {path} returned 401 despite valid X-API-Key"
            )

            # 6. Valid Bearer token -> MUST NOT return 401 (auth passed)
            res_valid_bearer = self.client.request(method, path, headers={"Authorization": f"Bearer {test_token}"}, **kwargs)
            self.assertNotEqual(
                res_valid_bearer.status_code, 401,
                f"Route {method} {path} returned 401 despite valid Bearer token"
            )


if __name__ == "__main__":
    unittest.main()
