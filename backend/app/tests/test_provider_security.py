"""Comprehensive tests for provider URL security, SSRF prevention, and key forwarding protection.

Verifies:
1. Valid provider URLs (official xAI, OpenAI, Google Gemini, Anthropic, Ollama localhost).
2. Invalid schemes (http for external providers, file://, ftp://, javascript:, gopher://).
3. Localhost and private IP rejections for external providers (127.0.0.1, 10.0.0.1, 192.168.1.1, 172.16.0.1).
4. Cloud metadata address rejections (169.254.169.254, fd00:ec2::254, metadata.google.internal).
5. Redirect blocking for connection tests (e.g., Ollama 301/302 redirects are rejected).
6. DNS resolution failures (unresolvable domain fails safely).
7. API key forwarding restrictions (unauthorized custom domains rejected unless explicitly allowed).
8. Ollama confinement (rejects external domains or private LAN addresses).
"""

import socket
import unittest
from unittest.mock import MagicMock, patch

from starlette.testclient import TestClient

from app import config, llm
from app.main import app
from app.provider_security import (
    PROVIDER_DEFAULT_HOSTS,
    is_prohibited_ip,
    validate_provider_url,
)
import ipaddress


class ProviderURLSecurityTests(unittest.TestCase):
    def setUp(self):
        self._orig_allow_custom = config.ALLOW_CUSTOM_PROVIDER_URLS
        self._orig_allowed_hosts = set(config.ALLOWED_PROVIDER_HOSTS)
        config.ALLOW_CUSTOM_PROVIDER_URLS = False
        config.ALLOWED_PROVIDER_HOSTS = set()

    def tearDown(self):
        config.ALLOW_CUSTOM_PROVIDER_URLS = self._orig_allow_custom
        config.ALLOWED_PROVIDER_HOSTS = self._orig_allowed_hosts

    def test_valid_default_provider_urls(self):
        """Standard official provider endpoints pass validation."""
        valid_cases = [
            ("grok", "https://api.x.ai/v1"),
            ("grok", "https://api.x.ai"),
            ("openai", "https://api.openai.com/v1"),
            ("openai", "https://api.openai.com"),
            ("gemini", "https://generativelanguage.googleapis.com"),
            ("anthropic", "https://api.anthropic.com"),
            ("ollama", "http://localhost:11434"),
            ("ollama", "http://127.0.0.1:11434"),
            ("ollama", "http://[::1]:11434"),
            ("none", ""),
            ("none", "http://any-url-does-not-matter"),
        ]
        for provider, url in valid_cases:
            with self.subTest(provider=provider, url=url):
                valid, err = validate_provider_url(provider, url)
                self.assertTrue(valid, f"Expected {url} to be valid for {provider}, got error: {err}")
                self.assertEqual(err, "")

    def test_external_providers_strictly_reject_http_scheme(self):
        """External cloud providers must never accept plaintext HTTP."""
        insecure_urls = [
            ("grok", "http://api.x.ai/v1"),
            ("openai", "http://api.openai.com/v1"),
            ("gemini", "http://generativelanguage.googleapis.com"),
            ("anthropic", "http://api.anthropic.com"),
        ]
        for provider, url in insecure_urls:
            with self.subTest(provider=provider, url=url):
                valid, err = validate_provider_url(provider, url)
                self.assertFalse(valid)
                self.assertIn("requires https", err.lower())

    def test_rejects_dangerous_and_unsupported_schemes(self):
        """Schemes like file://, ftp://, javascript:, gopher:// must be rejected."""
        dangerous_urls = [
            ("grok", "file:///etc/passwd"),
            ("openai", "ftp://api.openai.com/v1"),
            ("ollama", "file:///C:/Windows/System32/drivers/etc/hosts"),
            ("ollama", "gopher://localhost:11434"),
            ("anthropic", "javascript:alert(1)"),
            ("gemini", "data:text/html,test"),
        ]
        for provider, url in dangerous_urls:
            with self.subTest(provider=provider, url=url):
                valid, err = validate_provider_url(provider, url)
                self.assertFalse(valid)
                self.assertTrue(any(k in err.lower() for k in ("unsupported", "scheme", "malformed")))

    def test_rejects_embedded_credentials(self):
        """Userinfo in provider URLs (user:pass@host) must be blocked to prevent credential sniffing."""
        bad_urls = [
            ("grok", "https://user:pass@api.x.ai/v1"),
            ("openai", "https://admin:secret@api.openai.com/v1"),
            ("ollama", "http://user:pass@localhost:11434"),
        ]
        for provider, url in bad_urls:
            with self.subTest(provider=provider, url=url):
                valid, err = validate_provider_url(provider, url)
                self.assertFalse(valid)
                self.assertIn("credentials", err.lower())

    def test_external_providers_reject_loopback_and_private_ips(self):
        """Cloud providers cannot be pointed to internal loopback or private LAN."""
        internal_targets = [
            "https://127.0.0.1:8000/v1",
            "https://localhost:8000/v1",
            "https://10.0.0.1/v1",
            "https://192.168.1.1:8443/v1",
            "https://172.16.0.5/v1",
        ]
        for url in internal_targets:
            for provider in ("grok", "openai", "gemini", "anthropic"):
                with self.subTest(provider=provider, url=url):
                    valid, err = validate_provider_url(provider, url)
                    self.assertFalse(valid)
                    self.assertTrue(any(w in err.lower() for w in ("untrusted", "loopback", "private", "prohibited")))

    def test_rejects_cloud_metadata_destinations(self):
        """All providers must reject AWS/GCP/Azure instance metadata endpoints."""
        metadata_urls = [
            ("grok", "https://169.254.169.254/v1"),
            ("openai", "https://169.254.169.254/latest/meta-data"),
            ("ollama", "http://169.254.169.254/latest/meta-data"),
            ("ollama", "http://metadata.google.internal/computeMetadata/v1"),
            ("ollama", "http://instance-data/latest/meta-data"),
            ("anthropic", "https://[fd00:ec2::254]/v1"),
        ]
        for provider, url in metadata_urls:
            with self.subTest(provider=provider, url=url):
                valid, err = validate_provider_url(provider, url)
                self.assertFalse(valid)
                self.assertTrue(any(w in err.lower() for w in ("metadata", "untrusted", "prohibited", "non-loopback")))

    def test_ollama_rejects_external_hosts_and_lan_traversal(self):
        """Ollama is strictly confined to local loopback; public domains or private LAN are blocked."""
        bad_ollama_urls = [
            "http://google.com:11434",
            "http://example.com:11434",
            "http://192.168.1.50:11434",
            "http://10.1.2.3:11434",
        ]
        for url in bad_ollama_urls:
            with self.subTest(url=url):
                valid, err = validate_provider_url("ollama", url)
                self.assertFalse(valid)
                self.assertTrue(any(w in err.lower() for w in ("non-loopback", "localhost", "resolves")))

    def test_dns_resolution_failure_is_handled_safely(self):
        """If provider hostname cannot be resolved, validation fails gracefully."""
        with patch("socket.getaddrinfo", side_effect=socket.gaierror("Name or service not known")):
            valid, err = validate_provider_url("ollama", "http://nonexistent-local-host-999.xyz:11434")
            self.assertFalse(valid)
            self.assertIn("dns resolution failed", err.lower())

    def test_api_key_forwarding_restriction_blocks_arbitrary_domains(self):
        """Untrusted external domains are blocked by default to prevent API key leakage."""
        attacker_urls = [
            ("grok", "https://evil-attacker.com/v1"),
            ("openai", "https://steal-keys.example.com/api"),
            ("anthropic", "https://pwned.net/v1"),
        ]
        for provider, url in attacker_urls:
            with self.subTest(provider=provider, url=url):
                valid, err = validate_provider_url(provider, url)
                self.assertFalse(valid)
                self.assertIn("untrusted provider host", err.lower())

    def test_custom_provider_urls_safe_configuration(self):
        """When ALLOW_CUSTOM_PROVIDER_URLS is enabled, verified external domains pass, but SSRF checks remain active."""
        config.ALLOW_CUSTOM_PROVIDER_URLS = True

        with patch("socket.getaddrinfo") as mock_gai:
            # Resolves to a public IP
            mock_gai.return_value = [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))]
            valid, err = validate_provider_url("grok", "https://my-internal-proxy.company.com/v1")
            self.assertTrue(valid, f"Expected custom URL to be permitted when enabled, got {err}")

            # Even with custom URLs enabled, SSRF to private IP is blocked!
            mock_gai.return_value = [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("192.168.1.10", 443))]
            valid, err = validate_provider_url("grok", "https://bad-proxy.company.com/v1")
            self.assertFalse(valid)
            self.assertIn("prohibited destination", err.lower())


class ProviderConnectionSecurityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_provider_test_blocks_ssrf_and_key_forwarding_attempts(self):
        """POST /provider/test rejects SSRF targets before sending outbound requests."""
        malicious_payloads = [
            {"provider": "grok", "api_key": "xai-test-key", "base_url": "http://169.254.169.254/latest"},
            {"provider": "grok", "api_key": "xai-test-key", "base_url": "http://127.0.0.1:8000/secret"},
            {"provider": "grok", "api_key": "xai-test-key", "base_url": "https://evil-harvest-server.com/v1"},
            {"provider": "ollama", "base_url": "http://169.254.169.254/latest/meta-data"},
            {"provider": "ollama", "base_url": "http://192.168.1.1:11434"},
        ]
        for payload in malicious_payloads:
            with self.subTest(payload=payload):
                res = self.client.post("/provider/test", json=payload)
                self.assertEqual(res.status_code, 400)
                data = res.json()
                self.assertTrue(
                    any(term in data["detail"].lower() for term in ("security rejection", "failed", "untrusted", "loopback", "metadata")),
                    f"Unexpected detail: {data['detail']}"
                )

    def test_ollama_connection_test_blocks_redirects(self):
        """If Ollama endpoint responds with a 301/302 redirect, test fails closed."""
        with patch("httpx.get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.is_redirect = True
            mock_resp.status_code = 302
            mock_get.return_value = mock_resp

            success, msg = llm.test_provider_connection("ollama", base_url="http://localhost:11434")
            self.assertFalse(success)
            self.assertIn("redirect", msg.lower())


if __name__ == "__main__":
    unittest.main()

