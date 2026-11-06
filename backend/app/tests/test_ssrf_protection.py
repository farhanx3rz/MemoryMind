"""Unit and functional tests for SSRF protection and per-hop redirect validation.

Verifies:
1. is_safe_url blocks:
   - IPv4 loopback (127.0.0.0/8)
   - IPv4 private networks (10.0.0.0/8, 172.16.0.0/12, 192.168.0.0/16)
   - IPv4 link-local and cloud metadata (169.254.169.254, 169.254.0.0/16)
   - IPv4 multicast and broadcast (224.0.0.0/4, 255.255.255.255)
   - IPv6 loopback (::1)
   - IPv6 unique local (fc00::/7) and link-local (fe80::/10)
   - IPv6 cloud metadata (fd00:ec2::254)
   - IPv6-mapped IPv4 addresses (::ffff:127.0.0.1, ::ffff:169.254.169.254)
   - Dangerous schemes (file://, ftp://, gopher://, javascript:)
   - Embedded credentials (user:pass@host)
   - Internal hostname suffixes (.local, .lan, .internal, .corp, .home, .arpa, .biz)
   - Prohibited metadata hostnames (localhost, metadata.google.internal, instance-data)
   - DNS resolution that returns prohibited IPs
2. fetch_webpage redirect validation:
   - Blocks redirects pointing to private or cloud metadata IPs
   - Enforces maximum redirect hop limit (5 hops)
   - Detects DNS rebinding / socket connection to prohibited peer IP
3. Legitimate public URLs pass validation.
"""

import socket
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from app.source_collector import (
    _is_prohibited_ip,
    fetch_webpage,
    is_safe_url,
)
import ipaddress


class SSRFProtectionTests(unittest.TestCase):
    def test_blocks_ipv4_loopback_addresses(self):
        loopback_urls = [
            "http://127.0.0.1/admin",
            "http://127.0.0.2:8000/",
            "http://127.255.255.254/status",
            "http://localhost:8080/",
        ]
        for url in loopback_urls:
            with self.subTest(url=url):
                safe, reason = is_safe_url(url)
                self.assertFalse(safe)
                self.assertTrue(any(term in reason.lower() for term in ("loopback", "prohibited", "resolve")))

    def test_blocks_ipv4_private_and_reserved_addresses(self):
        private_urls = [
            "http://10.0.0.1/internal",
            "http://10.254.1.1:8080/",
            "http://172.16.0.1/",
            "http://172.31.255.255/api",
            "http://192.168.1.1/router",
            "http://192.168.0.254/",
            "http://0.0.0.0/",
            "http://255.255.255.255/",
        ]
        for url in private_urls:
            with self.subTest(url=url):
                safe, reason = is_safe_url(url)
                self.assertFalse(safe)
                self.assertTrue(
                    any(t in reason.lower() for t in ("private", "loopback", "unspecified", "reserved", "multicast", "prohibited"))
                )

    def test_blocks_cloud_metadata_endpoints(self):
        metadata_urls = [
            "http://169.254.169.254/latest/meta-data/",
            "http://169.254.169.254/computeMetadata/v1/",
            "http://169.254.1.1/",
            "http://[fd00:ec2::254]/latest/meta-data/",
            "http://metadata.google.internal/computeMetadata/v1/",
            "http://instance-data/latest/meta-data/",
            "http://metadata/v1/",
        ]
        for url in metadata_urls:
            with self.subTest(url=url):
                safe, reason = is_safe_url(url)
                self.assertFalse(safe, f"Expected {url} to be blocked as cloud metadata")
                self.assertTrue(
                    any(term in reason.lower() for term in ("metadata", "link-local", "prohibited", "private", "cloud"))
                )

    def test_blocks_ipv6_loopback_and_internal_addresses(self):
        ipv6_urls = [
            "http://[::1]/",
            "http://[fe80::1]/",
            "http://[fc00::1]/",
            "http://[fd12:3456:789a::1]/",
            "http://[::]/",
            "http://[::ffff:127.0.0.1]/",
            "http://[::ffff:192.168.1.1]/",
            "http://[::ffff:169.254.169.254]/",
        ]
        for url in ipv6_urls:
            with self.subTest(url=url):
                safe, reason = is_safe_url(url)
                self.assertFalse(safe)

    def test_blocks_dangerous_schemes_and_credentials(self):
        bad_urls = [
            "file:///etc/passwd",
            "file:///C:/Windows/win.ini",
            "ftp://ftp.example.com/file",
            "gopher://evil.com/1",
            "javascript:alert(1)",
            "data:text/html,test",
            "http://admin:secret@example.com/page",
            "https://user:password@example.com/",
            "http://example.com:0/path",
            "http://example.com:70000/path",
        ]
        for url in bad_urls:
            with self.subTest(url=url):
                safe, reason = is_safe_url(url)
                self.assertFalse(safe)

    def test_blocks_internal_domain_suffixes(self):
        internal_domains = [
            "http://service.local/api",
            "http://database.lan/query",
            "http://vault.internal/secrets",
            "http://auth.corp/login",
            "http://router.home/",
            "http://host.arpa/",
            "http://internal.biz/",
        ]
        for url in internal_domains:
            with self.subTest(url=url):
                safe, reason = is_safe_url(url)
                self.assertFalse(safe)
                self.assertIn("blocked", reason.lower())

    def test_blocks_dns_resolving_to_private_ip(self):
        """Hostname resolving to 127.0.0.1 or 10.x must be blocked during DNS check."""
        with patch("socket.getaddrinfo") as mock_gai:
            mock_gai.return_value = [
                (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 80))
            ]
            safe, reason = is_safe_url("https://rebound.example.com/api")
            self.assertFalse(safe)
            self.assertIn("prohibited destination", reason.lower())

    def test_permits_legitimate_public_urls(self):
        with patch("socket.getaddrinfo") as mock_gai:
            mock_gai.return_value = [
                (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))
            ]
            safe, reason = is_safe_url("https://example.com/article")
            self.assertTrue(safe)
            self.assertEqual(reason, "")


class SSRFRedirectValidationTests(unittest.IsolatedAsyncioTestCase):
    async def test_redirect_to_private_ip_is_blocked_per_hop(self):
        """When an initial safe URL redirects to a private IP, it must be blocked before connecting."""
        mock_resp_1 = MagicMock()
        mock_resp_1.status_code = 302
        mock_resp_1.headers = {"location": "http://127.0.0.1/admin"}

        with patch("app.source_collector.is_safe_url") as mock_is_safe:
            # First call for initial URL is safe; second call for redirect target is unsafe
            mock_is_safe.side_effect = [
                (True, ""),
                (False, "Loopback address blocked: 127.0.0.1"),
            ]
            with patch("httpx.AsyncClient") as mock_client_cls:
                mock_client = AsyncMock()
                mock_client.get.return_value = mock_resp_1
                mock_client_cls.return_value.__aenter__.return_value = mock_client

                title, text, error = await fetch_webpage("https://example.com/start")
                self.assertEqual(text, "")
                self.assertIn("Redirect to prohibited destination blocked", error)
                self.assertIn("Loopback address blocked", error)

    async def test_redirect_to_cloud_metadata_is_blocked(self):
        """Redirect pointing to 169.254.169.254 is blocked before next hop."""
        mock_resp = MagicMock()
        mock_resp.status_code = 301
        mock_resp.headers = {"location": "http://169.254.169.254/latest/meta-data/"}

        with patch("app.source_collector.is_safe_url") as mock_is_safe:
            mock_is_safe.side_effect = [
                (True, ""),
                (False, "Cloud metadata address blocked: 169.254.169.254"),
            ]
            with patch("httpx.AsyncClient") as mock_client_cls:
                mock_client = AsyncMock()
                mock_client.get.return_value = mock_resp
                mock_client_cls.return_value.__aenter__.return_value = mock_client

                title, text, error = await fetch_webpage("https://example.com/redirect")
                self.assertEqual(text, "")
                self.assertIn("Redirect to prohibited destination blocked", error)

    async def test_exceeding_max_redirects_is_blocked(self):
        """Exceeding 5 redirects is halted."""
        mock_resp = MagicMock()
        mock_resp.status_code = 302
        mock_resp.headers = {"location": "https://example.com/loop"}

        with patch("app.source_collector.is_safe_url", return_value=(True, "")):
            with patch("httpx.AsyncClient") as mock_client_cls:
                mock_client = AsyncMock()
                mock_client.get.return_value = mock_resp
                mock_client_cls.return_value.__aenter__.return_value = mock_client

                title, text, error = await fetch_webpage("https://example.com/loop")
                self.assertEqual(text, "")
                self.assertEqual(error, "Too many redirects")

    async def test_detects_prohibited_socket_peer_ip_post_connection(self):
        """If raw_socket peer returns private IP after connection (DNS rebinding), it is blocked."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.history = []
        mock_resp.text = "<html><body>Some content</body></html>"

        # Simulate socket connected to loopback
        mock_socket = MagicMock()
        mock_socket.getpeername.return_value = ("127.0.0.1", 80)
        mock_resp.raw_socket = mock_socket

        with patch("app.source_collector.is_safe_url", return_value=(True, "")):
            with patch("httpx.AsyncClient") as mock_client_cls:
                mock_client = AsyncMock()
                mock_client.get.return_value = mock_resp
                mock_client_cls.return_value.__aenter__.return_value = mock_client

                title, text, error = await fetch_webpage("https://example.com/legit")
                self.assertEqual(text, "")
                self.assertIn("Connected to prohibited destination", error)

    async def test_socket_validation_exception_fails_closed(self):
        """Connected socket check must fail closed if inspecting the socket raises an exception."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.history = []
        mock_resp.text = "<html><body>Some content</body></html>"

        # Socket throws an OSError or disconnected error on getpeername
        mock_socket = MagicMock()
        mock_socket.getpeername.side_effect = OSError("Socket inspection aborted: bad file descriptor")
        mock_resp.raw_socket = mock_socket

        with patch("app.source_collector.is_safe_url", return_value=(True, "")):
            with patch("httpx.AsyncClient") as mock_client_cls:
                mock_client = AsyncMock()
                mock_client.get.return_value = mock_resp
                mock_client_cls.return_value.__aenter__.return_value = mock_client

                title, text, error = await fetch_webpage("https://example.com/check")
                self.assertEqual(text, "")
                self.assertIn("Connected socket validation failed", error)

    async def test_oversized_response_content_length_header_blocked(self):
        """Responses advertising oversized Content-Length are blocked before consumption."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.headers = {"content-length": "25000000"}  # 25 MB
        mock_resp.history = []
        mock_resp.text = "Large content..."

        with patch("app.source_collector.is_safe_url", return_value=(True, "")):
            with patch("httpx.AsyncClient") as mock_client_cls:
                mock_client = AsyncMock()
                mock_client.get.return_value = mock_resp
                mock_client_cls.return_value.__aenter__.return_value = mock_client

                title, text, error = await fetch_webpage("https://example.com/huge")
                self.assertEqual(text, "")
                self.assertIn("Response size exceeds limit", error)

    async def test_oversized_response_body_blocked(self):
        """Responses with actual content exceeding limit are blocked."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.headers = {}
        mock_resp.content = b"X" * (12 * 1024 * 1024)  # 12 MB
        mock_resp.history = []

        with patch("app.source_collector.is_safe_url", return_value=(True, "")):
            with patch("httpx.AsyncClient") as mock_client_cls:
                mock_client = AsyncMock()
                mock_client.get.return_value = mock_resp
                mock_client_cls.return_value.__aenter__.return_value = mock_client

                title, text, error = await fetch_webpage("https://example.com/oversized")
                self.assertEqual(text, "")
                self.assertIn("Response size exceeds limit", error)


if __name__ == "__main__":
    unittest.main()
