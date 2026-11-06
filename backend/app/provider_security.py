"""Security validation for LLM provider URLs and endpoints.

Prevents Server-Side Request Forgery (SSRF) and credential/API-key forwarding
by enforcing:
1. Strict HTTPS requirement for all external cloud providers.
2. Provider-specific domain allowlists (xAI, OpenAI, Google Gemini, Anthropic).
3. Blocking loopback, private, link-local, multicast, and cloud metadata
   destinations for external provider requests.
4. Local loopback confinement for local providers (Ollama), blocking network traversal.
5. DNS pre-validation to block DNS rebinding to internal infrastructure.
"""

from __future__ import annotations

import ipaddress
import logging
import socket
from urllib.parse import urlparse

from . import config

logger = logging.getLogger("memorymind.provider_security")

# Official default host allowlists per provider
PROVIDER_DEFAULT_HOSTS: dict[str, set[str]] = {
    "grok": {"api.x.ai", "x.ai"},
    "openai": {"api.openai.com"},
    "gemini": {"generativelanguage.googleapis.com"},
    "anthropic": {"api.anthropic.com"},
    "ollama": {"localhost", "127.0.0.1", "::1", "[::1]"},
}

CLOUD_METADATA_IPS: set[str] = {
    "169.254.169.254",
    "169.254.169.253",
    "fd00:ec2::254",
}

CLOUD_METADATA_HOSTS: set[str] = {
    "metadata.google.internal",
    "instance-data",
    "metadata",
}


def is_prohibited_ip(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> tuple[bool, str]:
    """Check if an IP address belongs to loopback, private, link-local, or reserved ranges."""
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped:
        ip = ip.ipv4_mapped

    if ip.is_loopback:
        return True, f"Loopback address blocked: {ip}"
    if ip.is_private:
        return True, f"Private network address blocked: {ip}"
    if ip.is_link_local:
        return True, f"Link-local address blocked: {ip}"
    if ip.is_multicast:
        return True, f"Multicast address blocked: {ip}"
    if ip.is_reserved:
        return True, f"Reserved address blocked: {ip}"
    if ip.is_unspecified:
        return True, f"Unspecified address blocked: {ip}"

    ip_str = str(ip)
    if ip_str in CLOUD_METADATA_IPS:
        return True, f"Cloud metadata address blocked: {ip_str}"

    return False, ""


def validate_provider_url(provider: str, base_url: str | None) -> tuple[bool, str]:
    """Validate a provider base URL for SSRF security and credential protection.

    Returns: (is_valid, error_message)
    """
    p = (provider or "").lower().strip()

    if p == "none":
        return True, ""

    if not base_url:
        # If no URL is provided, fallback to provider default (which is verified safe)
        return True, ""

    url_str = base_url.strip()
    if not url_str:
        return True, ""

    try:
        parsed = urlparse(url_str)
    except Exception as e:
        return False, f"Malformed provider URL: {e}"

    scheme = (parsed.scheme or "").lower()
    if scheme not in ("http", "https"):
        return False, f"Unsupported URL scheme '{scheme}'. Only HTTP and HTTPS are allowed."

    if parsed.username or parsed.password:
        return False, "Embedded credentials in provider base_url are forbidden."

    hostname = (parsed.hostname or "").lower()
    if not hostname:
        return False, "Provider base_url must include a valid hostname."

    try:
        port = parsed.port
        if port is not None and (port < 1 or port > 65535):
            return False, f"Invalid port: {port}"
    except (ValueError, Exception) as e:
        return False, f"Invalid port: {e}"

    # Check for prohibited metadata hostnames
    if hostname in CLOUD_METADATA_HOSTS or any(hostname.startswith(m) for m in ("169.254.", "ec2-", "metadata.")):
        return False, f"Prohibited cloud metadata host '{hostname}'."

    # --- Ollama validation: Local loopback only ---
    if p == "ollama":
        if scheme not in ("http", "https"):
            return False, f"Ollama requires http or https scheme, got '{scheme}'."

        # Check if direct IP
        try:
            ip = ipaddress.ip_address(hostname.strip("[]"))
            if not ip.is_loopback:
                return False, f"Ollama base_url resolves to non-loopback IP '{ip}'. Ollama must run on localhost."
            return True, ""
        except ValueError:
            pass

        # If hostname, verify it resolves strictly to loopback
        try:
            addr_info = socket.getaddrinfo(hostname, None)
            if not addr_info:
                return False, f"Could not resolve Ollama host '{hostname}'."
            for entry in addr_info:
                ip_str = entry[4][0]
                ip = ipaddress.ip_address(ip_str)
                if not ip.is_loopback:
                    return False, f"Ollama host '{hostname}' resolves to non-loopback IP '{ip_str}'."
        except socket.gaierror as e:
            return False, f"DNS resolution failed for Ollama host '{hostname}': {e}"
        except Exception as e:
            return False, f"Validation failed for Ollama host '{hostname}': {e}"

        return True, ""

    # --- External cloud providers (Grok, OpenAI, Gemini, Anthropic) ---
    if p in ("grok", "openai", "gemini", "anthropic"):
        # 1. Enforce HTTPS
        if scheme != "https":
            return False, f"External provider '{p}' strictly requires HTTPS scheme (got '{scheme}')."

        # 2. Check allowlist
        allowed_hosts = PROVIDER_DEFAULT_HOSTS.get(p, set())
        custom_allowed = getattr(config, "ALLOW_CUSTOM_PROVIDER_URLS", False)
        configured_allowed_hosts = getattr(config, "ALLOWED_PROVIDER_HOSTS", set())

        is_allowed_host = (hostname in allowed_hosts) or (hostname in configured_allowed_hosts)

        if not is_allowed_host and not custom_allowed:
            return False, (
                f"Untrusted provider host '{hostname}' for provider '{p}'. "
                f"Allowed hosts: {', '.join(sorted(allowed_hosts))}. "
                "To permit custom enterprise or proxy endpoints, configure ALLOW_CUSTOM_PROVIDER_URLS=true."
            )

        # 3. Always enforce SSRF / IP restrictions even for custom endpoints
        try:
            ip = ipaddress.ip_address(hostname.strip("[]"))
            is_bad, reason = is_prohibited_ip(ip)
            if is_bad:
                return False, f"Provider host '{hostname}' points to prohibited IP: {reason}"
            return True, ""
        except ValueError:
            pass

        # 4. Resolve DNS and inspect all target addresses
        try:
            addr_info = socket.getaddrinfo(hostname, None)
            if not addr_info:
                return False, f"DNS resolution returned no addresses for host '{hostname}'."
            for entry in addr_info:
                ip_str = entry[4][0]
                try:
                    ip = ipaddress.ip_address(ip_str)
                    is_bad, reason = is_prohibited_ip(ip)
                    if is_bad:
                        return False, f"Provider host '{hostname}' resolves to prohibited destination: {reason}"
                except ValueError:
                    return False, f"Provider host '{hostname}' resolved to malformed IP '{ip_str}'."
        except socket.gaierror as e:
            return False, f"DNS resolution failed for provider host '{hostname}': {e}"
        except Exception as e:
            return False, f"Host validation error for '{hostname}': {e}"

        return True, ""

    return False, f"Unsupported provider '{provider}'."

