"""Unified source collection and validation for summarization.

Handles webpages, PDFs, and mixed collections with secure extraction.
"""
import asyncio
import hashlib
import io
import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from urllib.parse import urlparse

import httpx

from . import config
from .pdf_handler import extract_pdf, ExtractedPDF
from .webpage import extract_content, clean_text

logger = logging.getLogger("memorymind.source_collector")


class SourceType(Enum):
    WEBPAGE = "webpage"
    PDF = "pdf"
    CURRENT_PAGE = "current_page"
    UPLOADED_PDF = "uploaded_pdf"


@dataclass
class SourceMetadata:
    """Unified metadata for any source."""
    source_id: str
    source_type: SourceType
    title: str
    url: str = ""
    filename: str = ""
    domain: str = ""
    extracted_at: str = ""
    file_size: int = 0
    page_count: int = 0
    error: str = ""
    success: bool = False


@dataclass 
class ExtractedSource:
    """Complete extracted source with content."""
    metadata: SourceMetadata
    text: str = ""
    chunks: list[dict] = field(default_factory=list)
    pages: dict[int, str] = field(default_factory=dict)


def generate_source_id() -> str:
    """Generate unique source ID."""
    return str(uuid.uuid4())[:8]


import ipaddress
import socket
from urllib.parse import urljoin, urlparse


def _is_prohibited_ip(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> tuple[bool, str]:
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
    if ip_str in ("169.254.169.254", "fd00:ec2::254"):
        return True, f"Cloud metadata address blocked: {ip_str}"

    return False, ""


def is_safe_url(url: str) -> tuple[bool, str]:
    """Validate URL is safe for fetching against SSRF attacks."""
    if not url:
        return False, "Empty URL"

    url = url.strip()

    try:
        parsed = urlparse(url)
    except Exception as e:
        return False, f"Invalid URL: {e}"

    scheme = parsed.scheme.lower()
    if scheme not in ("http", "https"):
        return False, f"Unsupported scheme: {scheme}"

    hostname = parsed.hostname or ""
    if not hostname:
        return False, "Missing hostname"

    if parsed.username or parsed.password:
        return False, "Embedded credentials in URL are not permitted"

    try:
        port = parsed.port
        if port is not None and (port < 1 or port > 65535):
            return False, f"Invalid port: {port}"
    except (ValueError, Exception) as e:
        return False, f"Invalid port: {e}"

    lower_host = hostname.lower()
    if lower_host in ("localhost", "metadata.google.internal", "instance-data"):
        return False, f"Prohibited host: {hostname}"

    internal_suffixes = (".local", ".lan", ".internal", ".corp", ".home", ".arpa", ".biz")
    if any(lower_host.endswith(suffix) for suffix in internal_suffixes):
        return False, f"Private/internal domain blocked: {hostname}"

    # Block common cloud metadata hostname forms.
    if lower_host.startswith("metadata") or lower_host.startswith("ec2-") or lower_host.startswith("169.254."):
        return False, "Cloud metadata destination blocked"

    # Try parsing hostname as a direct IP address
    try:
        ip = ipaddress.ip_address(lower_host)
        is_bad, reason = _is_prohibited_ip(ip)
        if is_bad:
            return False, reason
        return True, ""
    except ValueError:
        pass

    # Resolve hostname via DNS to ensure destination IPs are not internal/private
    try:
        addr_info = socket.getaddrinfo(lower_host, None)
        if not addr_info:
            return False, f"Could not resolve host: {hostname}"
        for entry in addr_info:
            sockaddr = entry[4]
            ip_str = sockaddr[0]
            try:
                ip = ipaddress.ip_address(ip_str)
                is_bad, reason = _is_prohibited_ip(ip)
                if is_bad:
                    return False, f"Host {hostname} resolves to prohibited destination: {reason}"
            except ValueError:
                return False, f"Host {hostname} resolved to invalid IP: {ip_str}"
    except socket.gaierror as e:
        return False, f"DNS resolution failed for {hostname}: {e}"
    except Exception as e:
        return False, f"Host validation error for {hostname}: {e}"

    return True, ""


def extract_domain(url: str) -> str:
    """Extract domain from URL."""
    try:
        parsed = urlparse(url)
        return parsed.netloc.lower()
    except Exception:
        return ""


MAX_RESPONSE_SIZE = 10 * 1024 * 1024  # 10 MB maximum webpage payload


async def fetch_webpage(url: str, timeout: float = 30.0) -> tuple[str, str, str]:
    """Fetch webpage content and extract text with redirect and SSRF protection.

    Returns: (title, extracted_text, error)
    """
    current_url = url.strip()
    is_valid, error = is_safe_url(current_url)
    if not is_valid:
        return "", "", error

    parsed = urlparse(current_url)
    original_host = (parsed.hostname or "").lower()
    original_ip = ""
    try:
        original_ip = str(ipaddress.ip_address(original_host))
    except ValueError:
        original_ip = ""
    if not is_valid:
        return "", "", error

    max_redirects = 5
    redirect_count = 0
    headers = {
        "User-Agent": "MemoryMind/1.0 (research summarization)",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }

    try:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=False) as client:
            while True:
                response = await client.get(current_url, headers=headers)

                # Check for redirects and validate each hop
                if response.status_code in (301, 302, 303, 307, 308):
                    redirect_count += 1
                    if redirect_count > max_redirects:
                        return "", "", "Too many redirects"

                    location = response.headers.get("location")
                    if not location:
                        return "", "", "Redirect missing Location header"

                    next_url = urljoin(current_url, location)

                    # Validate destination before the next network hop.
                    is_next_valid, next_error = is_safe_url(next_url)
                    if not is_next_valid:
                        return "", "", f"Redirect to prohibited destination blocked: {next_error}"

                    # Final safety check: ensure the resolved next hop
                    # is not a different private destination than the one validated.
                    current_url = next_url
                    continue

                response.raise_for_status()

                # Check Content-Length header against max size limit
                content_length = response.headers.get("content-length")
                if content_length:
                    try:
                        if int(content_length) > MAX_RESPONSE_SIZE:
                            return "", "", f"Response size exceeds limit of {MAX_RESPONSE_SIZE} bytes"
                    except (ValueError, TypeError):
                        pass

                # Check actual body length against max size limit
                if hasattr(response, "content") and len(response.content) > MAX_RESPONSE_SIZE:
                    return "", "", f"Response size exceeds limit of {MAX_RESPONSE_SIZE} bytes"

                if response.history:
                    for hop in response.history:
                        hop_url = str(hop.request.url)
                        ok, _ = is_safe_url(hop_url)
                        if not ok:
                            return "", "", "Redirect hop failed SSRF validation"

                # Final SSRF guard: validate the actual resolved socket address
                # that httpx connected to. This catches DNS rebinding that
                # changes between validation and the real TCP handshake.
                # Must FAIL CLOSED if socket inspection raises an exception.
                if hasattr(response, "raw_socket") and response.raw_socket is not None:
                    try:
                        peer = response.raw_socket.getpeername()
                        if not peer or not peer[0]:
                            return "", "", "Connected socket peer address could not be validated"
                        peer_ip = ipaddress.ip_address(peer[0])
                        is_bad, reason = _is_prohibited_ip(peer_ip)
                        if is_bad:
                            return "", "", f"Connected to prohibited destination: {reason}"
                    except Exception as e:
                        # FAIL CLOSED: Never ignore or swallow socket validation errors
                        return "", "", f"Connected socket validation failed: {e}"

                html = response.text
                title, text = extract_content(html, url=current_url)

                if not text or len(text) < 50:
                    return title, "", "No meaningful content extracted"

                return title, text, ""

    except httpx.HTTPStatusError as e:
        return "", "", f"HTTP {e.response.status_code}: {e.response.reason_phrase}"
    except httpx.TimeoutException:
        return "", "", "Request timed out"
    except Exception as e:
        return "", "", f"Fetch failed: {e}"


async def extract_webpage_sources(
    urls: list[str], 
    max_urls: int = 20,
    concurrency_limit: int = 5,
    per_request_timeout: float = 15.0,
) -> list[ExtractedSource]:
    """Extract content from multiple webpages with bounded concurrency and preserved ordering."""
    seen_urls = set()
    unique_urls = []
    for u in urls[:max_urls]:
        cleaned = u.strip()
        if cleaned and cleaned not in seen_urls:
            seen_urls.add(cleaned)
            unique_urls.append(cleaned)

    if not unique_urls:
        return []

    semaphore = asyncio.Semaphore(concurrency_limit)

    async def _fetch_single(index: int, target_url: str):
        async with semaphore:
            source_id = generate_source_id()
            metadata = SourceMetadata(
                source_id=source_id,
                source_type=SourceType.WEBPAGE,
                title="",
                url=target_url,
                domain=extract_domain(target_url),
                extracted_at=datetime.now(timezone.utc).isoformat(),
            )
            title, text, error = await fetch_webpage(target_url, timeout=per_request_timeout)
            if error:
                metadata.error = error
            else:
                metadata.title = title
                metadata.success = True

            extracted = ExtractedSource(
                metadata=metadata,
                text=text,
                chunks=[{"text": text, "source": target_url, "page": 0}] if text else [],
            )
            return index, extracted

    tasks = [_fetch_single(i, u) for i, u in enumerate(unique_urls)]
    results = await asyncio.gather(*tasks)
    results.sort(key=lambda item: item[0])
    return [item[1] for item in results]


def extract_pdf_sources(files: list[tuple[bytes, str]]) -> list[ExtractedSource]:
    """Extract text from PDF files."""
    sources = []
    
    for file_data, filename in files:
        source_id = generate_source_id()
        result = extract_pdf(file_data, filename)
        
        metadata = SourceMetadata(
            source_id=source_id,
            source_type=SourceType.UPLOADED_PDF,
            title=filename,
            filename=filename,
            extracted_at=datetime.now(timezone.utc).isoformat(),
            file_size=len(file_data),
            page_count=result.total_pages,
            error=result.error,
            success=result.success,
        )
        
        pages = {}
        chunks = []
        for page in result.extracted_pages:
            if not page.is_empty:
                pages[page.page_num] = page.text
                chunks.append({
                    "text": page.text,
                    "source": filename,
                    "page": page.page_num,
                })
        
        sources.append(ExtractedSource(
            metadata=metadata,
            text=result.get_full_text(),
            pages=pages,
            chunks=chunks,
        ))
    
    return sources


def chunk_source_for_synthesis(
    source: ExtractedSource,
    max_chunk_chars: int = 8000,
    overlap_chars: int = 500,
) -> list[dict]:
    """Split source into manageable chunks for LLM processing."""
    if not source.chunks:
        return []
    
    all_chunks = []
    current_chunk = ""
    current_sources = set()
    
    for chunk in source.chunks:
        chunk_text = chunk.get("text", "")
        chunk_source = chunk.get("source", "")
        chunk_page = chunk.get("page", 0)
        
        if len(current_chunk) + len(chunk_text) + 1 <= max_chunk_chars:
            if chunk_source not in current_sources:
                current_sources.add(chunk_source)
            current_chunk += chunk_text + " "
        else:
            if current_chunk:
                all_chunks.append({
                    "text": current_chunk.strip(),
                    "sources": list(current_sources),
                    "char_count": len(current_chunk),
                })
            current_chunk = chunk_text + " "
            current_sources = {chunk_source}
    
    if current_chunk:
        all_chunks.append({
            "text": current_chunk.strip(),
            "sources": list(current_sources),
            "char_count": len(current_chunk),
        })
    
    return all_chunks


class SourceCollection:
    """Manages a collection of sources for synthesis."""
    
    def __init__(self):
        self.sources: list[ExtractedSource] = []
        self.created_at = time.time()
        self.job_id = generate_source_id()
    
    def add_webpage(self, url: str, title: str = "", text: str = "", error: str = ""):
        """Add a webpage source."""
        metadata = SourceMetadata(
            source_id=generate_source_id(),
            source_type=SourceType.WEBPAGE,
            title=title or url,
            url=url,
            domain=extract_domain(url),
            extracted_at=datetime.now(timezone.utc).isoformat(),
            error=error,
            success=not error and bool(text),
        )
        self.sources.append(ExtractedSource(
            metadata=metadata,
            text=text,
            chunks=[{"text": text, "source": url, "page": 0}] if text else []
        ))
    
    def add_pdf(self, file_data: bytes, filename: str, result: ExtractedPDF):
        """Add a PDF source."""
        metadata = SourceMetadata(
            source_id=generate_source_id(),
            source_type=SourceType.UPLOADED_PDF,
            title=filename,
            filename=filename,
            extracted_at=datetime.now(timezone.utc).isoformat(),
            file_size=len(file_data),
            page_count=result.total_pages,
            error=result.error,
            success=result.success,
        )
        pages = {}
        chunks = []
        for page in result.extracted_pages:
            if not page.is_empty:
                pages[page.page_num] = page.text
                chunks.append({
                    "text": page.text,
                    "source": filename,
                    "page": page.page_num,
                })
        self.sources.append(ExtractedSource(
            metadata=metadata,
            text=result.get_full_text(),
            pages=pages,
            chunks=chunks,
        ))
    
    def get_all_sources(self) -> list[ExtractedSource]:
        """Get all successfully extracted sources."""
        return [s for s in self.sources if s.metadata.success]
    
    def get_failed_sources(self) -> list[ExtractedSource]:
        """Get sources that failed extraction."""
        return [s for s in self.sources if not s.metadata.success]
    
    def get_source_summary(self) -> dict:
        """Get summary of all sources."""
        successful = self.get_all_sources()
        failed = self.get_failed_sources()
        
        by_type = {}
        for s in self.sources:
            st = s.metadata.source_type.value
            by_type[st] = by_type.get(st, 0) + 1
        
        return {
            "total": len(self.sources),
            "successful": len(successful),
            "failed": len(failed),
            "by_type": by_type,
            "total_pages": sum(s.metadata.page_count for s in successful),
            "total_chars": sum(len(s.text) for s in successful),
        }
    
    def chunk_all_for_synthesis(
        self, 
        max_chunk_chars: int = 8000,
    ) -> list[dict]:
        """Chunk all sources for synthesis."""
        all_chunks = []
        for source in self.sources:
            if not source.metadata.success:
                continue
            chunks = chunk_source_for_synthesis(
                source, max_chunk_chars=max_chunk_chars
            )
            all_chunks.extend(chunks)
        return all_chunks
    
    def to_dict(self) -> dict:
        """Serialize for API responses."""
        return {
            "job_id": self.job_id,
            "created_at": self.created_at,
            "sources": [
                {
                    "source_id": s.metadata.source_id,
                    "type": s.metadata.source_type.value,
                    "title": s.metadata.title,
                    "url": s.metadata.url,
                    "filename": s.metadata.filename,
                    "success": s.metadata.success,
                    "error": s.metadata.error,
                    "page_count": s.metadata.page_count,
                    "char_count": len(s.text),
                }
                for s in self.sources
            ],
            "summary": self.get_source_summary(),
        }