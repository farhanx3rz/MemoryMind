"""Generic webpage representation, content extraction, and chunking.

Wikipedia is deliberately not referenced here. The same WebDocument and
extraction/chunking functions are used for every webpage source.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlparse

from . import config


@dataclass
class WebDocument:
    """Source-neutral representation of a webpage moving through RECALL."""

    url: str
    title: str
    source: str
    extracted_text: str
    chunks: list[str] = field(default_factory=list)
    timestamp: str = ""
    document_id: str = ""
    topics: list[dict[str, Any]] = field(default_factory=list)
    tags: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def domain(self) -> str:
        return self.source


def clean_text(text: str) -> str:
    """Normalize extracted webpage text without changing its meaning."""
    text = (text or "").replace("\xa0", " ")
    text = " ".join(text.split())
    return text.strip()


def validate_source_url(url: str | None, allow_none: bool = True) -> str | None:
    """Validate that a URL is a valid, well-formed http or https URL.

    Rejects:
    - Dangerous schemes (javascript:, file:, data:, vbscript:, about:, chrome:)
    - Missing or empty host/network location
    - Control characters, embedded credentials (user:pass@), or malformed ports
    """
    if url is None:
        if allow_none:
            return None
        raise ValueError("URL must not be empty.")

    cleaned = url.strip()
    if not cleaned:
        if allow_none:
            return None
        raise ValueError("URL must not be empty.")

    if any(ord(c) < 32 or ord(c) == 127 for c in cleaned):
        raise ValueError("URL contains invalid control characters.")

    try:
        parsed = urlparse(cleaned)
    except Exception as exc:
        raise ValueError(f"Malformed URL: {exc}") from exc

    scheme = parsed.scheme.lower()
    if scheme not in ("http", "https"):
        raise ValueError(
            f"Unsupported URL scheme '{scheme}'. Only http and https URLs are accepted."
        )

    hostname = parsed.hostname
    if not hostname:
        raise ValueError("URL must include a valid host name.")

    if parsed.username or parsed.password:
        raise ValueError("URLs with embedded credentials are not permitted.")

    try:
        if parsed.port is not None and not (1 <= parsed.port <= 65535):
            raise ValueError("URL port must be between 1 and 65535.")
    except ValueError as exc:
        raise ValueError(f"Invalid URL port: {exc}") from exc

    return cleaned


def source_domain(url: str) -> str:
    """Return the network domain for a source URL, or an empty string."""
    try:
        parsed = urlparse(url or "")
        return parsed.netloc.lower()
    except ValueError:
        return ""


def _fallback_extract(html: str) -> tuple[str, str]:
    """Small dependency-light fallback when Trafilatura cannot extract text."""
    try:
        from bs4 import BeautifulSoup
    except ImportError as exc:
        raise RuntimeError(
            "Content extraction requires `trafilatura` or `beautifulsoup4`."
        ) from exc

    soup = BeautifulSoup(html, "html.parser")
    for element in soup(
        [
            "script",
            "style",
            "noscript",
            "template",
            "nav",
            "footer",
            "header",
            "aside",
            "form",
            "dialog",
        ]
    ):
        element.decompose()

    for element in soup.find_all(
        attrs={"role": re.compile(r"^(navigation|banner|contentinfo|complementary)$", re.I)}
    ):
        element.decompose()

    title = clean_text((soup.title.get_text(" ") if soup.title else ""))
    root = soup.find("article") or soup.find("main") or soup.body or soup
    text = clean_text(root.get_text(" ", strip=True))
    return title, text


def prune_html_bloat(html: str, max_chars: int | None = None) -> str:
    """Remove non-content bloat (scripts, styles, SVGs, canvases, data URIs)
    from large HTML payloads to prevent payload overflow and speed up extraction.
    """
    if not html:
        return ""
    if max_chars is not None and len(html) <= max_chars:
        return html

    cleaned = html
    # Fast regex cleanup of base64 data URIs
    if "data:" in cleaned:
        cleaned = re.sub(r'data:[^;\'"\s]+;base64,[A-Za-z0-9+/=]+', '', cleaned)
        if max_chars is not None and len(cleaned) <= max_chars:
            return cleaned

    # Strip scripts, styles, inline SVGs, and canvas tags
    cleaned = re.sub(r'<script\b[^>]*>[\s\S]*?<\/script>', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'<style\b[^>]*>[\s\S]*?<\/style>', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'<svg\b[^>]*>[\s\S]*?<\/svg>', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'<canvas\b[^>]*>[\s\S]*?<\/canvas>', '', cleaned, flags=re.IGNORECASE)

    if max_chars is not None and len(cleaned) > max_chars:
        cleaned = cleaned[:max_chars]

    return cleaned


def extract_content(
    html: str,
    url: str = "",
    title_hint: str = "",
) -> tuple[str, str]:
    """Extract article/main content and title from arbitrary webpage HTML.

    Trafilatura is the primary extractor. Its documented extraction pipeline
    uses its own rules with readability/jusText fallbacks. A small BeautifulSoup
    cleanup fallback keeps simple pages usable when the extractor is unavailable
    or returns too little content.
    """
    html = html or ""
    if not html.strip():
        raise ValueError("HTML content must not be empty")

    if len(html) > config.WEBPAGE_MAX_HTML_CHARS:
        html = prune_html_bloat(html, max_chars=config.WEBPAGE_MAX_HTML_CHARS)

    title = clean_text(title_hint)
    extracted = ""
    try:
        from trafilatura import extract

        extracted = clean_text(extract(html, url=url or None) or "")
    except Exception:
        extracted = ""

    fallback_title, fallback_text = _fallback_extract(html)
    if not title:
        title = fallback_title
    if len(extracted) < config.MIN_EXTRACTED_CHARS:
        extracted = fallback_text

    if not extracted:
        raise ValueError("Could not extract meaningful webpage content.")
    return title, extracted


def chunk_text(
    text: str,
    chunk_size: int = config.WEBPAGE_CHUNK_SIZE,
    overlap: int = config.WEBPAGE_CHUNK_OVERLAP,
) -> list[str]:
    """Split long webpage text into overlapping word-window chunks."""
    words = re.findall(r"\S+", clean_text(text))
    if not words:
        return []
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("chunk overlap must be non-negative and smaller than chunk size")

    step = chunk_size - overlap
    chunks: list[str] = []
    for start in range(0, len(words), step):
        chunk = " ".join(words[start : start + chunk_size]).strip()
        if chunk:
            chunks.append(chunk)
        if start + chunk_size >= len(words):
            break
    return chunks


def from_extracted_text(
    *,
    url: str = "",
    title: str = "",
    text: str,
    timestamp: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> WebDocument:
    """Build a WebDocument when extraction has already happened elsewhere."""
    cleaned = clean_text(text)
    if not cleaned:
        raise ValueError("Extracted text must not be empty")
    validated_url = validate_source_url(url, allow_none=True) or ""
    return WebDocument(
        url=validated_url,
        title=clean_text(title) or "Untitled webpage",
        source=source_domain(validated_url),
        extracted_text=cleaned,
        chunks=chunk_text(cleaned),
        timestamp=timestamp or datetime.now(timezone.utc).isoformat(),
        metadata=metadata or {},
    )


def from_html(
    *,
    url: str,
    html: str,
    title: str = "",
    timestamp: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> WebDocument:
    """Extract and build a source-neutral WebDocument from webpage HTML."""
    validated_url = validate_source_url(url, allow_none=False)
    extracted_title, extracted_text = extract_content(html, url=validated_url, title_hint=title)
    document = from_extracted_text(
        url=validated_url,
        title=extracted_title,
        text=extracted_text,
        timestamp=timestamp,
        metadata=metadata,
    )
    document.metadata.setdefault("source_type", "webpage")
    return document
