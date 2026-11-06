"""Multi-document synthesis pipeline for PDF summarization.

Synthesizes findings across multiple documents into a coherent report
with cross-document comparisons and source attribution.
"""
import logging
from typing import Any

from . import config, llm

logger = logging.getLogger("memorymind.synthesis")


SYNTHESIS_SYSTEM_PROMPT = """You are an expert research synthesizer. Your task is to analyze multiple documents and create a unified, coherent summary.

CRITICAL RULES:
- Synthesize findings across ALL documents, do NOT simply concatenate individual summaries
- Identify agreements, contradictions, and differences between sources
- Preserve source attribution using [Source: filename:page N] format
- Only make claims that can be traced back to the provided documents
- When sources disagree, attribute each position to its source
- Remove redundant explanations while retaining important details
- Extract key findings, arguments, definitions, statistics, and evidence
- Identify limitations and open questions

OUTPUT STRUCTURE:
1. Executive Summary (2-3 paragraphs)
2. Main Themes (group by topic, not by document)
3. Cross-Document Analysis
   - Agreements between sources
   - Contradictions or disagreements
   - Complementary findings
4. Key Findings and Evidence
5. Limitations and Open Questions
6. Conclusions

Do not fabricate statistics, citations, or page numbers. If a claim cannot be traced to a source, state "Not verified in provided documents."
"""

SYNTHESIS_USER_PROMPT_TEMPLATE = """Analyze and synthesize the following documents:

{document_sections}

TASK:
1. Extract key findings, arguments, definitions, statistics, and evidence from each source
2. Identify themes that span multiple documents
3. Note agreements and contradictions between sources
4. Synthesize into a coherent report
5. Cite sources using [Source: filename:page N] format where applicable

Begin your synthesis:
"""


def build_document_sections(extracted_pdfs: list) -> str:
    """Build document sections for synthesis prompt."""
    sections = []
    for idx, pdf in enumerate(extracted_pdfs, start=1):
        if not pdf.success:
            continue

        section = f"<document id=\"{idx}\" filename=\"{pdf.filename}\" pages=\"{pdf.total_pages}\">\n"

        for page in pdf.extracted_pages:
            if not page.is_empty:
                section += f"<page num=\"{page.page_num}\">\n{page.text}\n</page>\n"

        section += "</document>\n"
        sections.append(section)

    return "\n".join(sections)


def synthesize_documents(extracted_pdfs: list, progress_callback=None) -> tuple[str, list[dict]]:
    """Synthesize multiple PDF documents into a coherent summary.

    Args:
        extracted_pdfs: List of ExtractedPDF objects
        progress_callback: Optional callback for progress updates

    Returns:
        (synthesis_text, citation_list) tuple
    """
    if progress_callback:
        progress_callback("Building synthesis prompt...")

    document_sections = build_document_sections(extracted_pdfs)

    if not document_sections:
        return "No documents available for synthesis.", []

    user_prompt = SYNTHESIS_USER_PROMPT_TEMPLATE.format(document_sections=document_sections)

    if progress_callback:
        progress_callback("Synthesizing findings across documents...")

    if config.LLM_PROVIDER == "grok":
        raw_response = _synthesize_with_grok(user_prompt)
    elif config.LLM_PROVIDER == "openai":
        raw_response = _synthesize_with_openai(user_prompt)
    elif config.LLM_PROVIDER == "gemini":
        raw_response = _synthesize_with_gemini(user_prompt)
    elif config.LLM_PROVIDER == "anthropic":
        raw_response = _synthesize_with_anthropic(user_prompt)
    elif config.LLM_PROVIDER == "ollama":
        raw_response = _synthesize_with_ollama(user_prompt)
    else:
        raw_response = _synthesize_without_llm(extracted_pdfs)

    if progress_callback:
        progress_callback("Extracting citations...")

    citations = extract_citations(raw_response, extracted_pdfs)

    return raw_response, citations


def _synthesize_with_grok(user_prompt: str) -> str:
    """Synthesize using Grok."""
    client = llm._get_grok_synthesis_client()
    response = client.chat.completions.create(
        model=config.GROK_MODEL,
        messages=[
            {"role": "system", "content": SYNTHESIS_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.3,
        max_tokens=2000,
        timeout=config.SYNTHESIS_TIMEOUT,
    )
    return (response.choices[0].message.content or "").strip()


def _synthesize_with_openai(user_prompt: str) -> str:
    """Synthesize using OpenAI."""
    client = llm._get_openai_synthesis_client()
    response = client.chat.completions.create(
        model=config.OPENAI_MODEL,
        messages=[
            {"role": "system", "content": SYNTHESIS_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.3,
        max_tokens=2000,
        timeout=config.SYNTHESIS_TIMEOUT,
    )
    return (response.choices[0].message.content or "").strip()


def _synthesize_with_gemini(user_prompt: str) -> str:
    """Synthesize using Gemini."""
    from google.genai import types

    client = llm._get_gemini_client()
    response = client.models.generate_content(
        model=config.GEMINI_MODEL,
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYNTHESIS_SYSTEM_PROMPT,
            temperature=0.3,
        ),
    )
    return (response.text or "").strip()


def _synthesize_with_anthropic(user_prompt: str) -> str:
    """Synthesize using Anthropic."""
    import anthropic

    client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
    response = client.messages.create(
        model=config.ANTHROPIC_MODEL,
        max_tokens=2000,
        system=SYNTHESIS_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )
    return response.content[0].text.strip()


def _synthesize_with_ollama(user_prompt: str) -> str:
    """Synthesize using Ollama local endpoint."""
    return llm._complete_ollama(user_prompt, SYNTHESIS_SYSTEM_PROMPT)


def _synthesize_without_llm(extracted_pdfs: list) -> str:
    """Fallback when no LLM is configured."""
    lines = ["# Document Summary (No LLM configured)\n"]
    lines.append("Extracted text from the following documents:\n")

    for pdf in extracted_pdfs:
        if pdf.success:
            lines.append(f"## {pdf.filename}\n")
            lines.append(f"Pages: {pdf.page_count}/{pdf.total_pages}\n")
            lines.append(pdf.get_full_text()[:500] + "...\n\n")

    return "\n".join(lines)


def extract_citations(text: str, extracted_pdfs: list) -> list[dict]:
    """Extract and validate citations from synthesis text.

    Returns list of citation dicts with source info.
    """
    import re

    citations = []
    citation_pattern = r"\[Source:\s*([^:]+):page\s*(\d+)\]"

    matches = re.findall(citation_pattern, text)

    for filename, page_num in matches:
        page_num = int(page_num)
        filename = filename.strip()

        source_pdf = None
        for pdf in extracted_pdfs:
            if pdf.filename == filename:
                source_pdf = pdf
                break

        if source_pdf:
            page_text = ""
            for page in source_pdf.extracted_pages:
                if page.page_num == page_num:
                    page_text = page.text[:200] if page.text else ""
                    break

            citations.append({
                "filename": filename,
                "page": page_num,
                "preview": page_text,
            })

    return citations


def chunk_for_synthesis(text: str, max_chars: int = 8000) -> list[str]:
    """Split large text into chunks suitable for LLM processing."""
    if len(text) <= max_chars:
        return [text]

    chunks = []
    paragraphs = text.split("\n\n")
    current_chunk = ""

    for para in paragraphs:
        if len(current_chunk) + len(para) + 2 <= max_chars:
            current_chunk += para + "\n\n"
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = para + "\n\n"

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks
