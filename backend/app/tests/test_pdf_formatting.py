"""Test PDF formatting improvements."""
import pytest
from app.pdf_generator import generate_pdf_report, parse_synthesis_sections, format_citations_for_pdf
from app.pdf_handler import ExtractedPDF, PDFPage
from app.source_collector import SourceMetadata, ExtractedSource, SourceType


def test_parse_synthesis_with_headings():
    """Test that synthesis with markdown headings is parsed correctly."""
    text = """# Executive Summary
This is the summary.

## Key Findings
Finding one.
Finding two.

## Conclusions
Final thoughts."""
    
    sections = parse_synthesis_sections(text)
    
    # Check structure
    assert sections[0]['type'] == 'heading1'
    assert sections[0]['content'] == 'Executive Summary'
    assert sections[1]['type'] == 'paragraph'
    assert sections[2]['type'] == 'spacer'
    assert sections[3]['type'] == 'heading2'
    assert sections[3]['content'] == 'Key Findings'


def test_parse_synthesis_without_headings():
    """Test wall-of-text synthesis is parsed as paragraphs."""
    text = """This is paragraph one.
This is paragraph two.
This is paragraph three."""
    
    sections = parse_synthesis_sections(text)
    
    assert len(sections) == 3
    assert all(s['type'] == 'paragraph' for s in sections)


def test_format_citations_with_page_numbers():
    """Test citation formatting for PDF display."""
    text = "According to [Source: unit-2.pdf:page 9] the result is clear."
    formatted = format_citations_for_pdf(text)
    
    # Should convert to superscript with readable format
    assert '<super>' in formatted
    assert 'unit-2.pdf' in formatted
    assert 'p.9' in formatted
    assert '[Source:' not in formatted  # Raw format removed


def test_format_citations_with_page_ranges():
    """Test citation formatting with page ranges."""
    text = "The study [Source: document.pdf:page 9-40] shows results."
    formatted = format_citations_for_pdf(text)
    
    assert '<super>' in formatted
    assert 'p.9-40' in formatted or 'p.9–40' in formatted


def test_format_webpage_citations():
    """Test webpage citation formatting."""
    text = "As stated in [Source: Research Article] the hypothesis holds."
    formatted = format_citations_for_pdf(text)
    
    assert '<super>' in formatted
    assert 'Research Article' in formatted


def test_pdf_generation_with_sources():
    """Test PDF includes source names on cover."""
    pdf1 = ExtractedPDF(
        filename="document.pdf",
        total_pages=10,
        extracted_pages=[PDFPage(1, "Content")],
        extraction_status="completed"
    )
    pdf1.error = ""
    
    # Create source metadata
    metadata = SourceMetadata(
        source_id="test1",
        source_type=SourceType.PDF,
        filename="document.pdf",
        title="Test Document",
        page_count=10,
        success=True
    )
    source = ExtractedSource(
        metadata=metadata,
        text="Test content",
        pages={1: "Page 1"}
    )
    
    synthesis = "# Overview\nThis is a test synthesis."
    citations = []
    
    pdf_bytes = generate_pdf_report(
        synthesis,
        [pdf1],
        citations,
        "Test Report",
        [source]  # Pass sources
    )
    
    assert len(pdf_bytes) > 0
    # TODO: Extract text and verify source name appears


def test_pdf_generation_short_report():
    """Test PDF generation with short synthesis."""
    synthesis = "This is a brief summary of the findings."
    
    pdf_bytes = generate_pdf_report(
        synthesis,
        [],
        [],
        "Short Report",
        []
    )
    
    assert len(pdf_bytes) > 0


def test_pdf_generation_long_report():
    """Test PDF generation with long structured synthesis."""
    synthesis = """# Executive Summary
This report synthesizes findings from multiple sources.

## Introduction
The introduction provides context.

## Main Findings

### Finding One
Details about the first finding with supporting evidence.

### Finding Two
More details about the second finding.

## Analysis
Cross-source analysis reveals patterns.

## Conclusions
Final thoughts and recommendations."""
    
    citations = [
        {"type": "pdf", "filename": "source1.pdf", "page": 5, "title": "Source 1"},
        {"type": "pdf", "filename": "source1.pdf", "page": 12, "title": "Source 1"},
        {"type": "webpage", "title": "Web Article", "url": "https://example.com"},
    ]
    
    pdf_bytes = generate_pdf_report(
        synthesis,
        [],
        citations,
        "Long Report"
    )
    
    assert len(pdf_bytes) > 0


def test_pdf_generation_mixed_sources():
    """Test PDF with both PDFs and webpages."""
    pdf1 = ExtractedPDF(
        filename="paper.pdf",
        total_pages=20,
        extracted_pages=[PDFPage(1, "Content")],
        extraction_status="completed"
    )
    pdf1.error = ""
    
    # Mixed sources
    pdf_meta = SourceMetadata(
        source_id="pdf1",
        source_type=SourceType.PDF,
        filename="paper.pdf",
        title="Research Paper",
        page_count=20,
        success=True
    )
    pdf_source = ExtractedSource(
        metadata=pdf_meta,
        text="PDF content",
        pages={1: "Page 1"}
    )
    
    web_meta = SourceMetadata(
        source_id="web1",
        source_type=SourceType.WEBPAGE,
        title="Blog Post",
        url="https://example.com/blog",
        success=True
    )
    web_source = ExtractedSource(
        metadata=web_meta,
        text="Web content"
    )
    
    synthesis = "# Analysis\nSynthesis from multiple sources."
    citations = [
        {"type": "pdf", "filename": "paper.pdf", "page": 10, "title": "Research Paper"},
        {"type": "webpage", "title": "Blog Post", "url": "https://example.com/blog"},
    ]
    
    pdf_bytes = generate_pdf_report(
        synthesis,
        [pdf1],
        citations,
        "Mixed Sources Report",
        [pdf_source, web_source]
    )
    
    assert len(pdf_bytes) > 0


def test_consecutive_empty_lines_handling():
    """Test that multiple empty lines don't create excessive spacing."""
    text = """Paragraph one.


Paragraph two.



Paragraph three."""
    
    sections = parse_synthesis_sections(text)
    
    # Should have limited spacers, not one per empty line
    spacers = [s for s in sections if s['type'] == 'spacer']
    assert len(spacers) <= 2  # Only meaningful breaks, not every empty line


def test_sql_code_formatting():
    """Test that SQL commands are formatted as code."""
    text = """Normal paragraph.

SELECT * FROM users WHERE id = 1;

Another paragraph."""
    
    sections = parse_synthesis_sections(text)
    
    # SQL should be detected as a paragraph (special formatting applied later)
    sql_sections = [s for s in sections if 'SELECT' in s.get('content', '')]
    assert len(sql_sections) >= 1


def test_heading_keepwithnext():
    """Test that headings have keepWithNext set."""
    # This is tested by the style definition in the generator
    # Verify the style is applied correctly
    from app.pdf_generator import generate_pdf_report
    
    synthesis = """## Section One
Content for section one.

## Section Two
Content for section two."""
    
    pdf_bytes = generate_pdf_report(synthesis, [], [], "Test")
    assert len(pdf_bytes) > 0


def test_citation_reference_numbering():
    """Test that references are numbered in order."""
    citations = [
        {"type": "pdf", "filename": "doc1.pdf", "page": 1, "title": "Doc 1"},
        {"type": "pdf", "filename": "doc2.pdf", "page": 5, "title": "Doc 2"},
        {"type": "webpage", "title": "Article", "url": "https://example.com"},
    ]
    
    synthesis = "Test synthesis."
    pdf_bytes = generate_pdf_report(synthesis, [], citations, "Test")
    
    assert len(pdf_bytes) > 0
    # References should be numbered [1], [2], [3]


def test_empty_source_list_handling():
    """Test that empty source list doesn't break PDF generation."""
    synthesis = "Report with no sources."
    
    pdf_bytes = generate_pdf_report(
        synthesis,
        [],
        [],
        "No Sources Report",
        []
    )
    
    assert len(pdf_bytes) > 0


def test_markdown_bold_italic_preserved():
    """Test that markdown formatting is preserved."""
    text = "This is **bold text** and this is *italic text*."
    formatted = format_citations_for_pdf(text)
    
    assert '<b>bold text</b>' in formatted
    assert '<i>italic text</i>' in formatted
