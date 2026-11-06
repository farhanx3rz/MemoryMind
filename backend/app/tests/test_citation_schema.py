"""Test citation schema consistency between extraction and PDF generation."""
import pytest
from app.summarization_jobs import JobManager, JobConfig, SummaryStyle
from app.source_collector import SourceCollection, SourceMetadata, ExtractedSource, SourceType
from app.pdf_generator import generate_pdf_report
from app.pdf_handler import ExtractedPDF, PDFPage


def test_pdf_citation_extraction():
    """Test PDF citation with filename and page."""
    manager = JobManager()
    
    # Create source with known filename
    collection = SourceCollection()
    metadata = SourceMetadata(
        source_id="test1",
        source_type=SourceType.PDF,
        filename="Project Experiment.pdf",
        title="Project Experiment",
        page_count=5,
        success=True,
    )
    source = ExtractedSource(
        metadata=metadata,
        text="Sample PDF content",
        pages={1: "Page 1 content", 2: "Page 2 content"}
    )
    collection.sources.append(source)
    
    # Extract citations with the actual source
    text = "The study shows [Source: Project Experiment.pdf:page 2] significant results."
    citations = manager._extract_citations(text, collection.get_all_sources())
    
    assert len(citations) == 1
    assert citations[0]['type'] == 'pdf'
    assert citations[0]['filename'] == 'Project Experiment.pdf'
    assert citations[0]['page'] == 2
    assert citations[0]['title'] == 'Project Experiment'


def test_citation_with_source_field_compatibility():
    """Test that old 'source' field format is normalized to 'filename'."""
    manager = JobManager()
    
    collection = SourceCollection()
    metadata = SourceMetadata(
        source_id="test1",
        source_type=SourceType.PDF,
        filename="document.pdf",
        title="Document",
        page_count=3,
        success=True,
    )
    source = ExtractedSource(
        metadata=metadata,
        text="Content",
        pages={1: "Page 1"}
    )
    collection.sources.append(source)
    
    text = "According to [Source: document.pdf:page 1] the results are clear."
    citations = manager._extract_citations(text, collection.get_all_sources())
    
    assert citations[0]['type'] == 'pdf'
    assert 'filename' in citations[0]
    assert citations[0]['filename'] == 'document.pdf'
    assert 'source' not in citations[0]  # Old field should not exist


def test_webpage_citation_extraction():
    """Test webpage citation with title."""
    manager = JobManager()
    
    collection = SourceCollection()
    metadata = SourceMetadata(
        source_id="test1",
        source_type=SourceType.WEBPAGE,
        title="Research Article",
        url="https://example.com/article",
        success=True,
    )
    source = ExtractedSource(
        metadata=metadata,
        text="Article content"
    )
    collection.sources.append(source)
    
    text = "As stated in [Source: Research Article] the hypothesis was confirmed."
    citations = manager._extract_citations(text, collection.get_all_sources())
    
    assert len(citations) == 1
    assert citations[0]['type'] == 'webpage'
    assert citations[0]['title'] == 'Research Article'
    assert citations[0]['url'] == 'https://example.com/article'


def test_empty_citations():
    """Test handling of empty citation list."""
    collection = SourceCollection()
    
    manager = JobManager()
    text = "No citations in this text."
    citations = manager._extract_citations(text, collection.get_all_sources())
    
    assert citations == []


def test_mixed_pdf_and_webpage_citations():
    """Test document with both PDF and webpage citations."""
    manager = JobManager()
    
    collection = SourceCollection()
    
    # Add PDF source
    pdf_meta = SourceMetadata(
        source_id="pdf1",
        source_type=SourceType.PDF,
        filename="paper.pdf",
        title="Research Paper",
        page_count=10,
        success=True,
    )
    collection.sources.append(ExtractedSource(
        metadata=pdf_meta,
        text="PDF content",
        pages={3: "Page 3"}
    ))
    
    # Add webpage source
    web_meta = SourceMetadata(
        source_id="web1",
        source_type=SourceType.WEBPAGE,
        title="Blog Post",
        url="https://example.com/blog",
        success=True,
    )
    collection.sources.append(ExtractedSource(
        metadata=web_meta,
        text="Blog content"
    ))
    
    text = """The study [Source: paper.pdf:page 3] found interesting results.
    This was confirmed by [Source: Blog Post] in a related article."""
    
    citations = manager._extract_citations(text, collection.get_all_sources())
    
    assert len(citations) == 2
    
    # PDF citation
    pdf_cit = [c for c in citations if c['type'] == 'pdf'][0]
    assert pdf_cit['filename'] == 'paper.pdf'
    assert pdf_cit['page'] == 3
    
    # Webpage citation
    web_cit = [c for c in citations if c['type'] == 'webpage'][0]
    assert web_cit['title'] == 'Blog Post'
    assert web_cit['url'] == 'https://example.com/blog'


def test_pdf_generation_with_mixed_citations():
    """Test PDF report generation with mixed citation types."""
    # Create mock PDF sources
    pdf1 = ExtractedPDF(
        filename="study.pdf",
        total_pages=5,
        extracted_pages=[PDFPage(1, "Page 1 content")],
        extraction_status="completed"
    )
    pdf1.error = ""
    
    # Mixed citations
    citations = [
        {
            "type": "pdf",
            "filename": "study.pdf",
            "page": 3,
            "title": "Study"
        },
        {
            "type": "webpage",
            "title": "Article Title",
            "url": "https://example.com/article"
        },
        {
            "type": "pdf",
            "filename": "study.pdf",
            "page": 3,  # Duplicate
            "title": "Study"
        },
    ]
    
    synthesis_text = "Research findings from multiple sources."
    
    # Should not raise KeyError
    pdf_bytes = generate_pdf_report(
        synthesis_text,
        [pdf1],
        citations,
        "Test Report"
    )
    
    assert len(pdf_bytes) > 0


def test_pdf_generation_with_webpage_only_citations():
    """Test PDF generation with only webpage citations."""
    citations = [
        {
            "type": "webpage",
            "title": "News Article",
            "url": "https://news.example.com"
        },
        {
            "type": "webpage",
            "title": "Blog Post",
            "url": "https://blog.example.com"
        },
    ]
    
    synthesis_text = "Summary based on web sources."
    
    pdf_bytes = generate_pdf_report(
        synthesis_text,
        [],
        citations,
        "Web Sources Report"
    )
    
    assert len(pdf_bytes) > 0


def test_pdf_generation_with_empty_citations():
    """Test PDF generation with no citations."""
    pdf_bytes = generate_pdf_report(
        "Report with no citations.",
        [],
        [],
        "Citation-Free Report"
    )
    
    assert len(pdf_bytes) > 0


def test_malformed_citation_handling():
    """Test that malformed citations are skipped gracefully."""
    citations = [
        {"type": "pdf", "filename": "valid.pdf", "page": 1, "title": "Valid"},
        {"type": "pdf"},  # Missing required fields
        {"type": "webpage"},  # Missing title
        {"invalid": "structure"},  # Completely wrong
        {"type": "webpage", "title": "Valid Web", "url": "https://example.com"},
    ]
    
    # Should not crash
    pdf_bytes = generate_pdf_report(
        "Test with malformed citations.",
        [],
        citations,
        "Robust Test"
    )
    
    assert len(pdf_bytes) > 0


def test_citation_deduplication():
    """Test that duplicate citations are properly deduplicated."""
    citations = [
        {"type": "pdf", "filename": "doc.pdf", "page": 1, "title": "Doc"},
        {"type": "pdf", "filename": "doc.pdf", "page": 1, "title": "Doc"},  # Duplicate
        {"type": "pdf", "filename": "doc.pdf", "page": 2, "title": "Doc"},  # Different page
        {"type": "webpage", "title": "Article", "url": "https://example.com"},
        {"type": "webpage", "title": "Article", "url": "https://example.com"},  # Duplicate
    ]
    
    pdf_bytes = generate_pdf_report(
        "Test deduplication.",
        [],
        citations,
        "Dedup Test"
    )
    
    # Should have 3 unique citations in the PDF
    assert len(pdf_bytes) > 0


def test_unmatched_citation_fallback():
    """Test that LLM-generated citations that don't match sources still work."""
    manager = JobManager()
    
    collection = SourceCollection()
    metadata = SourceMetadata(
        source_id="test1",
        source_type=SourceType.PDF,
        filename="real_file.pdf",
        title="Real File",
        page_count=5,
        success=True,
    )
    collection.sources.append(ExtractedSource(
        metadata=metadata,
        text="Content",
        pages={1: "Page 1"}
    ))
    
    # LLM hallucinates a different filename
    text = "According to [Source: hallucinated_name.pdf:page 3] we find..."
    citations = manager._extract_citations(text, collection.get_all_sources())
    
    # Should still extract as citation with fallback
    assert len(citations) == 1
    assert citations[0]['type'] == 'pdf'
    assert citations[0]['filename'] == 'hallucinated_name.pdf'
    assert citations[0]['page'] == 3
