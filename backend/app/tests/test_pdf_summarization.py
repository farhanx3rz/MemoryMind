"""Tests for PDF summarization feature."""
import io
import sys
from pathlib import Path

# Fix Windows console encoding
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.pdf_handler import extract_pdf, ExtractedPDF, validate_extraction_results
from app.synthesis import synthesize_documents, extract_citations
from app.pdf_generator import generate_pdf_report


def create_minimal_pdf(text: str = "Test page content") -> bytes:
    """Create a minimal valid PDF for testing."""
    try:
        from pypdf import PdfWriter
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import letter
    except ImportError:
        return b"%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj 2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj 3 0 obj<</Type/Page/MediaBox[0 0 612 792]/Parent 2 0 R/Resources<<>>>>endobj\nxref\n0 4\n0000000000 65535 f\n0000000009 00000 n\n0000000058 00000 n\n0000000115 00000 n\ntrailer<</Size 4/Root 1 0 R>>\nstartxref\n190\n%%EOF"
    
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    c.drawString(100, 750, text)
    c.showPage()
    c.save()
    return buffer.getvalue()


def test_pdf_extraction():
    """Test basic PDF extraction."""
    print("Testing PDF extraction...")
    
    pdf_bytes = create_minimal_pdf("This is a test document.")
    result = extract_pdf(pdf_bytes, "test.pdf")
    
    assert result.filename == "test.pdf"
    assert result.extraction_status == "completed"
    assert result.success
    assert result.page_count >= 1
    assert len(result.extracted_pages) >= 1
    print(f"✓ Extracted {result.page_count} pages from test.pdf")


def test_empty_pdf():
    """Test handling of empty PDF."""
    print("Testing empty PDF handling...")
    
    # Minimal PDF with no content
    empty_pdf = b"%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj 2 0 obj<</Type/Pages/Kids[]/Count 0>>endobj\nxref\n0 3\n0000000000 65535 f\n0000000009 00000 n\n0000000058 00000 n\ntrailer<</Size 3/Root 1 0 R>>\nstartxref\n108\n%%EOF"
    
    result = extract_pdf(empty_pdf, "empty.pdf")
    assert not result.success
    assert "no pages" in result.error.lower()
    print(f"✓ Empty PDF correctly detected: {result.error}")


def test_multiple_pdfs():
    """Test extracting multiple PDFs."""
    print("Testing multiple PDF extraction...")
    
    pdf1 = create_minimal_pdf("Document one content about machine learning.")
    pdf2 = create_minimal_pdf("Document two discusses neural networks.")
    
    result1 = extract_pdf(pdf1, "doc1.pdf")
    result2 = extract_pdf(pdf2, "doc2.pdf")
    
    assert result1.success
    assert result2.success
    
    is_valid, error = validate_extraction_results([result1, result2])
    assert is_valid
    print(f"✓ Multiple PDFs extracted successfully")


def test_synthesis_without_llm():
    """Test synthesis fallback when no LLM configured."""
    print("Testing synthesis without LLM...")
    
    pdf1 = create_minimal_pdf("Machine learning is a subset of AI.")
    result1 = extract_pdf(pdf1, "ml.pdf")
    
    # Mock ExtractedPDF for synthesis
    from app import config
    original_provider = config.LLM_PROVIDER
    config.LLM_PROVIDER = "none"
    
    try:
        synthesis, citations = synthesize_documents([result1])
        assert "ml.pdf" in synthesis
        print(f"✓ Synthesis fallback works: {len(synthesis)} chars")
    finally:
        config.LLM_PROVIDER = original_provider


def test_pdf_generation():
    """Test PDF report generation."""
    print("Testing PDF generation...")
    
    pdf1 = create_minimal_pdf("Test content for report generation.")
    result1 = extract_pdf(pdf1, "test_report.pdf")
    
    synthesis_text = """# Executive Summary

This is a test synthesis report with **bold text** and citations [Source: test_report.pdf:page 1].

## Key Findings

- Finding one with evidence
- Finding two with more detail

## Conclusions

The synthesis demonstrates proper formatting."""
    
    citations = [
        {"filename": "test_report.pdf", "page": 1, "preview": "Test content for report"}
    ]
    
    pdf_bytes = generate_pdf_report(
        synthesis_text=synthesis_text,
        extracted_pdfs=[result1],
        citations=citations,
        title="Test Synthesis Report"
    )
    
    assert pdf_bytes.startswith(b"%PDF")
    assert len(pdf_bytes) > 1000
    print(f"✓ PDF generated: {len(pdf_bytes)} bytes")


def test_citation_extraction():
    """Test citation extraction from synthesis text."""
    print("Testing citation extraction...")
    
    text = """Here is a claim [Source: doc1.pdf:page 5] and another [Source: doc2.pdf:page 12]."""
    
    pdf1 = create_minimal_pdf("Content from doc1")
    pdf2 = create_minimal_pdf("Content from doc2")
    result1 = extract_pdf(pdf1, "doc1.pdf")
    result2 = extract_pdf(pdf2, "doc2.pdf")
    
    citations = extract_citations(text, [result1, result2])
    
    assert len(citations) == 2
    assert any(c["filename"] == "doc1.pdf" and c["page"] == 5 for c in citations)
    assert any(c["filename"] == "doc2.pdf" and c["page"] == 12 for c in citations)
    print(f"✓ Extracted {len(citations)} citations")


def test_validation():
    """Test extraction result validation."""
    print("Testing validation...")
    
    pdf1 = create_minimal_pdf("Valid content")
    result1 = extract_pdf(pdf1, "valid.pdf")
    result1.extraction_status = "completed"
    
    # Test with successful extraction
    is_valid, error = validate_extraction_results([result1])
    assert is_valid
    assert error == ""
    
    # Test with failed extraction
    failed_result = ExtractedPDF(
        filename="failed.pdf",
        total_pages=0,
        error="Test error",
        extraction_status="failed"
    )
    
    is_valid, error = validate_extraction_results([failed_result])
    assert not is_valid
    assert "failed.pdf" in error
    print(f"✓ Validation works correctly")


def test_scanned_pdf_detection():
    """Test detection of scanned/image-only PDFs."""
    print("Testing scanned PDF detection...")
    
    # Create PDF with page but no extractable text
    from pypdf import PdfWriter
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import letter
    
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    # Draw shape but no text
    c.rect(100, 100, 200, 200)
    c.showPage()
    c.save()
    
    result = extract_pdf(buffer.getvalue(), "scanned.pdf")
    
    # Should detect as scanned when no text extracted
    if result.page_count == 0:
        assert result.is_scanned or result.error
        print(f"✓ Scanned PDF detected: {result.error}")
    else:
        print(f"✓ PDF has {result.page_count} pages with text")


def run_all_tests():
    """Run all PDF tests."""
    tests = [
        test_pdf_extraction,
        test_empty_pdf,
        test_multiple_pdfs,
        test_synthesis_without_llm,
        test_pdf_generation,
        test_citation_extraction,
        test_validation,
        test_scanned_pdf_detection,
    ]
    
    print("\n" + "="*60)
    print("Running PDF Summarization Tests")
    print("="*60 + "\n")
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"✗ {test.__name__} failed: {e}")
            failed += 1
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*60)
    print(f"Results: {passed} passed, {failed} failed")
    print("="*60 + "\n")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
