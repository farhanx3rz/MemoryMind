# Citation Schema Fix - Summary

## Problem
The backend had a schema mismatch between citation extraction and PDF generation:
- `_extract_citations()` returned: `{"type": "pdf", "source": source_name, "page": page_num}`
- `generate_pdf_report()` expected: `cit['filename']` and `cit['page']`
- This caused `KeyError: 'filename'` when generating PDFs with citations

## Solution

### 1. Established Consistent Citation Schema

**PDF Citations:**
```python
{
    "type": "pdf",
    "filename": str,  # Actual PDF filename from source metadata
    "page": int,
    "title": str
}
```

**Webpage Citations:**
```python
{
    "type": "webpage",
    "title": str,
    "url": str
}
```

### 2. Files Changed

**backend/app/summarization_jobs.py** (lines 414-489)
- Rewrote `_extract_citations()` to:
  - Return consistent schema with `filename` field (not `source`)
  - Validate citations against actual source filenames
  - Support both PDF and webpage citations
  - Handle unmatched LLM-generated citations gracefully
  - Map citation references to actual source metadata

**backend/app/pdf_generator.py** (lines 146-194)
- Updated citation rendering to:
  - Handle both PDF and webpage citation types
  - Deduplicate citations safely with type-aware keys
  - Sort PDFs first (by filename, page), then webpages (by title)
  - Skip malformed citations without crashing
  - Render proper format for each citation type

### 3. Key Features

✓ **Source validation**: Citations matched against actual PDF filenames from source metadata
✓ **Fallback handling**: Unmatched LLM citations still work with fallback filename
✓ **Type safety**: Separate handling for PDF vs webpage citations
✓ **Deduplication**: Proper deduplication with type-aware keys
✓ **Robustness**: Skips malformed citations, never fabricates data
✓ **Mixed sources**: Supports documents with both PDF and webpage citations

### 4. Tests Added

Created `backend/app/tests/test_citation_schema.py` with 11 comprehensive tests:

1. ✓ `test_pdf_citation_extraction` - PDF with filename and page
2. ✓ `test_citation_with_source_field_compatibility` - Old field normalized
3. ✓ `test_webpage_citation_extraction` - Webpage with title
4. ✓ `test_empty_citations` - Empty list handling
5. ✓ `test_mixed_pdf_and_webpage_citations` - Mixed document types
6. ✓ `test_pdf_generation_with_mixed_citations` - PDF generation
7. ✓ `test_pdf_generation_with_webpage_only_citations` - Web-only
8. ✓ `test_pdf_generation_with_empty_citations` - No citations
9. ✓ `test_malformed_citation_handling` - Graceful degradation
10. ✓ `test_citation_deduplication` - Proper deduplication
11. ✓ `test_unmatched_citation_fallback` - LLM hallucination handling

### 5. Test Results

**All tests passing:**
- 58 total tests (47 original + 11 new citation tests)
- 0 failures
- Backend startup successful
- All routes registered correctly

### 6. What Was Fixed

**Before:**
```python
# Extraction returned:
{"type": "pdf", "source": "doc.pdf", "page": 3}

# Generator expected:
key = f"{cit['filename']}:{cit['page']}"  # KeyError!
```

**After:**
```python
# Extraction returns:
{"type": "pdf", "filename": "doc.pdf", "page": 3, "title": "Document"}

# Generator uses:
if cit_type == 'pdf' and 'filename' in cit and 'page' in cit:
    key = f"pdf:{cit['filename']}:{cit['page']}"  # Works!
```

### 7. Verification Steps

To test end-to-end:
1. Start backend: `cd backend && uvicorn app.main:app --reload`
2. Load extension in Chrome
3. Create a summarization job with PDFs
4. Wait for "completed" status
5. Click "View PDF" - should open without errors
6. Check citations section shows proper references

### 8. Additional Benefits

- **No data fabrication**: Never creates fake filenames or page numbers
- **Consistent API**: All citation objects follow same schema throughout
- **Better error handling**: Malformed citations logged but don't crash
- **Future-proof**: Easy to add new citation types (books, videos, etc.)
- **Validated mappings**: Citations reference actual source files

## Status: COMPLETE ✓

The citation schema mismatch is fully resolved. All tests pass, and the backend starts without errors.
