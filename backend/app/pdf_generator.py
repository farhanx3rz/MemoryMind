"""PDF report generation with proper formatting and citations.

Generates professional PDF reports from synthesis results with
typography, spacing, headings, page numbers, and source references.
"""
import io
import logging
import re
from datetime import datetime
from typing import Any

logger = logging.getLogger("memorymind.pdf_generator")


def generate_pdf_report(
    synthesis_text: str,
    extracted_pdfs: list,
    citations: list[dict],
    title: str = "Multi-Document Synthesis Report",
    sources: list = None,
) -> bytes:
    """Generate a formatted PDF report from synthesis text.

    Args:
        synthesis_text: The synthesized text from LLM
        extracted_pdfs: List of ExtractedPDF objects (legacy)
        citations: List of citation dicts
        title: Report title
        sources: List of ExtractedSource objects (preferred)

    Returns:
        PDF bytes
    """
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, PageBreak,
            Table, TableStyle, KeepTogether, KeepInFrame
        )
        from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
        from reportlab.lib import colors
    except ImportError as exc:
        logger.error("PDF generation failed: reportlab not installed")
        raise RuntimeError("PDF generation library not installed (reportlab required)") from exc

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=0.75*inch,
        leftMargin=0.75*inch,
        topMargin=0.75*inch,
        bottomMargin=0.75*inch,
        title=title,
    )

    styles = getSampleStyleSheet()
    
    # Custom styles with professional navy/indigo accents
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=22,
        textColor=colors.HexColor('#1a237e'),  # Deep indigo
        spaceAfter=16,
        spaceBefore=6,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold',
    )
    
    subtitle_style = ParagraphStyle(
        'Subtitle',
        parent=styles['Normal'],
        fontSize=10,
        textColor=colors.HexColor('#455a64'),  # Blue-grey
        spaceAfter=8,
        alignment=TA_CENTER,
        fontName='Helvetica',
    )
    
    heading1_style = ParagraphStyle(
        'CustomHeading1',
        parent=styles['Heading1'],
        fontSize=15,
        textColor=colors.HexColor('#283593'),  # Indigo
        spaceAfter=10,
        spaceBefore=18,
        fontName='Helvetica-Bold',
        keepWithNext=True,  # Prevent orphan headings
    )
    
    heading2_style = ParagraphStyle(
        'CustomHeading2',
        parent=styles['Heading2'],
        fontSize=13,
        textColor=colors.HexColor('#3949ab'),  # Lighter indigo
        spaceAfter=8,
        spaceBefore=14,
        fontName='Helvetica-Bold',
        keepWithNext=True,
    )
    
    body_style = ParagraphStyle(
        'CustomBody',
        parent=styles['BodyText'],
        fontSize=11,
        leading=17,  # Line height for readability
        alignment=TA_JUSTIFY,
        spaceAfter=12,
        textColor=colors.HexColor('#212121'),  # Dark readable text
        fontName='Helvetica',
    )
    
    source_list_style = ParagraphStyle(
        'SourceList',
        parent=styles['BodyText'],
        fontSize=10,
        leading=15,
        leftIndent=12,
        spaceAfter=6,
        textColor=colors.HexColor('#424242'),
        fontName='Helvetica',
    )
    
    citation_ref_style = ParagraphStyle(
        'CitationRef',
        parent=styles['BodyText'],
        fontSize=9,
        leading=13,
        leftIndent=16,
        spaceAfter=8,
        textColor=colors.HexColor('#616161'),
        fontName='Helvetica',
    )
    
    code_style = ParagraphStyle(
        'Code',
        parent=styles['BodyText'],
        fontSize=10,
        leading=14,
        fontName='Courier',
        textColor=colors.HexColor('#1565c0'),
        leftIndent=20,
        spaceAfter=8,
    )

    story = []

    # --- Professional Cover/Header ---
    story.append(Spacer(1, 0.4*inch))
    story.append(Paragraph(title, title_style))
    
    generation_time = datetime.now().strftime("%B %d, %Y")
    story.append(Paragraph(f"Generated {generation_time}", subtitle_style))
    story.append(Spacer(1, 0.3*inch))
    
    # Source documents list
    story.append(Paragraph("<b>Source Documents</b>", heading2_style))
    story.append(Spacer(1, 0.05*inch))
    
    # Collect all sources (from sources param or extracted_pdfs)
    source_names = []
    if sources:
        for src in sources:
            if src.metadata.success:
                if src.metadata.source_type.value == "pdf":
                    source_names.append(f"• {src.metadata.filename} ({src.metadata.page_count} pages)")
                else:
                    source_names.append(f"• {src.metadata.title}")
    elif extracted_pdfs:
        for pdf in extracted_pdfs:
            if pdf.success:
                source_names.append(f"• {pdf.filename} ({pdf.page_count} pages)")
    
    if source_names:
        for source_name in source_names:
            story.append(Paragraph(source_name, source_list_style))
    else:
        story.append(Paragraph("No source metadata available", source_list_style))
    
    story.append(Spacer(1, 0.3*inch))
    
    # Divider line
    from reportlab.platypus import HRFlowable
    story.append(HRFlowable(
        width="100%",
        thickness=1,
        color=colors.HexColor('#bdbdbd'),
        spaceAfter=0.2*inch,
        spaceBefore=0.1*inch
    ))

    # --- Parse and format synthesis text ---
    sections = parse_synthesis_sections(synthesis_text)
    
    # Check if synthesis has explicit structure
    has_headings = any(s['type'] in ('heading1', 'heading2') for s in sections)
    
    # If no headings, add an overview section
    if not has_headings and len(synthesis_text.strip()) > 100:
        story.append(Paragraph("Overview", heading1_style))
    
    for i, section in enumerate(sections):
        if section['type'] == 'heading1':
            # Keep heading with next paragraph
            heading = Paragraph(section['content'], heading1_style)
            story.append(heading)
            
        elif section['type'] == 'heading2':
            heading = Paragraph(section['content'], heading2_style)
            story.append(heading)
            
        elif section['type'] == 'paragraph':
            content = section['content']
            
            # Detect SQL/code blocks (lines starting with SELECT, INSERT, etc.)
            if re.match(r'^\s*(SELECT|INSERT|UPDATE|DELETE|CREATE|ALTER|DROP)\b', content, re.IGNORECASE):
                formatted_text = format_code_for_pdf(content)
                story.append(Paragraph(formatted_text, code_style))
            else:
                formatted_text = format_citations_for_pdf(content)
                para = Paragraph(formatted_text, body_style)
                story.append(para)
                
        elif section['type'] == 'list_item':
            formatted_text = format_citations_for_pdf(section['content'])
            story.append(Paragraph(f"• {formatted_text}", body_style))
            
        elif section['type'] == 'spacer':
            # Controlled spacing, not excessive
            story.append(Spacer(1, 0.15*inch))

    # --- Source References Section ---
    if citations:
        story.append(Spacer(1, 0.3*inch))
        story.append(HRFlowable(
            width="100%",
            thickness=1,
            color=colors.HexColor('#bdbdbd'),
            spaceAfter=0.15*inch,
            spaceBefore=0.1*inch
        ))
        story.append(Paragraph("References", heading1_style))
        story.append(Spacer(1, 0.1*inch))
        
        # Deduplicate citations
        unique_citations = {}
        for cit in citations:
            cit_type = cit.get('type', 'webpage')
            
            if cit_type == 'pdf' and 'filename' in cit and 'page' in cit:
                key = f"pdf:{cit['filename']}:{cit['page']}"
            elif cit_type == 'webpage' and 'title' in cit:
                key = f"webpage:{cit['title']}"
            else:
                continue
            
            if key not in unique_citations:
                unique_citations[key] = cit
        
        # Sort: PDFs first (by filename, then page), then webpages (by title)
        def citation_sort_key(cit):
            cit_type = cit.get('type', 'webpage')
            if cit_type == 'pdf':
                return (0, cit.get('filename', ''), cit.get('page', 0))
            else:
                return (1, cit.get('title', ''), 0)
        
        sorted_citations = sorted(unique_citations.values(), key=citation_sort_key)
        
        # Format citations compactly
        for idx, cit in enumerate(sorted_citations, 1):
            cit_type = cit.get('type', 'webpage')
            
            if cit_type == 'pdf':
                filename = cit.get('filename', 'Unknown')
                page = cit.get('page', 0)
                ref_text = f"[{idx}] <b>{filename}</b>, page {page}"
            else:
                title = cit.get('title', 'Unknown Source')
                url = cit.get('url', '')
                ref_text = f"[{idx}] <b>{title}</b>"
                if url:
                    ref_text += f"<br/>&nbsp;&nbsp;&nbsp;&nbsp;<font size=8>{url}</font>"
            
            story.append(Paragraph(ref_text, citation_ref_style))

    # --- Metadata (only if extracted_pdfs exist) ---
    if extracted_pdfs:
        story.append(Spacer(1, 0.3*inch))
        story.append(HRFlowable(
            width="100%",
            thickness=1,
            color=colors.HexColor('#bdbdbd'),
            spaceAfter=0.15*inch,
            spaceBefore=0.1*inch
        ))
        story.append(Paragraph("Extraction Details", heading1_style))
        story.append(Spacer(1, 0.1*inch))
        
        metadata_data = [
            ['Document', 'Status', 'Pages'],
        ]
        
        for pdf in extracted_pdfs:
            status = "Success" if pdf.success else f"Failed"
            pages = f"{pdf.page_count}/{pdf.total_pages}"
            metadata_data.append([pdf.filename, status, pages])
        
        metadata_table = Table(metadata_data, colWidths=[3.5*inch, 1.5*inch, 1*inch])
        metadata_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#283593')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
            ('TOPPADDING', (0, 0), (-1, 0), 10),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f5f5f5')),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e0e0e0')),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
            ('TOPPADDING', (0, 1), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 1), (-1, -1), 6),
        ]))
        
        story.append(metadata_table)

    # Build PDF
    doc.build(story)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    
    logger.info("Generated PDF report: %d bytes", len(pdf_bytes))
    return pdf_bytes


def parse_synthesis_sections(text: str) -> list[dict]:
    """Parse synthesis text into structured sections."""
    lines = text.split('\n')
    sections = []
    
    consecutive_empty = 0
    
    for line in lines:
        line_stripped = line.strip()
        
        if not line_stripped:
            consecutive_empty += 1
            # Only add spacer after first empty line, ignore multiple consecutive
            if consecutive_empty == 1:
                sections.append({'type': 'spacer', 'content': ''})
            continue
        
        consecutive_empty = 0
        
        # Detect headings
        if line_stripped.startswith('# '):
            sections.append({'type': 'heading1', 'content': line_stripped[2:].strip()})
        elif line_stripped.startswith('## '):
            sections.append({'type': 'heading2', 'content': line_stripped[3:].strip()})
        elif line_stripped.startswith('### '):
            sections.append({'type': 'heading2', 'content': line_stripped[4:].strip()})
        elif line_stripped.startswith('- ') or line_stripped.startswith('* '):
            sections.append({'type': 'list_item', 'content': line_stripped[2:].strip()})
        else:
            sections.append({'type': 'paragraph', 'content': line_stripped})
    
    return sections


def format_citations_for_pdf(text: str) -> str:
    """Convert citations to readable inline format."""
    # Escape XML special characters for reportlab
    text = text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
    
    # Convert [Source: filename:page N] to readable superscript citation
    # Handle both "page N" and "page N-M" ranges
    citation_pattern = r'\[Source:\s*([^:]+):page\s*([\d\-–]+)\]'
    text = re.sub(citation_pattern, r'<super>[<i>\1</i>, p.\2]</super>', text)
    
    # Handle webpage citations [Source: title]
    webpage_citation_pattern = r'\[Source:\s*([^\]]+)\]'
    text = re.sub(webpage_citation_pattern, r'<super>[<i>\1</i>]</super>', text)
    
    # Convert **bold** to <b>bold</b>
    text = re.sub(r'\*\*([^*]+)\*\*', r'<b>\1</b>', text)
    
    # Convert *italic* to <i>italic</i> (avoid matching already-processed citations)
    text = re.sub(r'(?<!</i>)\*([^*<>]+)\*(?!])', r'<i>\1</i>', text)
    
    return text


def format_code_for_pdf(text: str) -> str:
    """Format SQL/code for PDF with proper escaping."""
    # Escape XML special characters
    text = text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
    return f'<font name="Courier" size="10">{text}</font>'
