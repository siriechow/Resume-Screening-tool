# file_parsers.py

"""
Utilities for reading and extracting text from different resume file formats.
Supports: PDF, DOCX, and TXT.
"""

import io
from typing import Any

from pdfminer.high_level import extract_text
from docx import Document


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract text from a PDF file given its bytes."""
    with io.BytesIO(file_bytes) as f:
        return extract_text(f)


def extract_text_from_docx(file_bytes: bytes) -> str:
    """Extract text from a DOCX file given its bytes."""
    with io.BytesIO(file_bytes) as f:
        doc = Document(f)
    return "\n".join(p.text for p in doc.paragraphs)


def extract_text_from_txt(file_bytes: bytes) -> str:
    """Extract text from a TXT file given its bytes."""
    return file_bytes.decode("utf-8", errors="ignore")


def extract_text_from_file(uploaded_file: Any) -> str:
    """
    Universal function for Streamlit's UploadedFile object.
    Detects file type based on extension and extracts text.
    """
    file_bytes = uploaded_file.read()
    name = uploaded_file.name.lower()

    if name.endswith(".pdf"):
        return extract_text_from_pdf(file_bytes)
    elif name.endswith(".docx"):
        return extract_text_from_docx(file_bytes)
    elif name.endswith(".txt"):
        return extract_text_from_txt(file_bytes)
    else:
        raise ValueError(f"Unsupported file type for: {uploaded_file.name}")
