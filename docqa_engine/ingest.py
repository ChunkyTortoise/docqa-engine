"""Document Ingestion: PDF, DOCX, TXT, CSV with configurable chunking."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4


@dataclass
class DocumentChunk:
    chunk_id: str
    document_id: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    page_number: int | None = None
    char_offset: int = 0


@dataclass
class IngestResult:
    document_id: str
    filename: str
    chunks: list[DocumentChunk]
    total_chars: int
    page_count: int | None = None


def _chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[tuple[str, int]]:
    """Split text into overlapping chunks. Returns (chunk_text, char_offset) pairs."""
    if not text.strip():
        return []

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size

        # Try to break at sentence boundary
        if end < len(text):
            for sep in [". ", "\n\n", "\n", " "]:
                last_sep = text[start:end].rfind(sep)
                if last_sep > chunk_size // 2:
                    end = start + last_sep + len(sep)
                    break

        chunk = text[start:end].strip()
        if chunk:
            chunks.append((chunk, start))

        start = end - overlap if end < len(text) else len(text)

    return chunks


def ingest_txt(content: str, filename: str = "document.txt", **kwargs) -> IngestResult:
    """Ingest plain text content."""
    doc_id = str(uuid4())
    chunk_size = kwargs.get("chunk_size", 500)
    overlap = kwargs.get("overlap", 50)

    raw_chunks = _chunk_text(content, chunk_size=chunk_size, overlap=overlap)
    chunks = [
        DocumentChunk(
            chunk_id=str(uuid4()),
            document_id=doc_id,
            content=text,
            metadata={"source": filename, "type": "txt"},
            char_offset=offset,
        )
        for text, offset in raw_chunks
    ]

    return IngestResult(document_id=doc_id, filename=filename, chunks=chunks, total_chars=len(content))


def ingest_pdf(file_path: str | Path, **kwargs) -> IngestResult:
    """Ingest PDF file using PyPDF2."""
    try:
        from PyPDF2 import PdfReader
    except ImportError:
        raise ImportError("PyPDF2 required for PDF ingestion: pip install PyPDF2")

    reader = PdfReader(str(file_path))
    doc_id = str(uuid4())
    filename = Path(file_path).name
    chunk_size = kwargs.get("chunk_size", 500)
    overlap = kwargs.get("overlap", 50)

    all_chunks: list[DocumentChunk] = []

    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        raw_chunks = _chunk_text(text, chunk_size=chunk_size, overlap=overlap)
        for chunk_text, offset in raw_chunks:
            all_chunks.append(
                DocumentChunk(
                    chunk_id=str(uuid4()),
                    document_id=doc_id,
                    content=chunk_text,
                    metadata={"source": filename, "type": "pdf", "page": page_num},
                    page_number=page_num,
                    char_offset=offset,
                )
            )

    total_chars = sum(len(page.extract_text() or "") for page in reader.pages)
    return IngestResult(
        document_id=doc_id,
        filename=filename,
        chunks=all_chunks,
        total_chars=total_chars,
        page_count=len(reader.pages),
    )


def ingest_docx(file_path: str | Path, **kwargs) -> IngestResult:
    """Ingest DOCX file using python-docx."""
    try:
        from docx import Document
    except ImportError:
        raise ImportError("python-docx required for DOCX ingestion: pip install python-docx")

    doc = Document(str(file_path))
    doc_id = str(uuid4())
    filename = Path(file_path).name
    chunk_size = kwargs.get("chunk_size", 500)
    overlap = kwargs.get("overlap", 50)

    full_text = "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())
    raw_chunks = _chunk_text(full_text, chunk_size=chunk_size, overlap=overlap)

    chunks = [
        DocumentChunk(
            chunk_id=str(uuid4()),
            document_id=doc_id,
            content=text,
            metadata={"source": filename, "type": "docx"},
            char_offset=offset,
        )
        for text, offset in raw_chunks
    ]

    return IngestResult(document_id=doc_id, filename=filename, chunks=chunks, total_chars=len(full_text))


def ingest_csv(file_path: str | Path, **kwargs) -> IngestResult:
    """Ingest CSV file — each row becomes a chunk."""
    doc_id = str(uuid4())
    filename = Path(file_path).name

    with open(file_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    chunks = []
    for i, row in enumerate(rows):
        content = " | ".join(f"{k}: {v}" for k, v in row.items() if v)
        chunks.append(
            DocumentChunk(
                chunk_id=str(uuid4()),
                document_id=doc_id,
                content=content,
                metadata={"source": filename, "type": "csv", "row": i + 1},
            )
        )

    total_chars = sum(len(c.content) for c in chunks)
    return IngestResult(document_id=doc_id, filename=filename, chunks=chunks, total_chars=total_chars)


def ingest_file(file_path: str | Path, **kwargs) -> IngestResult:
    """Auto-detect file type and ingest."""
    path = Path(file_path)
    ext = path.suffix.lower()

    if ext == ".pdf":
        return ingest_pdf(path, **kwargs)
    elif ext == ".docx":
        return ingest_docx(path, **kwargs)
    elif ext == ".csv":
        return ingest_csv(path, **kwargs)
    elif ext in (".txt", ".md", ".rst"):
        content = path.read_text(encoding="utf-8")
        return ingest_txt(content, filename=path.name, **kwargs)
    else:
        raise ValueError(f"Unsupported file type: {ext}")
