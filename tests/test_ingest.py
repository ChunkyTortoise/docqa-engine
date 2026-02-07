"""Tests for document ingestion module."""

from __future__ import annotations

from pathlib import Path

import pytest

from docqa_engine.ingest import (
    DocumentChunk,
    IngestResult,
    _chunk_text,
    ingest_csv,
    ingest_file,
    ingest_txt,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_text() -> str:
    return (
        "The real estate market in Rancho Cucamonga has seen steady growth. "
        "Home prices rose 12% year-over-year. Inventory remains tight with "
        "only 2.3 months of supply. Buyers face stiff competition for well-priced "
        "listings. New construction starts have increased in the Inland Empire."
    )


@pytest.fixture()
def long_text() -> str:
    """~5 000 chars of repeating sentences for multi-chunk tests."""
    return ("This is sentence number one. This is sentence number two. ") * 100


# ---------------------------------------------------------------------------
# _chunk_text
# ---------------------------------------------------------------------------


class TestChunkText:
    """Tests for the low-level _chunk_text helper."""

    def test_basic_chunking_returns_tuples(self, sample_text: str) -> None:
        chunks = _chunk_text(sample_text, chunk_size=100, overlap=20)
        assert len(chunks) >= 2
        for text, offset in chunks:
            assert isinstance(text, str)
            assert isinstance(offset, int)

    def test_custom_chunk_size_and_overlap(self, long_text: str) -> None:
        small = _chunk_text(long_text, chunk_size=200, overlap=30)
        large = _chunk_text(long_text, chunk_size=1000, overlap=100)
        assert len(small) > len(large)

    def test_empty_text_returns_empty_list(self) -> None:
        assert _chunk_text("") == []

    def test_whitespace_only_returns_empty_list(self) -> None:
        assert _chunk_text("   \n\t\n  ") == []

    def test_short_text_single_chunk(self) -> None:
        chunks = _chunk_text("Hello world.", chunk_size=500)
        assert len(chunks) == 1
        assert chunks[0][0] == "Hello world."
        assert chunks[0][1] == 0

    def test_sentence_boundary_splitting(self) -> None:
        text = "First sentence. Second sentence. Third sentence is longer padding here."
        chunks = _chunk_text(text, chunk_size=40, overlap=5)
        assert len(chunks) >= 2
        # The first chunk should ideally end at a sentence boundary
        first_text = chunks[0][0]
        assert first_text.endswith(".") or first_text.endswith(". ")

    def test_offsets_are_non_decreasing(self, long_text: str) -> None:
        chunks = _chunk_text(long_text, chunk_size=200, overlap=30)
        offsets = [offset for _, offset in chunks]
        for i in range(1, len(offsets)):
            assert offsets[i] >= offsets[i - 1]


# ---------------------------------------------------------------------------
# ingest_txt
# ---------------------------------------------------------------------------


class TestIngestTxt:
    """Tests for plain-text ingestion."""

    def test_returns_ingest_result(self, sample_text: str) -> None:
        result = ingest_txt(sample_text)
        assert isinstance(result, IngestResult)
        assert result.document_id
        assert result.filename == "document.txt"
        assert result.total_chars == len(sample_text)

    def test_custom_chunk_size(self, long_text: str) -> None:
        result = ingest_txt(long_text, chunk_size=200, overlap=20)
        assert len(result.chunks) > 5

    def test_empty_document_has_no_chunks(self) -> None:
        result = ingest_txt("")
        assert len(result.chunks) == 0
        assert result.total_chars == 0

    def test_chunk_metadata_populated(self, sample_text: str) -> None:
        result = ingest_txt(sample_text, filename="market_report.txt")
        chunk = result.chunks[0]
        assert isinstance(chunk, DocumentChunk)
        assert chunk.document_id == result.document_id
        assert chunk.metadata["source"] == "market_report.txt"
        assert chunk.metadata["type"] == "txt"
        assert chunk.chunk_id  # non-empty UUID


# ---------------------------------------------------------------------------
# ingest_file  (auto-detect by extension)
# ---------------------------------------------------------------------------


class TestIngestFile:
    """Tests for the auto-detecting ingest_file entry point."""

    def test_txt_extension(self, tmp_path: Path) -> None:
        txt_file = tmp_path / "notes.txt"
        txt_file.write_text("Some plain text content for testing.", encoding="utf-8")
        result = ingest_file(txt_file)
        assert isinstance(result, IngestResult)
        assert result.filename == "notes.txt"
        assert len(result.chunks) >= 1

    def test_md_extension(self, tmp_path: Path) -> None:
        md_file = tmp_path / "readme.md"
        md_file.write_text("# Heading\n\nParagraph of markdown content.", encoding="utf-8")
        result = ingest_file(md_file)
        assert result.filename == "readme.md"
        assert len(result.chunks) >= 1

    def test_unsupported_extension_raises(self, tmp_path: Path) -> None:
        bad_file = tmp_path / "data.xyz"
        bad_file.write_bytes(b"binary blob")
        with pytest.raises(ValueError, match="Unsupported file type"):
            ingest_file(bad_file)


# ---------------------------------------------------------------------------
# ingest_csv
# ---------------------------------------------------------------------------


class TestIngestCsv:
    """Tests for CSV ingestion (each row becomes a chunk)."""

    def test_basic_csv(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "leads.csv"
        csv_file.write_text("name,score,status\nAlice,85,Hot\nBob,42,Warm\nCarl,15,Cold\n")
        result = ingest_csv(csv_file)
        assert isinstance(result, IngestResult)
        assert len(result.chunks) == 3
        assert "Alice" in result.chunks[0].content
        assert result.chunks[0].metadata["type"] == "csv"
        assert result.chunks[0].metadata["row"] == 1
        assert result.chunks[2].metadata["row"] == 3

    def test_csv_via_ingest_file(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("col_a,col_b\nx,1\ny,2\n")
        result = ingest_file(csv_file)
        assert result.chunks[0].metadata["type"] == "csv"
