"""Tests for document processor module."""

from pathlib import Path

import pytest

from mini_rag.document_processor import DocumentProcessor
from mini_rag.exceptions import MiniRAGError


class TestDocumentProcessor:
    """Tests for DocumentProcessor class."""

    def test_processor_initialization(self) -> None:
        """Test processor initialization."""
        processor = DocumentProcessor()
        assert processor.chunk_size == 200
        assert processor.chunk_overlap == 20

    def test_load_pdf_success(self, sample_pdf_path: Path) -> None:
        """Test successful PDF loading."""
        processor = DocumentProcessor()
        doc = processor.load_pdf(sample_pdf_path)

        assert doc.file_name == "test_document.pdf"
        assert doc.total_pages >= 1

    def test_load_pdf_not_found(self, temp_dir: Path) -> None:
        """Test loading non-existent PDF raises error."""
        processor = DocumentProcessor()

        with pytest.raises(MiniRAGError, match="File not found"):
            processor.load_pdf(temp_dir / "nonexistent.pdf")

    def test_load_pdf_wrong_extension(self, temp_dir: Path) -> None:
        """Test loading wrong file type raises error."""
        processor = DocumentProcessor()
        txt_file = temp_dir / "test.txt"
        txt_file.write_text("Test content")

        with pytest.raises(MiniRAGError, match="Unsupported file format"):
            processor.load_pdf(txt_file)

    def test_load_directory_success(
        self, temp_dir: Path, sample_pdf_content: bytes
    ) -> None:
        """Test loading directory with PDFs."""
        for i in range(2):
            (temp_dir / f"doc{i}.pdf").write_bytes(sample_pdf_content)

        processor = DocumentProcessor()
        docs = processor.load_directory(temp_dir)

        assert len(docs) == 2

    def test_load_directory_empty(self, temp_dir: Path) -> None:
        """Test loading empty directory raises error."""
        processor = DocumentProcessor()

        with pytest.raises(MiniRAGError, match="No PDF files found"):
            processor.load_directory(temp_dir)
