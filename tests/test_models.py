"""Tests for document models."""

import pytest

from mini_rag.models import Document, DocumentChunk, RetrievalResult


class TestDocumentChunk:
    """Tests for DocumentChunk class."""

    def test_chunk_creation(self) -> None:
        """Test basic chunk creation."""
        chunk = DocumentChunk(
            content="Test content",
            chunk_id=0,
            file_name="test.pdf",
            file_path="/path/test.pdf",
            page_number=1,
        )
        assert chunk.content == "Test content"
        assert chunk.chunk_id == 0
        assert chunk.file_name == "test.pdf"

    def test_chunk_validation_empty_content(self) -> None:
        """Test that empty content raises error."""
        with pytest.raises(ValueError, match="cannot be empty"):
            DocumentChunk(
                content="   ",
                chunk_id=0,
                file_name="test.pdf",
                file_path="/path/test.pdf",
            )

    def test_chunk_citation(self) -> None:
        """Test citation generation."""
        chunk = DocumentChunk(
            content="Test content",
            chunk_id=5,
            file_name="document.pdf",
            file_path="/path/document.pdf",
            page_number=3,
        )
        citation = chunk.citation
        assert "document.pdf" in citation
        assert "Page 3" in citation
        assert "Chunk 5" in citation

    def test_chunk_round_trip(self) -> None:
        """Test to_dict and from_dict round trip."""
        original = DocumentChunk(
            content="Test content",
            chunk_id=5,
            file_name="test.pdf",
            file_path="/path/test.pdf",
            page_number=2,
        )
        data = original.to_dict()
        restored = DocumentChunk.from_dict(data)

        assert restored.content == original.content
        assert restored.chunk_id == original.chunk_id
        assert restored.file_name == original.file_name


class TestDocument:
    """Tests for Document class."""

    def test_document_creation(self) -> None:
        """Test basic document creation."""
        doc = Document(
            file_path="/path/test.pdf",
            file_name="test.pdf",
            total_pages=5,
        )
        assert doc.file_name == "test.pdf"
        assert doc.total_pages == 5
        assert doc.chunk_count == 0

    def test_document_with_chunks(self) -> None:
        """Test document with chunks."""
        chunks = [
            DocumentChunk(
                content=f"Chunk {i}",
                chunk_id=i,
                file_name="test.pdf",
                file_path="/path/test.pdf",
            )
            for i in range(3)
        ]
        doc = Document(
            file_path="/path/test.pdf",
            file_name="test.pdf",
            chunks=chunks,
        )
        assert doc.chunk_count == 3


class TestRetrievalResult:
    """Tests for RetrievalResult class."""

    def test_retrieval_result_creation(self) -> None:
        """Test basic retrieval result creation."""
        chunk = DocumentChunk(
            content="Test content",
            chunk_id=0,
            file_name="test.pdf",
            file_path="/path/test.pdf",
        )
        result = RetrievalResult(chunk=chunk, score=0.85, rank=1)

        assert result.chunk == chunk
        assert result.score == 0.85
        assert result.rank == 1
