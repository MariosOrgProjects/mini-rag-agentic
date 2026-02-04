"""Tests for custom exceptions."""

from mini_rag.exceptions import MiniRAGError


class TestExceptions:
    """Tests for exception classes."""

    def test_mini_rag_error_base(self) -> None:
        """Test base exception without context."""
        error = MiniRAGError("Test error")
        assert str(error) == "Test error"
        assert error.context is None

    def test_mini_rag_error_with_context(self) -> None:
        """Test exception with context."""
        error = MiniRAGError("Failed to process", "file.pdf")
        assert "Failed to process" in str(error)
        assert "file.pdf" in str(error)
        assert error.context == "file.pdf"
