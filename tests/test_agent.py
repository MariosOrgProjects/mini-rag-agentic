"""Tests for the Agent module."""

from pathlib import Path

import pytest
from pytest_mock import MockerFixture

from mini_rag.agent import Agent, AgentResponse
from mini_rag.configuration import Settings
from mini_rag.exceptions import MiniRAGError


class TestAgentResponse:
    """Tests for AgentResponse class."""

    def test_response_creation(self) -> None:
        """Test response creation."""
        response = AgentResponse(
            answer="Test answer",
            citations=["[source1]"],
            sources=[{"file": "test.pdf", "chunk_id": 0, "score": 0.85}],
            trace_id="123-456",
            success=True,
        )

        assert response.answer == "Test answer"
        assert response.success
        assert len(response.citations) == 1

    def test_response_format_success(self) -> None:
        """Test formatting successful response."""
        response = AgentResponse(
            answer="The answer is 42.",
            citations=["[doc.pdf, Chunk 1]"],
            sources=[{"file": "doc.pdf", "chunk_id": 1, "score": 0.92}],
            trace_id="123",
            success=True,
        )

        formatted = response.format_answer()

        assert "The answer is 42." in formatted
        assert "Sources:" in formatted
        assert "doc.pdf" in formatted

    def test_response_format_error(self) -> None:
        """Test formatting error response."""
        response = AgentResponse(
            answer="",
            citations=[],
            sources=[],
            trace_id="123",
            success=False,
            error="Something went wrong",
        )

        formatted = response.format_answer()

        assert "Error:" in formatted
        assert "Something went wrong" in formatted


class TestAgent:
    """Tests for Agent class."""

    def test_agent_initialization(self, settings: Settings) -> None:
        """Test agent initialization."""
        agent = Agent(settings)

        assert agent.settings == settings
        assert not agent._is_initialized

    def test_agent_query_not_initialized(self, settings: Settings) -> None:
        """Test querying before initialization raises error."""
        agent = Agent(settings)

        with pytest.raises(MiniRAGError, match="not initialized"):
            agent.query("Test question")

    def test_agent_initialize_with_pdf(
        self, settings: Settings, sample_pdf_path: Path
    ) -> None:
        """Test initialization and query with a PDF file."""
        agent = Agent(settings)
        agent.initialize(corpus_path=sample_pdf_path, force_reindex=True)

        assert agent._is_initialized

        # Also test query works
        response = agent.query("What is in this document?")
        assert response.success
        assert len(response.trace_id) == 36  # UUID format

    def test_agent_initialize_no_documents(
        self, settings: Settings, temp_dir: Path
    ) -> None:
        """Test initialization with empty directory raises error."""
        agent = Agent(settings)

        with pytest.raises(MiniRAGError):
            agent.initialize(corpus_path=temp_dir, force_reindex=True)

    def test_agent_get_stats(self, settings: Settings, sample_pdf_path: Path) -> None:
        """Test getting agent statistics."""
        agent = Agent(settings)

        # Before init
        stats = agent.get_stats()
        assert not stats["initialized"]
        assert stats["embedding_backend"] is None

        # After init
        agent.initialize(corpus_path=sample_pdf_path, force_reindex=True)
        stats = agent.get_stats()
        assert stats["initialized"]
        assert stats["document_count"] >= 1
        assert "embedding_backend" in stats


class TestAgentErrorHandling:
    """Tests for Agent error handling."""

    def test_agent_handles_retrieval_errors(
        self, settings: Settings, sample_pdf_path: Path, mocker: MockerFixture
    ) -> None:
        """Test handling retrieval errors gracefully."""
        agent = Agent(settings)
        agent.initialize(corpus_path=sample_pdf_path, force_reindex=True)

        mocker.patch.object(
            agent.retriever, "retrieve", side_effect=MiniRAGError("Test error")
        )

        response = agent.query("Test question")

        assert not response.success
        assert "Test error" in response.error

    def test_agent_empty_query_result(
        self, settings: Settings, sample_pdf_path: Path, mocker: MockerFixture
    ) -> None:
        """Test handling empty retrieval results."""
        agent = Agent(settings)
        agent.initialize(corpus_path=sample_pdf_path, force_reindex=True)

        mocker.patch.object(agent.retriever, "retrieve", return_value=[])

        response = agent.query("Completely unrelated query")

        assert response.success
        assert "couldn't find" in response.answer.lower() or len(response.sources) == 0
