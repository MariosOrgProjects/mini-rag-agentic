"""Tests for logging utilities."""

from mini_rag.configuration import Settings
from mini_rag.logging_utils import QueryTrace, StructuredLogger
from mini_rag.models import DocumentChunk, RetrievalResult


class TestQueryTrace:
    """Tests for QueryTrace class."""

    def test_trace_initialization(self) -> None:
        """Test trace initialization."""
        trace = QueryTrace("What is X?")
        assert trace.question == "What is X?"
        assert len(trace.trace_id) == 36  # UUID format

    def test_trace_full_workflow(self) -> None:
        """Test complete trace workflow."""
        trace = QueryTrace("Test question")
        trace.start()
        trace.set_plan(["plan", "retrieve", "draft", "cite"])

        chunk = DocumentChunk(
            content="Test content",
            chunk_id=1,
            file_name="test.pdf",
            file_path="/path/test.pdf",
            page_number=2,
        )
        result = RetrievalResult(chunk=chunk, score=0.85, rank=1)
        trace.add_retrieval_results([result])
        trace.set_answer("The answer is 42.", ["[source1]"])
        trace.finalize()

        data = trace.to_dict()
        assert data["question"] == "Test question"
        assert data["plan"] == ["plan", "retrieve", "draft", "cite"]
        assert len(data["retrieval"]) == 1


class TestStructuredLogger:
    """Tests for StructuredLogger class."""

    def test_logger_initialization(self, settings: Settings) -> None:
        """Test logger initialization."""
        logger = StructuredLogger(settings)
        assert logger.log_dir == settings.log_dir
        assert logger.log_dir.exists()

    def test_logger_log_trace(self, settings: Settings) -> None:
        """Test logging a trace."""
        logger = StructuredLogger(settings)

        trace = QueryTrace("Test question")
        trace.set_plan(["plan", "retrieve"])
        trace.finalize()
        logger.log_trace(trace)

        assert logger.trace_file.exists()
        with open(logger.trace_file) as f:
            content = f.read()
            assert "Test question" in content

    def test_logger_get_recent_traces(self, settings: Settings) -> None:
        """Test getting recent traces."""
        logger = StructuredLogger(settings)

        for i in range(3):
            trace = QueryTrace(f"Question {i}")
            trace.finalize()
            logger.log_trace(trace)

        traces = logger.get_recent_traces(2)
        assert len(traces) == 2
