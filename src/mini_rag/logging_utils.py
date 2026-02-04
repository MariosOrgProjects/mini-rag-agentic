"""Structured logging for the RAG system."""

import json
import logging
import sys
import time
import uuid
from typing import Any

from mini_rag.configuration import Settings
from mini_rag.models import RetrievalResult


class QueryTrace:
    """Trace object for tracking a single query execution."""

    def __init__(self, question: str):
        """Initialize query trace.

        Args:
            question: The user's question.
        """
        self.trace_id = str(uuid.uuid4())
        self.question = question
        self.plan: list[str] = []
        self.retrieval: list[dict[str, Any]] = []
        self.draft_tokens: int = 0
        self.latency_ms: dict[str, float] = {}
        self.errors: list[str] = []
        self._start_time: float | None = None
        self._step_start: float | None = None
        self._current_step: str | None = None
        self.answer: str = ""
        self.citations: list[str] = []

    def start(self) -> None:
        """Start the trace timer."""
        self._start_time = time.perf_counter()

    def start_step(self, step_name: str) -> None:
        """Start timing a step.

        Args:
            step_name: Name of the step.
        """
        self._step_start = time.perf_counter()
        self._current_step = step_name

    def end_step(self) -> None:
        """End timing the current step."""
        if self._step_start is not None and self._current_step is not None:
            elapsed = (time.perf_counter() - self._step_start) * 1000
            self.latency_ms[self._current_step] = round(elapsed, 2)
            self._step_start = None
            self._current_step = None

    def finalize(self) -> None:
        """Finalize the trace with total time."""
        if self._start_time is not None:
            total = (time.perf_counter() - self._start_time) * 1000
            self.latency_ms["total"] = round(total, 2)

    def set_plan(self, steps: list[str]) -> None:
        """Set the execution plan.

        Args:
            steps: List of planned steps.
        """
        self.plan = steps

    def add_retrieval_results(self, results: list[RetrievalResult]) -> None:
        """Add retrieval results to trace.

        Args:
            results: List of retrieval results.
        """
        self.retrieval = [r.to_dict() for r in results]

    def add_error(self, error: str) -> None:
        """Add an error to the trace.

        Args:
            error: Error message.
        """
        self.errors.append(error)

    def set_answer(self, answer: str, citations: list[str]) -> None:
        """Set the final answer.

        Args:
            answer: Generated answer.
            citations: List of citations used.
        """
        self.answer = answer
        self.citations = citations
        # Estimate token count (rough approximation)
        self.draft_tokens = len(answer.split())

    def to_dict(self) -> dict[str, Any]:
        """Convert trace to dictionary.

        Returns:
            Dictionary representation of the trace.
        """
        # Only include retrieve, draft, total in latency_ms per assignment spec
        filtered_latency = {
            k: v
            for k, v in self.latency_ms.items()
            if k in ("retrieve", "draft", "total")
        }
        return {
            "trace_id": self.trace_id,
            "question": self.question,
            "plan": self.plan,
            "retrieval": self.retrieval,
            "draft_tokens": self.draft_tokens,
            "latency_ms": filtered_latency,
            "errors": self.errors,
        }

    def to_json(self) -> str:
        """Convert trace to JSON string.

        Returns:
            JSON representation.
        """
        return json.dumps(self.to_dict(), indent=2)


class StructuredLogger:
    """Structured logger for query traces."""

    def __init__(self, settings: Settings | None = None):
        """Initialize the logger.

        Args:
            settings: Application settings.
        """
        self.settings = settings or Settings()
        self.log_dir = self.settings.log_dir
        self._setup_logging()

    def _setup_logging(self) -> None:
        """Set up the logging configuration."""
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Configure root logger
        self.logger = logging.getLogger("mini_rag")
        self.logger.setLevel(getattr(logging, self.settings.log_level.upper()))

        # Clear existing handlers
        self.logger.handlers.clear()

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_format = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        console_handler.setFormatter(console_format)
        self.logger.addHandler(console_handler)

        # File handler for JSON traces
        log_file = self.log_dir / "queries.jsonl"
        self.trace_file = log_file

    def log_trace(self, trace: QueryTrace) -> None:
        """Log a query trace.

        Args:
            trace: The query trace to log.
        """
        trace_json = trace.to_dict()

        # Write to JSONL file
        with open(self.trace_file, "a") as f:
            f.write(json.dumps(trace_json) + "\n")

        # Also log summary to console
        self.logger.info(
            f"Query completed - trace_id={trace.trace_id}, "
            f"latency={trace.latency_ms.get('total', 0):.0f}ms, "
            f"chunks_retrieved={len(trace.retrieval)}"
        )

    def log_info(self, message: str) -> None:
        """Log an info message.

        Args:
            message: Message to log.
        """
        self.logger.info(message)

    def log_error(self, message: str, exc: Exception | None = None) -> None:
        """Log an error message.

        Args:
            message: Error message.
            exc: Optional exception.
        """
        if exc:
            self.logger.error(f"{message}: {exc}")
        else:
            self.logger.error(message)

    def log_debug(self, message: str) -> None:
        """Log a debug message.

        Args:
            message: Message to log.
        """
        self.logger.debug(message)

    def get_recent_traces(self, n: int = 10) -> list[dict[str, Any]]:
        """Get the most recent query traces.

        Args:
            n: Number of traces to retrieve.

        Returns:
            List of trace dictionaries.
        """
        if not self.trace_file.exists():
            return []

        traces = []
        with open(self.trace_file) as f:
            for line in f:
                if line.strip():
                    traces.append(json.loads(line))

        return traces[-n:]
