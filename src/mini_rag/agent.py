"""Agentic loop for RAG Q&A with plan-retrieve-draft-cite workflow."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mini_rag.configuration import Settings
from mini_rag.document_processor import DocumentProcessor
from mini_rag.exceptions import MiniRAGError
from mini_rag.llm import LLMService, SimpleLLM
from mini_rag.logging_utils import QueryTrace, StructuredLogger
from mini_rag.models import Document, RetrievalResult
from mini_rag.retriever import Retriever


@dataclass
class AgentResponse:
    """Response from the agent."""

    answer: str
    citations: list[str]
    sources: list[dict[str, Any]]
    trace_id: str
    success: bool
    error: str | None = None

    def format_answer(self) -> str:
        """Format the answer with citations for display.

        Returns:
            Formatted answer string.
        """
        if not self.success:
            return f"Error: {self.error}"

        output = [self.answer, "", "---", "Sources:"]
        for source in self.sources:
            output.append(
                f"  - {source['file']} (Chunk {source['chunk_id']}, "
                f"Score: {source['score']:.2f})"
            )
        return "\n".join(output)


class Agent:
    """Agentic RAG system with plan-retrieve-draft-cite workflow."""

    PLAN_STEPS = ["plan", "retrieve", "draft", "cite"]

    def __init__(self, settings: Settings | None = None):
        """Initialize the agent."""

        self.settings = settings
        self.doc_processor = DocumentProcessor(self.settings)
        self.retriever = Retriever(self.settings)
        self.logger = StructuredLogger(self.settings)

        # Initialize LLM (with fallback)
        self._llm: LLMService | SimpleLLM | None = None
        self._llm_available = False
        self._documents: list[Document] = []
        self._is_initialized = False

    def _init_llm(self) -> None:
        """Initialize LLM service with fallback."""
        llm_service = LLMService(self.settings)
        if llm_service.is_available():
            self._llm = llm_service
            self._llm_available = True
            self.logger.log_info(f"Using Ollama LLM: {self.settings.llm_model}")
        else:
            self._llm = SimpleLLM()
            self._llm_available = False
            self.logger.log_info("Ollama LLM unavailable, using simple extraction")

    def initialize(
        self, corpus_path: Path | None = None, force_reindex: bool = False
    ) -> None:
        """Initialize the agent by loading and indexing documents."""

        # Try to load existing index
        if not force_reindex and self.retriever.load_index():
            self.logger.log_info(
                f"Loaded existing index with {self.retriever.chunk_count} chunks"
            )
            self._is_initialized = True
            self._init_llm()
            return

        # Load documents
        self.logger.log_info(f"Loading documents from: {corpus_path}")

        if corpus_path.is_file():
            if corpus_path.suffix.lower() == ".pdf":
                self._documents = [self.doc_processor.load_pdf(corpus_path)]
            else:
                raise MiniRAGError(f"Unsupported file type: {corpus_path.suffix}")
        else:
            self._documents = self.doc_processor.load_directory(corpus_path)

        if not self._documents:
            raise MiniRAGError(f"No documents found in: {corpus_path}")

        # Get all chunks
        all_chunks = self.doc_processor.get_all_chunks(self._documents)
        self.logger.log_info(
            f"Loaded {len(self._documents)} document(s) with {len(all_chunks)} chunks"
        )

        # Index chunks
        self.logger.log_info("Indexing chunks...")
        self.retriever.index_chunks(all_chunks)
        self.logger.log_info(
            f"Indexing complete. Using {self.retriever.embedding_backend} backend."
        )

        self._is_initialized = True
        self._init_llm()

    def query(self, question: str) -> AgentResponse:
        """Process a query through the agentic loop."""

        if not self._is_initialized:
            raise MiniRAGError("Agent not initialized. Call initialize() first.")

        # Create trace
        trace = QueryTrace(question)
        trace.start()

        try:
            # Step 1: Plan
            trace.start_step("plan")
            trace.set_plan(self.PLAN_STEPS)
            self.logger.log_debug(f"Plan: {self.PLAN_STEPS}")
            trace.end_step()

            # Step 2: Retrieve
            trace.start_step("retrieve")
            retrieval_results = self._retrieve(question, trace)
            trace.end_step()

            if not retrieval_results:
                trace.add_error("No relevant documents found")
                trace.finalize()
                self.logger.log_trace(trace)
                return AgentResponse(
                    answer="I couldn't find any relevant information in the documents.",
                    citations=[],
                    sources=[],
                    trace_id=trace.trace_id,
                    success=True,
                )

            # Step 3: Draft
            trace.start_step("draft")
            context, citations = self._prepare_context(retrieval_results)
            answer = self._draft_answer(question, context, citations)
            trace.end_step()

            # Step 4: Cite (already included in draft)
            trace.start_step("cite")
            trace.set_answer(answer, citations)
            trace.end_step()

            # Finalize
            trace.finalize()
            self.logger.log_trace(trace)

            return AgentResponse(
                answer=answer,
                citations=citations,
                sources=[r.to_dict() for r in retrieval_results],
                trace_id=trace.trace_id,
                success=True,
            )

        except Exception as e:
            trace.add_error(str(e))
            trace.finalize()
            self.logger.log_trace(trace)
            self.logger.log_error("Query failed", e)

            return AgentResponse(
                answer="",
                citations=[],
                sources=[],
                trace_id=trace.trace_id,
                success=False,
                error=str(e),
            )

    def _retrieve(self, question: str, trace: QueryTrace) -> list[RetrievalResult]:
        """Retrieve relevant chunks.

        Args:
            question: User's question.
            trace: Query trace for logging.

        Returns:
            List of retrieval results.
        """
        try:
            results = self.retriever.retrieve(question)
            trace.add_retrieval_results(results)
            return results
        except MiniRAGError as e:
            trace.add_error(f"Retrieval error: {e}")
            raise

    def _prepare_context(self, results: list[RetrievalResult]) -> tuple[str, list[str]]:
        """Prepare context and citations from retrieval results.

        Args:
            results: Retrieval results.

        Returns:
            Tuple of (context string, citations list).
        """
        context_parts = []
        citations = []

        for result in results:
            chunk = result.chunk
            citation = chunk.citation
            citations.append(citation)

            # Add chunk content with citation
            context_parts.append(f"{citation}:\n{chunk.content}")

        context = "\n\n".join(context_parts)
        return context, citations

    def _draft_answer(self, question: str, context: str, citations: list[str]) -> str:
        """Draft an answer using LLM or simple extraction.

        Args:
            question: User's question.
            context: Retrieved context.
            citations: Available citations.

        Returns:
            Generated answer.
        """
        if self._llm is None:
            self._init_llm()

        try:
            answer = self._llm.generate_with_context(question, context, citations)  # type: ignore
            return answer
        except MiniRAGError as e:
            self.logger.log_error("LLM generation failed, using fallback", e)
            # Fallback to simple extraction
            simple_llm = SimpleLLM()
            return simple_llm.generate_with_context(question, context, citations)

    def get_stats(self) -> dict[str, Any]:
        """Get agent statistics.

        Returns:
            Dictionary with agent stats.
        """
        return {
            "initialized": self._is_initialized,
            "document_count": len(self._documents),
            "chunk_count": self.retriever.chunk_count,
            "embedding_backend": self.retriever.embedding_backend
            if self._is_initialized
            else None,
            "llm_available": self._llm_available,
        }
