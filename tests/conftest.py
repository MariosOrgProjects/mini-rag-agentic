"""Test fixtures and configuration for pytest."""

import tempfile
from pathlib import Path
from typing import Generator

import numpy as np
import pytest

from mini_rag.configuration import Settings
from mini_rag.models import DocumentChunk


# Path to test files directory
TEST_FILES_DIR = Path(__file__).parent / "test_files"


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


def _set_test_env_vars(monkeypatch, temp_dir: Path) -> None:
    """Set all required environment variables for testing."""
    monkeypatch.setenv("MINI_RAG_CHUNK_SIZE", "200")
    monkeypatch.setenv("MINI_RAG_CHUNK_OVERLAP", "20")
    monkeypatch.setenv("MINI_RAG_CORPUS_DIR", str(temp_dir))
    monkeypatch.setenv("MINI_RAG_EMBEDDING_BACKEND", "tfidf")
    monkeypatch.setenv("MINI_RAG_OLLAMA_BASE_URL", "http://localhost:11434")
    monkeypatch.setenv("MINI_RAG_OLLAMA_MODEL", "nomic-embed-text")
    monkeypatch.setenv("MINI_RAG_OLLAMA_TIMEOUT", "30")
    monkeypatch.setenv("MINI_RAG_TOP_K", "3")
    monkeypatch.setenv("MINI_RAG_SIMILARITY_THRESHOLD", "0.0")
    monkeypatch.setenv("MINI_RAG_LLM_MODEL", "llama3.2")
    monkeypatch.setenv("MINI_RAG_LLM_TIMEOUT", "120")
    monkeypatch.setenv("MINI_RAG_MAX_TOKENS", "512")
    monkeypatch.setenv("MINI_RAG_LOG_DIR", str(temp_dir / "logs"))
    monkeypatch.setenv("MINI_RAG_LOG_LEVEL", "INFO")
    monkeypatch.setenv("MINI_RAG_VECTORSTORE_PATH", str(temp_dir / ".vectorstore"))


@pytest.fixture(autouse=True)
def setup_test_env(temp_dir: Path, monkeypatch) -> None:
    """Automatically set up test environment variables for all tests."""
    _set_test_env_vars(monkeypatch, temp_dir)


@pytest.fixture
def settings(temp_dir: Path, monkeypatch) -> Settings:
    """Create test settings using environment variables."""
    _set_test_env_vars(monkeypatch, temp_dir)
    return Settings()


@pytest.fixture
def sample_chunks() -> list[DocumentChunk]:
    """Create sample document chunks for testing."""
    return [
        DocumentChunk(
            content="The quick brown fox jumps over the lazy dog.",
            chunk_id=0,
            file_name="test.pdf",
            file_path="/test/test.pdf",
            page_number=1,
        ),
        DocumentChunk(
            content="Machine learning is a subset of artificial intelligence.",
            chunk_id=1,
            file_name="test.pdf",
            file_path="/test/test.pdf",
            page_number=1,
        ),
        DocumentChunk(
            content="Python is a popular programming language for data science.",
            chunk_id=2,
            file_name="test.pdf",
            file_path="/test/test.pdf",
            page_number=2,
        ),
        DocumentChunk(
            content="Neural networks are inspired by the human brain.",
            chunk_id=3,
            file_name="test2.pdf",
            file_path="/test/test2.pdf",
            page_number=1,
        ),
        DocumentChunk(
            content="Deep learning uses multiple layers of neural networks.",
            chunk_id=4,
            file_name="test2.pdf",
            file_path="/test/test2.pdf",
            page_number=1,
        ),
    ]


@pytest.fixture
def sample_embeddings() -> np.ndarray:
    """Create sample embeddings for testing."""
    np.random.seed(42)
    return np.random.rand(5, 768).astype(np.float32)


@pytest.fixture
def sample_pdf_content() -> bytes:
    """Load PDF content from test_files/test.pdf."""
    test_pdf = TEST_FILES_DIR / "test.pdf"
    return test_pdf.read_bytes()


@pytest.fixture
def sample_pdf_path(temp_dir: Path, sample_pdf_content: bytes) -> Path:
    """Copy test PDF to temp directory for isolation."""
    pdf_path = temp_dir / "test_document.pdf"
    pdf_path.write_bytes(sample_pdf_content)
    return pdf_path
