"""Tests for embedding module."""

import numpy as np
import pytest
from pytest_mock import MockerFixture

from mini_rag.configuration import Settings
from mini_rag.embeddings import EmbeddingService, OllamaEmbedding, TfidfEmbedding
from mini_rag.exceptions import MiniRAGError
from mini_rag.models import DocumentChunk


class TestTfidfEmbedding:
    """Tests for TF-IDF embedding."""

    def test_tfidf_initialization(self, settings: Settings) -> None:
        """Test TF-IDF embedding initialization."""
        tfidf = TfidfEmbedding(settings)
        assert not tfidf._is_fitted

    def test_tfidf_fit_and_embed(self) -> None:
        """Test TF-IDF fitting and embedding."""
        tfidf = TfidfEmbedding()
        texts = ["Hello world", "Machine learning", "Data science"]
        tfidf.fit(texts)

        assert tfidf._is_fitted
        embedding = tfidf.embed_text("Hello")
        assert isinstance(embedding, np.ndarray)
        assert embedding.dtype == np.float32

    def test_tfidf_fit_empty_corpus(self) -> None:
        """Test TF-IDF fitting with empty corpus."""
        tfidf = TfidfEmbedding()
        with pytest.raises(MiniRAGError, match="empty corpus"):
            tfidf.fit([])

    def test_tfidf_embed_chunks(self, sample_chunks: list[DocumentChunk]) -> None:
        """Test embedding document chunks."""
        tfidf = TfidfEmbedding()
        embeddings = tfidf.embed_chunks(sample_chunks)

        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape[0] == len(sample_chunks)
        assert tfidf._is_fitted


class TestOllamaEmbedding:
    """Tests for Ollama embedding."""

    def test_ollama_initialization(self, settings: Settings) -> None:
        """Test Ollama embedding initialization."""
        settings.ollama_base_url = "http://localhost:11434"
        settings.ollama_model = "nomic-embed-text"
        ollama = OllamaEmbedding(settings)

        assert ollama.base_url == "http://localhost:11434"
        assert ollama.model == "nomic-embed-text"

    def test_ollama_embed_text_success(
        self, settings: Settings, mocker: MockerFixture
    ) -> None:
        """Test successful text embedding."""
        mock_response = mocker.Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"embedding": [0.1, 0.2, 0.3]}
        mocker.patch("requests.post", return_value=mock_response)

        ollama = OllamaEmbedding(settings)
        embedding = ollama.embed_text("Hello world")

        assert isinstance(embedding, np.ndarray)
        assert len(embedding) == 3

    def test_ollama_embed_text_error(
        self, settings: Settings, mocker: MockerFixture
    ) -> None:
        """Test error handling."""
        import requests

        mocker.patch("requests.post", side_effect=requests.ConnectionError())

        ollama = OllamaEmbedding(settings)
        with pytest.raises(MiniRAGError, match="request failed"):
            ollama.embed_text("Hello world")


class TestEmbeddingService:
    """Tests for unified EmbeddingService."""

    def test_service_initialization(self, settings: Settings) -> None:
        """Test embedding service initialization."""
        service = EmbeddingService(settings)
        assert service.settings == settings

    def test_service_use_tfidf(self, settings: Settings) -> None:
        """Test direct TF-IDF usage."""
        settings.embedding_backend = "tfidf"
        service = EmbeddingService(settings)
        backend = service.get_backend()

        assert isinstance(backend, TfidfEmbedding)
        assert service.active_backend == "tfidf"

    def test_service_embed_chunks(
        self, settings: Settings, sample_chunks: list[DocumentChunk]
    ) -> None:
        """Test embedding chunks through service."""
        settings.embedding_backend = "tfidf"
        service = EmbeddingService(settings)
        embeddings = service.embed_chunks(sample_chunks)

        assert embeddings.shape[0] == len(sample_chunks)
