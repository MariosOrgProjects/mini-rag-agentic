"""Embedding service with Ollama and TF-IDF fallback."""

from abc import ABC, abstractmethod

import numpy as np
import requests
from numpy.typing import NDArray
from sklearn.feature_extraction.text import TfidfVectorizer

from mini_rag.configuration import Settings
from mini_rag.exceptions import MiniRAGError
from mini_rag.models import DocumentChunk


class BaseEmbedding(ABC):
    """Abstract base class for embedding services."""

    @abstractmethod
    def embed_text(self, text: str) -> NDArray[np.float32]:
        """Embed a single text."""
        pass

    @abstractmethod
    def embed_texts(self, texts: list[str]) -> NDArray[np.float32]:
        """Embed multiple texts."""
        pass

    @abstractmethod
    def embed_chunks(self, chunks: list[DocumentChunk]) -> NDArray[np.float32]:
        """Embed document chunks."""
        pass


class OllamaEmbedding(BaseEmbedding):
    """Ollama-based embedding service."""

    def __init__(self, settings: Settings):
        """Initialize Ollama embedding service.

        Args:
            settings: Application settings.
        """
        self.base_url = settings.ollama_base_url
        self.model = settings.ollama_model
        self.timeout = settings.ollama_timeout
        self._embedding_dim: int | None = None

    def _check_availability(self) -> bool:
        """Check if Ollama is available.

        Returns:
            True if Ollama is reachable.
        """
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except requests.RequestException:
            return False

    def embed_text(self, text: str) -> NDArray[np.float32]:
        """Embed a single text using Ollama."""

        if not text.strip():
            raise MiniRAGError("Cannot embed empty text", "ollama")

        try:
            response = requests.post(
                f"{self.base_url}/api/embeddings",
                json={"model": self.model, "prompt": text},
                timeout=self.timeout,
            )
            response.raise_for_status()

            data = response.json()
            embedding = np.array(data["embedding"], dtype=np.float32)
            self._embedding_dim = len(embedding)
            return embedding

        except requests.Timeout:
            raise MiniRAGError(
                f"Ollama request timed out after {self.timeout}s", "ollama"
            )
        except requests.RequestException as e:
            raise MiniRAGError(f"Ollama request failed: {e}", "ollama") from e
        except (KeyError, ValueError) as e:
            raise MiniRAGError(f"Invalid response from Ollama: {e}", "ollama") from e

    def embed_texts(self, texts: list[str]) -> NDArray[np.float32]:
        """Embed multiple texts."""

        if not texts:
            raise MiniRAGError("Cannot embed empty list of texts", "ollama")

        embeddings = []
        for text in texts:
            embedding = self.embed_text(text)
            embeddings.append(embedding)

        return np.vstack(embeddings)

    def embed_chunks(self, chunks: list[DocumentChunk]) -> NDArray[np.float32]:
        """Embed document chunks."""

        texts = [chunk.content for chunk in chunks]
        return self.embed_texts(texts)


class TfidfEmbedding(BaseEmbedding):
    """TF-IDF based embedding as fallback."""

    def __init__(self, settings: Settings | None = None):
        """Initialize TF-IDF embedding."""

        self.vectorizer: TfidfVectorizer | None = None
        self._is_fitted = False
        self._corpus_texts: list[str] = []

    def _create_vectorizer(self, n_docs: int) -> TfidfVectorizer:
        """Create a vectorizer appropriate for the corpus size.

        Args:
            n_docs: Number of documents in corpus.

        Returns:
            Configured TfidfVectorizer.
        """
        # Adjust max_df based on corpus size to avoid errors with small corpora
        max_df = 1.0 if n_docs < 5 else 0.95
        min_df = 1

        return TfidfVectorizer(
            max_features=768,  # Match typical embedding dimension
            stop_words="english",
            ngram_range=(1, 2),
            min_df=min_df,
            max_df=max_df,
        )

    def fit(self, texts: list[str]) -> None:
        """Fit the TF-IDF vectorizer on corpus.

        Args:
            texts: Corpus texts to fit on.
        """
        if not texts:
            raise MiniRAGError("Cannot fit on empty corpus", "tfidf")

        self._corpus_texts = texts
        self.vectorizer = self._create_vectorizer(len(texts))
        self.vectorizer.fit(texts)
        self._is_fitted = True

    def embed_text(self, text: str) -> NDArray[np.float32]:
        """Embed a single text."""

        if not self._is_fitted or self.vectorizer is None:
            raise MiniRAGError(
                "TF-IDF vectorizer not fitted. Call fit() first.", "tfidf"
            )

        if not text.strip():
            raise MiniRAGError("Cannot embed empty text", "tfidf")

        vector = self.vectorizer.transform([text]).toarray()[0]
        return vector.astype(np.float32)

    def embed_texts(self, texts: list[str]) -> NDArray[np.float32]:
        """Embed multiple texts."""

        if not self._is_fitted or self.vectorizer is None:
            raise MiniRAGError(
                "TF-IDF vectorizer not fitted. Call fit() first.", "tfidf"
            )

        if not texts:
            raise MiniRAGError("Cannot embed empty list of texts", "tfidf")

        vectors = self.vectorizer.transform(texts).toarray()
        return vectors.astype(np.float32)

    def embed_chunks(self, chunks: list[DocumentChunk]) -> NDArray[np.float32]:
        """Embed document chunks."""

        texts = [chunk.content for chunk in chunks]

        # Auto-fit if not fitted
        if not self._is_fitted:
            self.fit(texts)

        return self.embed_texts(texts)


class EmbeddingService:
    """Unified embedding service with automatic fallback."""

    def __init__(self, settings: Settings | None = None):
        """Initialize embedding service.

        Args:
            settings: Application settings.
        """
        self.settings = settings or Settings()
        self._ollama: OllamaEmbedding | None = None
        self._tfidf: TfidfEmbedding | None = None
        self._active_backend: str | None = None

    def _init_ollama(self) -> OllamaEmbedding | None:
        """Initialize Ollama backend if available."""
        if self._ollama is None:
            self._ollama = OllamaEmbedding(self.settings)
            if self._ollama._check_availability():
                return self._ollama
            self._ollama = None
        return self._ollama

    def _init_tfidf(self) -> TfidfEmbedding:
        """Initialize TF-IDF backend."""
        if self._tfidf is None:
            self._tfidf = TfidfEmbedding(self.settings)
        return self._tfidf

    def get_backend(self) -> BaseEmbedding:
        """Get the active embedding backend."""

        # Try preferred backend first
        if self.settings.embedding_backend == "ollama":
            ollama = self._init_ollama()
            if ollama is not None:
                self._active_backend = "ollama"
                return ollama
            # Fallback to TF-IDF
            self._active_backend = "tfidf"
            return self._init_tfidf()
        else:
            self._active_backend = "tfidf"
            return self._init_tfidf()

    @property
    def active_backend(self) -> str:
        """Get the name of the active backend."""
        if self._active_backend is None:
            self.get_backend()
        return self._active_backend or "unknown"

    def embed_text(self, text: str) -> NDArray[np.float32]:
        """Embed a single text."""

        return self.get_backend().embed_text(text)

    def embed_texts(self, texts: list[str]) -> NDArray[np.float32]:
        """Embed multiple texts."""

        return self.get_backend().embed_texts(texts)

    def embed_chunks(self, chunks: list[DocumentChunk]) -> NDArray[np.float32]:
        """Embed document chunks.

        Args:
            chunks: List of document chunks.

        Returns:
            2D array of embedding vectors.
        """
        backend = self.get_backend()

        # For TF-IDF, ensure it's fitted on the corpus
        if isinstance(backend, TfidfEmbedding) and not backend._is_fitted:
            texts = [chunk.content for chunk in chunks]
            backend.fit(texts)

        return backend.embed_chunks(chunks)
