"""Document retriever with vector similarity search."""

import json
import pickle
import shutil
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from mini_rag.configuration import Settings
from mini_rag.embeddings import EmbeddingService, TfidfEmbedding
from mini_rag.exceptions import MiniRAGError
from mini_rag.models import DocumentChunk, RetrievalResult


class VectorStore:
    """Simple vector store for document embeddings."""

    def __init__(self, store_path: Path | None = None):
        """Initialize vector store.

        Args:
            store_path: Path to persist the vector store.
        """
        self.store_path = store_path
        self.embeddings: NDArray[np.float32] | None = None
        self.chunks: list[DocumentChunk] = []
        self._is_loaded = False

    def add(self, chunks: list[DocumentChunk], embeddings: NDArray[np.float32]) -> None:
        """Add chunks and their embeddings to the store.

        Args:
            chunks: Document chunks to store.
            embeddings: Corresponding embedding vectors.

        Raises:
            MiniRAGError: If dimensions don't match.
        """
        if len(chunks) != len(embeddings):
            raise MiniRAGError(
                f"Chunk count ({len(chunks)}) doesn't match embedding count ({len(embeddings)})"
            )

        self.chunks = chunks
        self.embeddings = embeddings
        self._is_loaded = True

    def search(
        self,
        query_embedding: NDArray[np.float32],
        top_k: int = 3,
        threshold: float = 0.0,
    ) -> list[tuple[int, float]]:
        """Search for similar vectors.

        Args:
            query_embedding: Query vector.
            top_k: Number of results to return.
            threshold: Minimum similarity threshold.

        Returns:
            List of (index, score) tuples.

        Raises:
            MiniRAGError: If store is empty.
        """
        if self.embeddings is None or len(self.chunks) == 0:
            raise MiniRAGError("Vector store is empty. Add documents first.")

        # Compute cosine similarity
        scores = self._cosine_similarity(query_embedding, self.embeddings)

        # Get top-k indices above threshold
        indices = np.argsort(scores)[::-1][:top_k]
        results = [
            (int(idx), float(scores[idx]))
            for idx in indices
            if scores[idx] >= threshold
        ]

        return results

    def _cosine_similarity(
        self, query: NDArray[np.float32], documents: NDArray[np.float32]
    ) -> NDArray[np.float32]:
        """Compute cosine similarity between query and documents.

        Args:
            query: Query vector (1D).
            documents: Document vectors (2D).

        Returns:
            Array of similarity scores.
        """
        # Normalize vectors
        query_norm = query / (np.linalg.norm(query) + 1e-10)
        doc_norms = documents / (
            np.linalg.norm(documents, axis=1, keepdims=True) + 1e-10
        )

        # Compute dot product
        similarities = np.dot(doc_norms, query_norm)
        return similarities.astype(np.float32)

    def save(self) -> None:
        """Persist vector store to disk.

        Raises:
            MiniRAGError: If save fails.
        """
        if self.store_path is None:
            raise MiniRAGError("No store path specified")

        try:
            self.store_path.mkdir(parents=True, exist_ok=True)

            # Save embeddings
            embeddings_path = self.store_path / "embeddings.npy"
            if self.embeddings is not None:
                np.save(embeddings_path, self.embeddings)

            # Save chunks as JSON-serializable data
            chunks_path = self.store_path / "chunks.pkl"
            with open(chunks_path, "wb") as f:
                pickle.dump([c.to_dict() for c in self.chunks], f)

            # Save metadata
            meta_path = self.store_path / "metadata.json"
            with open(meta_path, "w") as f:
                json.dump(
                    {
                        "chunk_count": len(self.chunks),
                        "embedding_dim": self.embeddings.shape[1]
                        if self.embeddings is not None
                        else 0,
                    },
                    f,
                )

        except Exception as e:
            raise MiniRAGError(f"Failed to save vector store: {e}") from e

    def load(self) -> bool:
        """Load vector store from disk.

        Returns:
            True if loaded successfully, False if store doesn't exist.

        Raises:
            MiniRAGError: If load fails.
        """
        if self.store_path is None:
            return False

        embeddings_path = self.store_path / "embeddings.npy"
        chunks_path = self.store_path / "chunks.pkl"

        if not embeddings_path.exists() or not chunks_path.exists():
            return False

        try:
            self.embeddings = np.load(embeddings_path)

            with open(chunks_path, "rb") as f:
                chunk_data = pickle.load(f)
                self.chunks = [DocumentChunk.from_dict(d) for d in chunk_data]

            self._is_loaded = True
            return True

        except Exception as e:
            raise MiniRAGError(f"Failed to load vector store: {e}") from e

    def clear(self) -> None:
        """Clear the vector store."""
        self.embeddings = None
        self.chunks = []
        self._is_loaded = False

        if self.store_path and self.store_path.exists():
            shutil.rmtree(self.store_path)

    @property
    def is_loaded(self) -> bool:
        """Check if store has data."""
        return self._is_loaded and len(self.chunks) > 0


class Retriever:
    """Document retriever combining embedding and search."""

    def __init__(self, settings: Settings | None = None):
        """Initialize retriever.

        Args:
            settings: Application settings.
        """
        self.settings = settings or Settings()
        self.embedding_service = EmbeddingService(settings)
        self.vector_store = VectorStore(self.settings.vectorstore_path)
        self._chunks: list[DocumentChunk] = []

    def index_chunks(self, chunks: list[DocumentChunk], persist: bool = True) -> None:
        """Index document chunks.

        Args:
            chunks: Chunks to index.
            persist: Whether to save to disk.

        Raises:
            MiniRAGError: If indexing fails.
        """
        if not chunks:
            raise MiniRAGError("No chunks to index")

        try:
            self._chunks = chunks
            embeddings = self.embedding_service.embed_chunks(chunks)
            self.vector_store.add(chunks, embeddings)

            if persist:
                self.vector_store.save()

        except Exception as e:
            raise MiniRAGError(f"Failed to index chunks: {e}") from e

    def load_index(self) -> bool:
        """Load existing index from disk.

        Returns:
            True if loaded successfully.
        """
        loaded = self.vector_store.load()
        if loaded:
            self._chunks = self.vector_store.chunks
            # Re-fit the TF-IDF vectorizer if using TF-IDF backend
            if self._chunks:
                texts = [chunk.content for chunk in self._chunks]
                # Force TF-IDF to re-fit by embedding the chunks
                backend = self.embedding_service.get_backend()
                if isinstance(backend, TfidfEmbedding) and not backend._is_fitted:
                    backend.fit(texts)
        return loaded

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievalResult]:
        """Retrieve relevant chunks for a query.

        Args:
            query: Search query.
            top_k: Number of results (defaults to settings).

        Returns:
            List of RetrievalResult objects.

        Raises:
            MiniRAGError: If retrieval fails.
        """
        if not query.strip():
            raise MiniRAGError("Query cannot be empty")

        if not self.vector_store.is_loaded:
            raise MiniRAGError(
                "No index loaded. Call index_chunks() or load_index() first."
            )

        top_k = top_k or self.settings.top_k

        try:
            # Embed query
            query_embedding = self.embedding_service.embed_text(query)

            # Search
            results = self.vector_store.search(
                query_embedding,
                top_k=top_k,
                threshold=self.settings.similarity_threshold,
            )

            # Build results
            retrieval_results = []
            for rank, (idx, score) in enumerate(results, start=1):
                chunk = self.vector_store.chunks[idx]
                retrieval_results.append(
                    RetrievalResult(chunk=chunk, score=score, rank=rank)
                )

            return retrieval_results

        except Exception as e:
            if isinstance(e, MiniRAGError):
                raise
            raise MiniRAGError(f"Retrieval failed: {e}") from e

    @property
    def chunk_count(self) -> int:
        """Get number of indexed chunks."""
        return len(self.vector_store.chunks)

    @property
    def embedding_backend(self) -> str:
        """Get the active embedding backend."""
        return self.embedding_service.active_backend
