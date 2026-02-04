"""Tests for retriever module."""

from pathlib import Path

import numpy as np
import pytest

from mini_rag.configuration import Settings
from mini_rag.exceptions import MiniRAGError
from mini_rag.models import DocumentChunk
from mini_rag.retriever import Retriever, VectorStore


class TestVectorStore:
    """Tests for VectorStore class."""

    def test_vector_store_initialization(self, temp_dir: Path) -> None:
        """Test vector store initialization."""
        store = VectorStore(temp_dir / "store")
        assert store.embeddings is None
        assert store.chunks == []
        assert not store.is_loaded

    def test_vector_store_add_and_search(
        self, sample_chunks: list[DocumentChunk], sample_embeddings: np.ndarray
    ) -> None:
        """Test adding and searching embeddings."""
        store = VectorStore()
        store.add(sample_chunks, sample_embeddings)

        assert len(store.chunks) == len(sample_chunks)
        assert store.is_loaded

        query = sample_embeddings[0]
        results = store.search(query, top_k=3)
        assert len(results) <= 3
        assert results[0][0] == 0  # First result matches query

    def test_vector_store_search_empty(self) -> None:
        """Test search on empty store."""
        store = VectorStore()
        query = np.random.rand(768).astype(np.float32)

        with pytest.raises(MiniRAGError, match="empty"):
            store.search(query)

    def test_vector_store_save_and_load(
        self,
        temp_dir: Path,
        sample_chunks: list[DocumentChunk],
        sample_embeddings: np.ndarray,
    ) -> None:
        """Test saving and loading vector store."""
        store_path = temp_dir / "vectorstore"

        store = VectorStore(store_path)
        store.add(sample_chunks, sample_embeddings)
        store.save()

        store2 = VectorStore(store_path)
        loaded = store2.load()

        assert loaded
        assert len(store2.chunks) == len(sample_chunks)


class TestRetriever:
    """Tests for Retriever class."""

    def test_retriever_initialization(self, settings: Settings) -> None:
        """Test retriever initialization."""
        retriever = Retriever(settings)
        assert retriever.settings == settings
        assert retriever.chunk_count == 0

    def test_retriever_index_and_retrieve(
        self, settings: Settings, sample_chunks: list[DocumentChunk]
    ) -> None:
        """Test indexing and retrieval."""
        retriever = Retriever(settings)
        retriever.index_chunks(sample_chunks, persist=False)

        assert retriever.chunk_count == len(sample_chunks)

        results = retriever.retrieve("machine learning")
        assert len(results) > 0
        assert all(hasattr(r, "chunk") for r in results)

    def test_retriever_index_empty_chunks(self, settings: Settings) -> None:
        """Test indexing empty list."""
        retriever = Retriever(settings)
        with pytest.raises(MiniRAGError, match="No chunks"):
            retriever.index_chunks([])

    def test_retriever_retrieve_not_indexed(self, settings: Settings) -> None:
        """Test retrieval without indexing."""
        retriever = Retriever(settings)
        with pytest.raises(MiniRAGError, match="No index"):
            retriever.retrieve("test query")

    def test_retriever_persist_and_load(
        self, settings: Settings, sample_chunks: list[DocumentChunk]
    ) -> None:
        """Test persisting and loading index."""
        retriever = Retriever(settings)
        retriever.index_chunks(sample_chunks, persist=True)

        retriever2 = Retriever(settings)
        loaded = retriever2.load_index()

        assert loaded
        assert retriever2.chunk_count == len(sample_chunks)
