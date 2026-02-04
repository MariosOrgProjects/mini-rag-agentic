"""Document models for the RAG system."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class DocumentChunk:
    """Represents a chunk of a document with metadata."""

    content: str
    chunk_id: int
    file_name: str
    file_path: str
    page_number: int | None = None
    start_char: int = 0
    end_char: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate chunk data."""
        if not self.content.strip():
            raise ValueError("Chunk content cannot be empty")
        if self.chunk_id < 0:
            raise ValueError("Chunk ID must be non-negative")

    @property
    def citation(self) -> str:
        """Generate citation string for this chunk."""
        parts = [f"[{self.file_name}"]
        if self.page_number is not None:
            parts.append(f", Page {self.page_number}")
        parts.append(f", Chunk {self.chunk_id}]")
        return "".join(parts)

    def to_dict(self) -> dict[str, Any]:
        """Convert chunk to dictionary representation."""
        return {
            "content": self.content,
            "chunk_id": self.chunk_id,
            "file_name": self.file_name,
            "file_path": self.file_path,
            "page_number": self.page_number,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DocumentChunk":
        """Create chunk from dictionary."""
        return cls(
            content=data["content"],
            chunk_id=data["chunk_id"],
            file_name=data["file_name"],
            file_path=data["file_path"],
            page_number=data.get("page_number"),
            start_char=data.get("start_char", 0),
            end_char=data.get("end_char", 0),
            metadata=data.get("metadata", {}),
        )


@dataclass
class Document:
    """Represents a full document with its chunks."""

    file_path: Path
    file_name: str
    chunks: list[DocumentChunk] = field(default_factory=list)
    total_pages: int = 0
    total_chars: int = 0

    def __post_init__(self) -> None:
        """Initialize document."""
        if isinstance(self.file_path, str):
            self.file_path = Path(self.file_path)

    @property
    def chunk_count(self) -> int:
        """Return the number of chunks in this document."""
        return len(self.chunks)


@dataclass
class RetrievalResult:
    """Result from retrieval with similarity score."""

    chunk: DocumentChunk
    score: float
    rank: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "file": self.chunk.file_name,
            "chunk_id": self.chunk.chunk_id,
            "score": round(self.score, 4),
        }
