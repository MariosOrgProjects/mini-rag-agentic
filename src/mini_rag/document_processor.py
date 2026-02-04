"""PDF document processor with chunking functionality."""

import re
from pathlib import Path

from pypdf import PdfReader

from mini_rag.configuration import Settings
from mini_rag.exceptions import MiniRAGError
from mini_rag.models import Document, DocumentChunk


class DocumentProcessor:
    """Process PDF documents and split into chunks."""

    def __init__(self, settings: Settings | None = None):
        """Initialize the document processor.

        Args:
            settings: Application settings. Uses defaults if not provided.
        """
        self.settings = settings or Settings()
        self.chunk_size = self.settings.chunk_size
        self.chunk_overlap = self.settings.chunk_overlap

    def load_pdf(self, file_path: Path) -> Document:
        """Load a PDF file and extract text.

        Args:
            file_path: Path to the PDF file.

        Returns:
            Document object with extracted text chunks.

        Raises:
            MiniRAGError: If the file cannot be processed.
        """
        if not file_path.exists():
            raise MiniRAGError(f"File not found: {file_path}", str(file_path))

        if not file_path.suffix.lower() == ".pdf":
            raise MiniRAGError(
                f"Unsupported file format: {file_path.suffix}", str(file_path)
            )

        try:
            reader = PdfReader(file_path)
            document = Document(
                file_path=file_path,
                file_name=file_path.name,
                total_pages=len(reader.pages),
            )

            # Extract text from all pages with page tracking
            page_texts: list[tuple[int, str]] = []
            for page_num, page in enumerate(reader.pages, start=1):
                text = page.extract_text() or ""
                if text.strip():
                    page_texts.append((page_num, text))
                    document.total_chars += len(text)

            # Create chunks with page awareness
            document.chunks = self._create_chunks(page_texts, file_path)

            return document

        except Exception as e:
            if isinstance(e, MiniRAGError):
                raise
            raise MiniRAGError(f"Failed to process PDF: {e}", str(file_path)) from e

    def load_directory(self, directory: Path) -> list[Document]:
        """Load all PDF files from a directory.

        Args:
            directory: Path to the directory containing PDFs.

        Returns:
            List of processed Document objects.

        Raises:
            MiniRAGError: If the directory doesn't exist.
        """
        if not directory.exists():
            raise MiniRAGError(f"Directory not found: {directory}")

        if not directory.is_dir():
            raise MiniRAGError(f"Not a directory: {directory}")

        documents = []
        pdf_files = list(directory.glob("*.pdf")) + list(directory.glob("*.PDF"))

        if not pdf_files:
            raise MiniRAGError(f"No PDF files found in: {directory}")

        for pdf_path in sorted(pdf_files):
            try:
                doc = self.load_pdf(pdf_path)
                documents.append(doc)
            except MiniRAGError:
                # Log but continue with other files
                continue

        return documents

    def _create_chunks(
        self, page_texts: list[tuple[int, str]], file_path: Path
    ) -> list[DocumentChunk]:
        """Create chunks from page texts with overlap.

        Args:
            page_texts: List of (page_number, text) tuples.
            file_path: Path to the source file.

        Returns:
            List of DocumentChunk objects.
        """
        chunks: list[DocumentChunk] = []
        chunk_id = 0

        for page_num, text in page_texts:
            # Clean the text
            text = self._clean_text(text)
            if not text.strip():
                continue

            # Split into sentences for better chunk boundaries
            sentences = self._split_into_sentences(text)

            current_chunk = ""
            current_start = 0

            for sentence in sentences:
                # Check if adding this sentence exceeds chunk size
                if (
                    len(current_chunk) + len(sentence) > self.chunk_size
                    and current_chunk
                ):
                    # Save current chunk
                    chunk = DocumentChunk(
                        content=current_chunk.strip(),
                        chunk_id=chunk_id,
                        file_name=file_path.name,
                        file_path=str(file_path),
                        page_number=page_num,
                        start_char=current_start,
                        end_char=current_start + len(current_chunk),
                    )
                    chunks.append(chunk)
                    chunk_id += 1

                    # Start new chunk with overlap
                    overlap_text = (
                        current_chunk[-self.chunk_overlap :]
                        if self.chunk_overlap
                        else ""
                    )
                    current_chunk = overlap_text + sentence
                    current_start = (
                        current_start + len(current_chunk) - len(overlap_text)
                    )
                else:
                    current_chunk += sentence

            # Don't forget the last chunk
            if current_chunk.strip():
                chunk = DocumentChunk(
                    content=current_chunk.strip(),
                    chunk_id=chunk_id,
                    file_name=file_path.name,
                    file_path=str(file_path),
                    page_number=page_num,
                    start_char=current_start,
                    end_char=current_start + len(current_chunk),
                )
                chunks.append(chunk)
                chunk_id += 1

        return chunks

    def _clean_text(self, text: str) -> str:
        """Clean extracted text.

        Args:
            text: Raw text from PDF.

        Returns:
            Cleaned text.
        """
        # Replace multiple whitespace with single space
        text = re.sub(r"\s+", " ", text)
        # Remove control characters except newlines
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", text)
        return text.strip()

    def _split_into_sentences(self, text: str) -> list[str]:
        """Split text into sentences.

        Args:
            text: Text to split.

        Returns:
            List of sentences.
        """
        # Simple sentence splitting - handles common cases
        # Pattern matches sentence endings followed by space and capital letter
        pattern = r"(?<=[.!?])\s+(?=[A-Z])"
        sentences = re.split(pattern, text)

        # If no splits, return the whole text
        if len(sentences) == 1 and len(text) > self.chunk_size:
            # Fallback: split by any punctuation or by fixed size
            sentences = re.split(r"(?<=[.!?;:])\s+", text)

        return [s.strip() + " " for s in sentences if s.strip()]

    def get_all_chunks(self, documents: list[Document]) -> list[DocumentChunk]:
        """Get all chunks from a list of documents.

        Args:
            documents: List of Document objects.

        Returns:
            Flat list of all DocumentChunk objects.
        """
        all_chunks = []
        for doc in documents:
            all_chunks.extend(doc.chunks)
        return all_chunks
