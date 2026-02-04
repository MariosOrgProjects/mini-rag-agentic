import os
from pathlib import Path
from dotenv import load_dotenv


# Load environment variables from .env file at module import
load_dotenv()


class Configuration:
    """Configuration class for Mini RAG application.

    All settings are read from environment variables with MINI_RAG_ prefix.
    Requires .env file or environment variables to be set.
    """

    def __init__(self):
        # Document Processing
        self.chunk_size: int = int(os.environ.get("MINI_RAG_CHUNK_SIZE"))
        self.chunk_overlap: int = int(os.environ.get("MINI_RAG_CHUNK_OVERLAP"))
        self.corpus_dir: Path = Path(os.environ.get("MINI_RAG_CORPUS_DIR"))

        # Embedding Settings
        self.embedding_backend: str = os.environ.get("MINI_RAG_EMBEDDING_BACKEND")
        self.ollama_base_url: str = os.environ.get("MINI_RAG_OLLAMA_BASE_URL")
        self.ollama_model: str = os.environ.get("MINI_RAG_OLLAMA_MODEL")
        self.ollama_timeout: int = int(os.environ.get("MINI_RAG_OLLAMA_TIMEOUT"))

        # Retrieval Settings
        self.top_k: int = int(os.environ.get("MINI_RAG_TOP_K"))
        self.similarity_threshold: float = float(
            os.environ.get("MINI_RAG_SIMILARITY_THRESHOLD")
        )

        # LLM Settings
        self.llm_model: str = os.environ.get("MINI_RAG_LLM_MODEL")
        self.llm_timeout: int = int(os.environ.get("MINI_RAG_LLM_TIMEOUT"))
        self.max_tokens: int = int(os.environ.get("MINI_RAG_MAX_TOKENS"))

        # Logging
        self.log_dir: Path = Path(os.environ.get("MINI_RAG_LOG_DIR"))
        self.log_level: str = os.environ.get("MINI_RAG_LOG_LEVEL").upper()

        # Vector Store
        self.vectorstore_path: Path = Path(os.environ.get("MINI_RAG_VECTORSTORE_PATH"))


def read_configuration_from_env() -> Configuration:
    """Read configuration from environment variables.

    Returns:
        Configuration: Configuration instance with values from environment.
    """
    return Configuration()


# Backward compatibility aliases
Settings = Configuration
get_settings = read_configuration_from_env
