class MiniRAGError(Exception):
    """Application error for Mini RAG system."""

    def __init__(self, message: str, context: str | None = None):
        self.context = context
        super().__init__(f"{message}" + (f" ({context})" if context else ""))
