"""LLM service for text generation using Ollama."""

import requests

from mini_rag.configuration import Settings
from mini_rag.exceptions import MiniRAGError


class LLMService:
    """LLM service using Ollama for text generation."""

    def __init__(self, settings: Settings | None = None):
        """Initialize LLM service.

        Args:
            settings: Application settings.
        """
        self.settings = settings or Settings()
        self.base_url = self.settings.ollama_base_url
        self.model = self.settings.llm_model
        self.timeout = self.settings.llm_timeout
        self.max_tokens = self.settings.max_tokens

    def is_available(self) -> bool:
        """Check if Ollama LLM is available.

        Returns:
            True if available.
        """
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                # Check if model is available
                models = response.json().get("models", [])
                model_names = [m.get("name", "").split(":")[0] for m in models]
                return self.model.split(":")[0] in model_names
            return False
        except requests.RequestException:
            return False

    def generate(self, prompt: str, system_prompt: str | None = None) -> str:
        """Generate text using the LLM.

        Args:
            prompt: User prompt.
            system_prompt: Optional system prompt.

        Returns:
            Generated text.

        Raises:
            MiniRAGError: If generation fails.
        """
        if not prompt.strip():
            raise MiniRAGError("Prompt cannot be empty", self.model)

        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": self.max_tokens,
                },
            }

            if system_prompt:
                payload["system"] = system_prompt

            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()

            data = response.json()
            return data.get("response", "").strip()

        except requests.Timeout:
            raise MiniRAGError(
                f"LLM request timed out after {self.timeout}s", self.model
            )
        except requests.RequestException as e:
            raise MiniRAGError(f"LLM request failed: {e}", self.model) from e
        except (KeyError, ValueError) as e:
            raise MiniRAGError(f"Invalid response from LLM: {e}", self.model) from e

    def generate_with_context(
        self, question: str, context: str, citations: list[str]
    ) -> str:
        """Generate an answer with context and citations.

        Args:
            question: User question.
            context: Retrieved context.
            citations: List of citation strings.

        Returns:
            Generated answer with citations.
        """
        system_prompt = """You are an expert research assistant that provides accurate, detailed answers based ONLY on the provided context.

Your approach:
1. Carefully read and understand the question
2. Identify ALL relevant information from the context passages
3. Synthesize a comprehensive answer that directly addresses the question
4. Always cite sources using [Source: filename, Chunk N] format
5. If information is partial or unclear, explain what is known and what is missing

Rules:
- ONLY use information from the provided context - never make up facts
- Be thorough - include all relevant details from the context
- Be precise - use exact terminology from the source documents
- If the context doesn't contain the answer, clearly state: "The provided documents do not contain information about..."
"""

        citation_list = "\n".join(f"  • {c}" for c in citations)

        prompt = f"""=== RETRIEVED CONTEXT ===
{context}

=== AVAILABLE SOURCES ===
{citation_list}

=== QUESTION ===
{question}

=== INSTRUCTIONS ===
Provide a comprehensive answer to the question above using ONLY the retrieved context.
- Quote or paraphrase relevant passages
- Include citations after each claim using [Source: filename, Chunk N]
- If multiple sources support a point, cite all of them
- Structure your answer clearly with paragraphs if needed

=== ANSWER ==="""

        return self.generate(prompt, system_prompt)


class SimpleLLM:
    """Simple fallback LLM that creates answers from context without API calls."""

    def generate_with_context(
        self, question: str, context: str, citations: list[str]
    ) -> str:
        """Generate a simple answer by extracting relevant content.

        Args:
            question: User question.
            context: Retrieved context.
            citations: List of citation strings.

        Returns:
            Answer constructed from context.
        """
        if not context.strip():
            return "No relevant information found in the documents."

        # Simple keyword-based extraction
        question_words = set(question.lower().split())
        stop_words = {
            "what",
            "is",
            "the",
            "a",
            "an",
            "how",
            "does",
            "do",
            "are",
            "was",
            "were",
            "can",
            "could",
            "would",
            "should",
            "who",
            "where",
            "when",
            "why",
            "which",
        }
        keywords = question_words - stop_words

        # Find sentences containing keywords
        sentences = context.replace("\n", " ").split(". ")
        relevant_sentences = []

        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(kw in sentence_lower for kw in keywords):
                relevant_sentences.append(sentence.strip())

        if relevant_sentences:
            # Take top 3 most relevant sentences
            answer_text = ". ".join(relevant_sentences[:3])
            if not answer_text.endswith("."):
                answer_text += "."
        else:
            # Fallback to first few sentences of context
            answer_text = ". ".join(sentences[:2])
            if not answer_text.endswith("."):
                answer_text += "."

        # Add citations
        citation_str = " ".join(citations[:3])
        return f"{answer_text}\n\nSources: {citation_str}"
