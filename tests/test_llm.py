"""Tests for LLM module."""

import pytest
from pytest_mock import MockerFixture

from mini_rag.configuration import Settings
from mini_rag.exceptions import MiniRAGError
from mini_rag.llm import LLMService, SimpleLLM


class TestLLMService:
    """Tests for LLMService class."""

    def test_llm_service_initialization(self, settings: Settings) -> None:
        """Test LLM service initialization."""
        llm = LLMService(settings)
        assert llm.base_url == settings.ollama_base_url
        assert llm.model == settings.llm_model

    def test_llm_service_is_available(
        self, settings: Settings, mocker: MockerFixture
    ) -> None:
        """Test availability check."""
        mock_response = mocker.Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"models": [{"name": "llama3.2:latest"}]}
        mocker.patch("requests.get", return_value=mock_response)

        settings.llm_model = "llama3.2"
        llm = LLMService(settings)
        assert llm.is_available()

    def test_llm_service_generate_success(
        self, settings: Settings, mocker: MockerFixture
    ) -> None:
        """Test successful generation."""
        mock_response = mocker.Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"response": "Hello, world!"}
        mocker.patch("requests.post", return_value=mock_response)

        llm = LLMService(settings)
        result = llm.generate("Say hello")
        assert result == "Hello, world!"

    def test_llm_service_generate_empty_prompt(self, settings: Settings) -> None:
        """Test generation with empty prompt."""
        llm = LLMService(settings)
        with pytest.raises(MiniRAGError, match="empty"):
            llm.generate("  ")

    def test_llm_service_generate_error(
        self, settings: Settings, mocker: MockerFixture
    ) -> None:
        """Test error handling."""
        import requests

        mocker.patch("requests.post", side_effect=requests.ConnectionError())

        llm = LLMService(settings)
        with pytest.raises(MiniRAGError, match="request failed"):
            llm.generate("Test prompt")


class TestSimpleLLM:
    """Tests for SimpleLLM fallback."""

    def test_simple_llm_empty_context(self) -> None:
        """Test with empty context."""
        llm = SimpleLLM()
        result = llm.generate_with_context(
            question="What is X?", context="  ", citations=[]
        )
        assert "No relevant information" in result

    def test_simple_llm_with_context(self) -> None:
        """Test with valid context."""
        llm = SimpleLLM()
        result = llm.generate_with_context(
            question="What is machine learning?",
            context="Machine learning is a type of AI.",
            citations=["[doc.pdf, Chunk 1]"],
        )
        assert len(result) > 0
        assert "[doc.pdf, Chunk 1]" in result
