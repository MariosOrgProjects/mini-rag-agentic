"""Tests for CLI module."""

from pathlib import Path

import pytest
from click.testing import CliRunner

from mini_rag.cli_command_handler import main


@pytest.fixture
def cli_runner() -> CliRunner:
    """Create a CLI test runner."""
    return CliRunner()


class TestCLIIndex:
    """Tests for index command."""

    def test_cli_index_missing_corpus(self, cli_runner: CliRunner) -> None:
        """Test index command without required corpus option."""
        result = cli_runner.invoke(main, ["index"])

        assert result.exit_code != 0
        assert "Missing option" in result.output or "required" in result.output.lower()

    def test_cli_index_success(
        self, cli_runner: CliRunner, sample_pdf_path: Path
    ) -> None:
        """Test successful indexing."""
        result = cli_runner.invoke(
            main, ["index", "--corpus", str(sample_pdf_path.parent), "--force"]
        )

        assert result.exit_code == 0
        assert "Indexed" in result.output


class TestCLIQuery:
    """Tests for query command."""

    def test_cli_query_no_corpus(self, cli_runner: CliRunner, temp_dir: Path) -> None:
        """Test query command with non-existent corpus."""
        result = cli_runner.invoke(
            main, ["query", "Test question", "--corpus", str(temp_dir / "nonexistent")]
        )

        assert result.exit_code != 0

    def test_cli_query_missing_argument(self, cli_runner: CliRunner) -> None:
        """Test query without question argument."""
        result = cli_runner.invoke(main, ["query"])

        assert result.exit_code != 0
        assert "Missing argument" in result.output or "QUESTION" in result.output

    def test_cli_query_success(
        self, cli_runner: CliRunner, sample_pdf_path: Path
    ) -> None:
        """Test successful query."""
        result = cli_runner.invoke(
            main,
            [
                "query",
                "What is in this document?",
                "--corpus",
                str(sample_pdf_path.parent),
            ],
        )

        assert result.exit_code == 0
