"""Command-line interface for Mini RAG."""

from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from mini_rag import __version__
from mini_rag.agent import Agent
from mini_rag.configuration import Settings
from mini_rag.exceptions import MiniRAGError

console = Console()


def print_error(message: str) -> None:
    """Print an error message."""
    console.print(f"[red]Error:[/red] {message}")


def print_success(message: str) -> None:
    """Print a success message."""
    console.print(f"[green]✓[/green] {message}")


def print_info(message: str) -> None:
    """Print an info message."""
    console.print(f"[blue]ℹ[/blue] {message}")


@click.group()
@click.version_option(version=__version__, prog_name="mini-rag")
def main() -> None:
    """Mini RAG - A lightweight agentic Q&A system over PDF documents."""


@main.command()
@click.option(
    "--corpus",
    "-c",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to corpus directory or single PDF file.",
)
@click.option(
    "--force",
    "-f",
    is_flag=True,
    default=False,
    help="Force re-indexing even if index exists.",
)
def index(corpus: Path, force: bool) -> None:
    """Index PDF documents for searching. Command that needs to be executed before querying."""
    try:
        settings = Settings()

        with console.status("[bold blue]Indexing documents..."):
            agent = Agent(settings)
            agent.initialize(corpus_path=corpus, force_reindex=force)

        stats = agent.get_stats()
        print_success(
            f"Indexed {stats['chunk_count']} chunks from "
            f"{stats['document_count']} document(s)"
        )
        print_info(f"Embedding backend: {stats['embedding_backend']}")

    except MiniRAGError as e:
        print_error(str(e))
        raise SystemExit(1)
    except Exception as e:
        print_error(f"Unexpected error: {e}")
        raise SystemExit(1)


@main.command()
@click.argument("question", type=str)
@click.option(
    "--corpus",
    "-c",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Path to corpus directory or single PDF file.",
)
@click.option(
    "--top-k",
    "-k",
    type=int,
    default=None,
    help="Number of chunks to retrieve (default: 3).",
)
def query(
    question: str,
    corpus: Path | None,
    top_k: int | None,
) -> None:
    """Ask a question about the indexed documents.

    QUESTION: The question to ask about the documents.

    Example:
        mini-rag query "What is the purpose of this document?"
        mini-rag query "Who is the author?" --corpus ./corpus --top-k 5
    """
    try:
        settings = Settings()
        if top_k:
            settings.top_k = top_k

        # Initialize agent (loads existing index or creates new)
        with console.status("[bold blue]Initializing..."):
            agent = Agent(settings)
            agent.initialize(corpus_path=corpus)

        # Query
        with console.status("[bold blue]Thinking..."):
            response = agent.query(question)

        if response.success:
            # Display answer
            console.print()
            console.print(Panel(response.answer, title="Answer", border_style="green"))

            # Display sources (citations)
            if response.sources:
                console.print()
                table = Table(title="Sources")
                table.add_column("File", style="cyan")
                table.add_column("Chunk", justify="right")
                table.add_column("Score", justify="right", style="green")

                for source in response.sources:
                    table.add_row(
                        source["file"],
                        str(source["chunk_id"]),
                        f"{source['score']:.2f}",
                    )

                console.print(table)
        else:
            print_error(response.error or "Unknown error")
            raise SystemExit(1)

    except MiniRAGError as e:
        print_error(str(e))
        raise SystemExit(1)
    except Exception as e:
        print_error(f"Unexpected error: {e}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
