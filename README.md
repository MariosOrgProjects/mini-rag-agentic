# Mini RAG Agentic Q&A System

A lightweight RAG (Retrieval-Augmented Generation) system with an agentic workflow for question-answering over PDF documents. Features a plan-retrieve-draft-cite loop with structured logging and citations.

## Features

- **PDF Processing**: Parse and chunk PDF documents with configurable chunk sizes
- **Dual Embedding Support**: Ollama embeddings (`nomic-embed-text`) or TF-IDF fallback
- **Agentic Workflow**: Plan → Retrieve → Draft → Cite execution loop
- **Citations**: Every answer includes source file names, page numbers, and chunk references
- **Structured Logging**: JSON-formatted query traces with latency metrics
- **CLI Interface**: Simple command-line tool for indexing and querying
- **Docker Support**: Containerized deployment with docker-compose

---

## Get Started

### Prerequisites

- Python >=3.10
    - for Windows users: Use [Software on Demand](https://btondemand.pfizer.com/software) or [Python](https://www.python.org/downloads/) to downloda
    - for Macos users:
    ```
    brew install python@3.11
    ```
- [Ollama](https://ollama.ai/) for embeddings and LLM (recommended)
- [Docker](https://docs.docker.com/engine/install/) (optional, for containerized deployment, will need Docker Engine for docker compose)

### 1. Clone the Repository

```bash
git clone git@github.com:MariosOrgProjects/mini-rag-agentic.git
```

### 2. Change Directory

```bash
cd mini-rag-agentic
```

### 3. Set Up Python Environment

- for **Macos** users:
    1. ```python -m venv venv```
    2. ```source venv/bin/activate```
    3. ```pip install -e ".[dev]"```

- for **Windows** users:
    1. ```python -m venv venv```
    2. ```source venv/Scripts/activate```
    3. ```pip install -e ".[dev]"```


### 4. Configure Environment Variables

Copy the example environment file and adjust as needed:


```bash
cp .env.example .env
```

### 5. Set Up Ollama

Below the instructions on how to install and start ollama on your local machine (required for the application)

- Download from [Ollama](https://ollama.ai/), then:
  ```
  ollama serve
  ollama pull nomic-embed-text
  ollama pull llama3.2
  ```

**Note:**
- When you properly install ollama then there is a command: ```ollama --help``` that shows all the options for ollama.
- To properly install ollama make sure that you are not connected to VPN
- If Ollama is unavailable, the system automatically falls back to TF-IDF embeddings and simple text extraction.


### 6. Add PDF Documents

Place your PDF files in the `corpus/` directory if you want to place new files:
```bash
cp /path/to/your/documents/*.pdf corpus/
```

>**Note:** already added pdf files from assignment

---

## CLI Usage

### Available Commands

| Command | Description |
|---------|-------------|
| `mini-rag index -c <path>` | Index PDF documents |
| `mini-rag index -c <path> --force` | Force re-index (rebuild) |
| `mini-rag query "<question>"` | Ask a question |
| `mini-rag query "<question>" --top-k N` | Ask with custom number of results (default: 5) |
| `mini-rag --help` | Show help / Possible options|

### Index Documents

- Before querying, you must index your documents:
    ```
    mini-rag index -c corpus/
    ```


- To force a complete re-index:
    ```
    mini-rag index -c corpus/ --force
    ```

- Or index a specific document:
    ```
    mini-rag index -c corpus/<file_name> --force
    ```

### Query Documents

- Query on document(s)
    ```
    mini-rag query "What is the purpose of informed consent?"
    ```

- To retrieve more or fewer chunks, use the `--top-k` option:
    - Retrieve top 5 most relevant chunks:
        ```
        mini-rag query "What is the purpose of informed consent?" --top-k 5
        ```

---

## Sample Queries and Responses

Below are three sample queries demonstrating the system's capabilities with citations:

### Sample Query 1: General Document Overview

**Command:**
```bash
mini-rag query "What is the main topic of these documents?"
```

**Answer:**
> Based on the provided context, it appears that the document is about Informed Consent in Clinical Research. The context provides information on various aspects of informed consent, including the clinical investigator's role, standardized language and format for consent forms, payment to research subjects, and potential concerns or questions related to informed consent.
>
> The U.S. Department of Health and Human Services Food and Drug Administration (FDA) is mentioned as a key authority on this topic. The FDA provides guidance for Institutional Review Boards (IRBs), clinical investigators, and sponsors regarding informed consent in clinical research.

**Sources:**
| File | Chunk | Score |
|------|-------|-------|
| ICD - Document 1.pdf | 23 | 0.67 |
| ICD - Document 1.pdf | 324 | 0.65 |
| ICD - Document 1.pdf | 0 | 0.64 |

---

### Sample Query 2: Specific Regulatory Requirements

**Command:**
```bash
mini-rag query "What are the FDA requirements for informed consent?"
```

**Answer:**
> According to the context, the FDA's informed consent requirements are set forth in 21 CFR part 50, specifically at sections 50.20 and 50.25 [ICD - Document 1.pdf, Page 7, Chunk 35].
>
> The informed consent process must satisfy the general requirements under 21 CFR 50.20 and include the relevant elements of informed consent described at 21 CFR 50.25. IRBs may recommend that investigators provide summaries to all subjects for their awareness [ICD - Document 1.pdf, Page 66, Chunk 612].
>
> Additionally, the context mentions requirements for witness signatures when subjects cannot read, as specified in 21 CFR 50.27(b)(2) [ICD - Document 1.pdf, Page 34, Chunk 236].

**Sources:**
| File | Chunk | Score |
|------|-------|-------|
| ICD - Document 1.pdf | 35 | 0.87 |
| ICD - Document 1.pdf | 612 | 0.82 |
| ICD - Document 1.pdf | 236 | 0.79 |

---

### Sample Query 3: Clinical Study Report Structure

**Command:**
```bash
mini-rag query "What is the structure of a clinical study report according to ICH E3?"
```

**Answer:**
> According to ICH E3, the structure of a clinical study report includes multiple sections that can be referenced under the same heading. The guidance outlines that topics not explicitly addressed should be placed in the most relevant section [E3 Structure - Document 2.pdf, Page 5, Chunk 19].
>
> Key structural elements include:
> - Efficacy and safety sections preceding results
> - Clinical pharmacology studies
> - Pharmacokinetic reports and assay validation reports
> - Patient-reported outcomes, images, and diagnostic tests should be well referenced in the CSR body [E3 Structure - Document 2.pdf, Page 7, Chunk 44]
>
> The ICH E3 is interpreted as guidance, not strict requirements - the specific sequence and grouping may be changed if alternatives are more logical for a particular study [E3 Structure - Document 2.pdf, Page 3, Chunk 10].

**Sources:**
| File | Chunk | Score |
|------|-------|-------|
| E3 Structure - Document 2.pdf | 19 | 0.79 |
| E3 Structure - Document 2.pdf | 44 | 0.76 |
| E3 Structure - Document 2.pdf | 10 | 0.73 |

---

## Run with local Docker

### Prerequisites
- Build Image via Docker Compose

  ```docker compose build```

    If there are issues and you want to start "clean", add `--no-cache` to the above.

- Build specific image
    - for main application: ```docker compose build main-app```
    - for tests execution: ```docker compose build test```


### Run Application Commands


1. Show help dialog
    ```
    docker compose run --rm main-app
    ```

2. Index documents
    ```
    docker compose run --rm main-app index -c /app/corpus
    ```

3. Force re-index of the document(s)
    ```
    docker compose run --rm main-app index -c /app/corpus --force
    ```

4. Query documents
    ```
    docker compose run --rm main-app query "What is informed consent?"
    ```


### Run Tests

- Run all tests with coverage report
    ```
    docker compose run --rm test
    ```


### Docker Notes

- Requires Ollama running on host machine (`ollama serve`)
- PDF corpus files are mounted from `./corpus/`
- Logs persist to `./logs/`
- Vector index persists to `./vectorstore/`
- Coverage reports saved to `./coverage_report/`
- if you want to open the report on the web:
    - for Macos users type in terminal: ```open coverage_report/index.html```
    - for Windows users type in terminal: ```start coverage_report/index.html```

---

## Manual Test Executions Commands


- Run all tests with coverage: ```pytest```

- Run with verbose output: ```pytest -v```

- Run specific test file: ```pytest tests/<file_name>.py```

- Quick test run: ```pytest -q```

- Generate HTML coverage report: ```pytest --cov-report=html```

- If you want to see report on web:
    - for Macos: ```open htmlcov/index.html```
    - for Windows: ```start htmlcov/index.html```


---

## Structured Logging

Each query generates a structured JSON log entry in `logs/queries.jsonl`:

```json
{
  "trace_id": "b967aa38-8ed8-4e53-8d4c-c93991aa614e",
  "question": "What is informed consent?",
  "plan": ["plan", "retrieve", "draft", "cite"],
  "retrieval": [
    {"file": "ICD - Document 1.pdf", "chunk_id": 35, "score": 0.87},
    {"file": "ICD - Document 1.pdf", "chunk_id": 612, "score": 0.82},
    {"file": "ICD - Document 1.pdf", "chunk_id": 236, "score": 0.79}
  ],
  "answer": "According to the context...",
  "citations": ["[ICD - Document 1.pdf, Page 7, Chunk 35]"],
  "draft_tokens": 156,
  "latency_ms": {"plan": 1, "retrieve": 45, "draft": 15234, "cite": 2, "total": 15282},
  "errors": []
}
```

---

## Assumptions & Limitations

### Assumptions

- PDF files contain extractable text (not scanned images)
- Documents are in English
- Ollama is running locally for best results
- Corpus size fits in memory for vector indexing

### Limitations

- **No OCR**: Scanned PDFs without text layers are not supported
- **English only**: No multi-language support
- **No conversation memory**: Each query is independent
- **No table/image extraction**: Only text content is processed
- **TF-IDF fallback**: Lower semantic understanding compared to Ollama embeddings
- **Chunk size trade-off**: Smaller chunks = better precision, larger = better context

### Known Issues

- First query after indexing may be slower due to model loading
- Very long documents may take time to index with Ollama embeddings

---


## Code Style and Linting

This project uses [Ruff](https://docs.astral.sh/ruff/) for linting and formatting.

### Ruff Commands

| Command | Description |
|---------|-------------|
| `ruff check .` | Check for linting errors |
| `ruff check . --fix` | Auto-fix linting errors |
| `ruff format .` | Format all Python files |
| `ruff format . --check` | Check formatting without changes |
| `ruff check . && ruff format .` | Full lint + format |


---

## Pre-commit Hooks

This project uses [pre-commit](https://pre-commit.com/) to run checks before each commit.

### Setup

```bash
# Install pre-commit hooks (one-time setup)
pre-commit install
```

### Usage

| Command | Description |
|---------|-------------|
| `pre-commit run --all-files` | Run all hooks on all files |
| `pre-commit run` | Run hooks on staged files only |
| `pre-commit autoupdate` | Update hooks to latest versions |


**Note:**
- Pre-commit hooks run automatically on `git commit`. If any hook fails, the commit is aborted and you must fix the issues before committing again.
- To skip pre-commit hooks for a specific commit, use:
    - git commit -m "Your commit message" --no-verify

---

## Troubleshooting

#### For Windows users
