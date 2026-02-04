# Use Python 3.11 slim image
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /app

# Copy project configuration - done before source code for dependency caching
COPY pyproject.toml ./

# Copy source code
COPY src/ ./src/

# Install the package and dependencies
RUN pip install --no-cache-dir -e .

# Create directories for runtime data
RUN mkdir -p /app/logs /app/vectorstore /app/corpus

# Set mini-rag CLI as entrypoint
ENTRYPOINT ["mini-rag"]
CMD ["--help"]
