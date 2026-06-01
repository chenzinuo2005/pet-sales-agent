FROM python:3.13-slim AS backend

WORKDIR /app

# System dependencies for torch + opencv
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files first for layer caching
COPY pyproject.toml uv.lock ./

# Install dependencies (no dev deps)
RUN pip install --no-cache-dir uv && \
    uv sync --frozen --no-dev && \
    rm -rf /root/.cache/uv

# Copy application code
COPY app/ ./app/
COPY data/ ./data/
COPY scripts/ ./scripts/
COPY resources/ ./resources/
COPY main.py ./

# Pre-create resource directories if they don't exist
RUN mkdir -p resources/models resources/outputs resources/chroma_db

EXPOSE 8000

ENV PYTHONUNBUFFERED=1
ENV LOG_FORMAT=json

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/v1/health')"

CMD ["uv", "run", "python", "-m", "app.main", "serve", "--host", "0.0.0.0", "--port", "8000"]
