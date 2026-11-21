# Multi-purpose image running FastAPI (port 8000) and Streamlit (port 8501) for Dual RAG.
# Performance summary:
# - Installs all required Python packages in a single layer with --no-cache-dir to avoid
#   bloating the Docker image with wheel caches.
# - Sets HF_HOME=/app/.cache/huggingface so the Cross-Encoder model caches cleanly inside the container.
# - Uses `exec` form CMD for clean signal handling (SIGTERM/SIGINT) and zero shell wrapper overhead.

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface \
    API_BASE_URL=http://backend:8000

WORKDIR /app

# Install minimal system libraries needed by PDF/scientific/FAISS packages; clean apt lists immediately.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt if present, then explicitly install all required Dual RAG + Streamlit packages.
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip \
    && if [ -s /app/requirements.txt ]; then pip install -r /app/requirements.txt; fi \
    && pip install \
        "fastapi>=0.110.0" \
        "uvicorn[standard]>=0.29.0" \
        "pydantic>=2.7.0" \
        "python-dotenv>=1.0.0" \
        "pypdf>=4.0.0" \
        "langchain>=0.3.0" \
        "langchain-community>=0.3.0" \
        "langchain-core>=0.3.0" \
        "langchain-openai>=0.2.0" \
        "langchain-text-splitters>=0.3.0" \
        "langchain-neo4j>=0.1.0" \
        "langchain-experimental>=0.3.0" \
        "neo4j>=5.18.0" \
        "faiss-cpu>=1.8.0" \
        "rank-bm25>=0.2.2" \
        "sentence-transformers>=2.6.0" \
        "streamlit>=1.35.0" \
        "requests>=2.31.0"

# Copy application code, frontend UI, and sample PDF data into /app.
COPY backend /app/backend
COPY frontend /app/frontend
COPY data /app/data

# Create storage and HuggingFace cache directories with predictable permissions.
RUN mkdir -p /app/backend/storage/faiss_index /app/.cache/huggingface

EXPOSE 8000 8501

# Default command starts FastAPI; docker-compose overrides CMD for the Streamlit service.
CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]