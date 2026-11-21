# Centralized configuration for paths, PDF storage, Neo4j AuraDB, and retrieval/LLM parameters.
# Performance summary:
# - Centralizing defaults in config.py avoids hardcoding filenames across routes and CLI flags.
# - Reading .env once at import time keeps environment lookups fast and consistent across modules.

import os  # Reads environment variables (API keys, Neo4j AuraDB credentials, optional overrides).
from pathlib import Path  # Safe path construction across macOS/Linux/WSL2; zero runtime overhead.
from dotenv import load_dotenv  # Loads .env values into os.environ once at startup.

# Project root is live_demo_rag/ (two levels above backend/app/config.py).
PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = PROJECT_ROOT / "backend"
DATA_DIR = PROJECT_ROOT / "data"
FAISS_INDEX_DIR = BACKEND_DIR / "storage" / "faiss_index"

# Load .env from project root so CLI and FastAPI (uvicorn) share identical settings.
load_dotenv(PROJECT_ROOT / ".env")


class Settings:
    # Default PDF inside data/ set to spotify_web_app_architecture.pdf; override via DEFAULT_PDF_NAME
    # in .env or per request/CLI flag without editing code.
    default_pdf_path: Path = DATA_DIR / os.getenv(
        "DEFAULT_PDF_NAME", "spotify_web_app_architecture.pdf"
    )
    faiss_index_dir: Path = FAISS_INDEX_DIR

    # OpenAI model settings (used by Hybrid RAG embeddings/answer + KG RAG extraction/Cypher QA).
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    chat_model: str = os.getenv("CHAT_MODEL", "gpt-4o-mini")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

    # Hybrid RAG (BM25 + FAISS + Cross-Encoder) tuning parameters.
    # Smaller chunk_size = finer retrieval granularity; larger chunk_size = fewer chunks to embed.
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "800"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "150"))
    candidate_k: int = int(os.getenv("CANDIDATE_K", "6"))
    top_n: int = int(os.getenv("TOP_N", "4"))
    reranker_model: str = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L6-v2")

    # Neo4j AuraDB connection parameters (Track B: Knowledge Graph RAG).
    neo4j_uri: str = os.getenv("NEO4J_URI", "")
    neo4j_username: str = os.getenv("NEO4J_USERNAME", "neo4j")
    neo4j_password: str = os.getenv("NEO4J_PASSWORD", "")
    neo4j_database: str = os.getenv("NEO4J_DATABASE", "neo4j")


settings = Settings()