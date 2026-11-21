from __future__ import annotations  # Defers annotation evaluation so `Optional[...]` and type hints work cleanly on Python 3.9 with zero runtime cost.

# Backend entry point exposing FastAPI endpoints, serving frontend UI on port 8000,
# and providing a CLI runner for Track A (Hybrid RAG) and Track B (Neo4j AuraDB KG RAG).
# Performance summary:
# - Lazy initialization defers Cross-Encoder loading and AuraDB connections until first use,
#   keeping startup, static UI serving, and /health checks fast.
# - Module-level singletons reuse loaded pipelines across requests instead of re-initializing per call.
# - Serving frontend/ directly via StaticFiles on port 8000 eliminates a second dev server process.

import argparse  # Parses CLI flags (--pdf, --question, --no-ingest, --no-graph, --max-chunks); zero runtime overhead.
import time  # Measures wall-clock duration of ingestion and query stages.
import warnings  # Suppresses noisy Pydantic serialization UserWarnings from LLMGraphTransformer; zero runtime cost.
from pathlib import Path  # Safe file path resolution relative to project root and DATA_DIR.
from typing import Optional  # Explicit Optional type so Pydantic v2 on Python 3.9 evaluates model fields cleanly.

from fastapi import FastAPI, HTTPException  # Lightweight async web API framework.
from fastapi.middleware.cors import CORSMiddleware  # Enables cross-origin requests if frontend is opened separately.
from fastapi.staticfiles import StaticFiles  # Serves static HTML/JS frontend directly from FastAPI on port 8000.
from pydantic import BaseModel, Field  # Validates request bodies cleanly at API boundaries.

from backend.app.config import DATA_DIR, PROJECT_ROOT, settings  # Shared paths, Neo4j AuraDB credentials, and RAG settings.
from backend.app.pipelines.hybrid_rag import HybridRAGPipeline  # Track A: BM25 + FAISS + Cross-Encoder.
from backend.app.pipelines.kg_rag import KnowledgeGraphRAGPipeline  # Track B: Neo4j AuraDB KG RAG.

# Filter non-fatal Pydantic v2 serialization warnings emitted during structured graph extraction.
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

FRONTEND_DIR = PROJECT_ROOT / "frontend"

app = FastAPI(
    title="Dual RAG API (Hybrid RAG + Neo4j AuraDB KG RAG)",
    version="0.4.0",
)

# Allow browser clients to call FastAPI endpoints; adds negligible per-request overhead.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cached pipeline singletons avoid reloading model weights or reopening Neo4j connections per call.
_hybrid_pipeline: Optional[HybridRAGPipeline] = None
_kg_pipeline: Optional[KnowledgeGraphRAGPipeline] = None


def get_hybrid_pipeline() -> HybridRAGPipeline:
    # Lazily create HybridRAGPipeline on first use so /health and static UI loads stay instantaneous.
    global _hybrid_pipeline
    if _hybrid_pipeline is None:
        _hybrid_pipeline = HybridRAGPipeline(
            index_dir=settings.faiss_index_dir,
            reranker_model=settings.reranker_model,
            candidate_k=settings.candidate_k,
            top_n=settings.top_n,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            chat_model=settings.chat_model,
            embedding_model=settings.embedding_model,
        )
    return _hybrid_pipeline


def get_kg_pipeline() -> KnowledgeGraphRAGPipeline:
    # Lazily create KnowledgeGraphRAGPipeline and validate Neo4j AuraDB credentials.
    global _kg_pipeline
    if _kg_pipeline is None:
        if not settings.neo4j_uri or not settings.neo4j_password:
            raise HTTPException(
                status_code=400,
                detail="NEO4J_URI and NEO4J_PASSWORD must be set in .env for KG RAG.",
            )
        _kg_pipeline = KnowledgeGraphRAGPipeline(
            neo4j_uri=settings.neo4j_uri,
            neo4j_username=settings.neo4j_username,
            neo4j_password=settings.neo4j_password,
            neo4j_database=settings.neo4j_database,
            chat_model=settings.chat_model,
        )
    return _kg_pipeline


def _resolve_pdf_path(pdf_filename: Optional[str]) -> Path:
    # Resolve PDF inside data/ and block path traversal outside DATA_DIR.
    target_name = pdf_filename or settings.default_pdf_path.name
    resolved = (DATA_DIR / target_name).resolve()
    if DATA_DIR.resolve() not in resolved.parents and resolved != DATA_DIR.resolve():
        raise HTTPException(status_code=400, detail="Invalid PDF path.")
    if not resolved.exists() or not resolved.is_file():
        raise HTTPException(status_code=404, detail=f"PDF not found in data/: {target_name}")
    return resolved


class IngestRequest(BaseModel):
    pdf_filename: Optional[str] = Field(default=None, description="PDF filename inside data/")
    build_graph: bool = Field(default=True, description="Extract entities & write to Neo4j AuraDB")
    max_chunks: Optional[int] = Field(default=10, ge=1, le=200)


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Question for both RAG pipelines")
    ingest_first: bool = Field(default=False)
    pdf_filename: Optional[str] = Field(default=None)
    build_graph: bool = Field(default=False)
    max_chunks: Optional[int] = Field(default=10, ge=1, le=200)


@app.get("/health")
def health_check():
    # Lightweight liveness probe; lists available PDFs in data/ without invoking LLMs or AuraDB.
    available_pdfs = sorted(p.name for p in DATA_DIR.glob("*.pdf")) if DATA_DIR.exists() else []
    return {
        "status": "ok",
        "data_dir": str(DATA_DIR),
        "available_pdfs": available_pdfs,
        "hybrid_initialized": _hybrid_pipeline is not None,
        "kg_initialized": _kg_pipeline is not None,
    }


@app.post("/ingest")
def ingest_document(req: IngestRequest):
    # Runs Track A (BM25 + FAISS) and optionally Track B (Neo4j AuraDB KG) on the target PDF.
    pdf_path = _resolve_pdf_path(req.pdf_filename)

    hybrid = get_hybrid_pipeline()
    t0 = time.perf_counter()
    hybrid_stats = hybrid.ingest_pdf(pdf_path)
    hybrid_ingest_sec = round(time.perf_counter() - t0, 3)

    kg_stats = None
    kg_ingest_sec = 0.0
    if req.build_graph:
        kg = get_kg_pipeline()
        t1 = time.perf_counter()
        kg_stats = kg.ingest_pdf_to_aura(pdf_path, max_chunks=req.max_chunks)
        kg_ingest_sec = round(time.perf_counter() - t1, 3)

    return {
        "pdf": pdf_path.name,
        "track_a_hybrid_rag": {
            "stats": hybrid_stats,
            "elapsed_sec": hybrid_ingest_sec,
        },
        "track_b_kg_rag": {
            "skipped": not req.build_graph,
            "stats": kg_stats,
            "elapsed_sec": kg_ingest_sec,
        },
    }


@app.post("/query")
def query_both_pipelines(req: QueryRequest):
    # Queries both Hybrid RAG and Neo4j AuraDB KG RAG and returns answers + Cypher + stage timings.
    if req.ingest_first:
        ingest_document(
            IngestRequest(
                pdf_filename=req.pdf_filename,
                build_graph=req.build_graph,
                max_chunks=req.max_chunks,
            )
        )

    hybrid = get_hybrid_pipeline()
    kg = get_kg_pipeline()

    # Track A: BM25 + FAISS retrieval -> Cross-Encoder rerank -> LLM synthesis.
    t0 = time.perf_counter()
    hybrid_result = hybrid.ask(req.question)
    hybrid_sec = round(time.perf_counter() - t0, 3)

    # Track B: Natural language -> Cypher generation -> Neo4j AuraDB query -> LLM answer.
    t1 = time.perf_counter()
    kg_result = kg.ask(req.question)
    kg_sec = round(time.perf_counter() - t1, 3)

    return {
        "question": req.question,
        "track_a_hybrid_rag": {
            "answer": hybrid_result["answer"],
            "sources": hybrid_result["sources"],
            "elapsed_sec": hybrid_sec,
        },
        "track_b_kg_rag": {
            "answer": kg_result["answer"],
            "cypher": kg_result.get("cypher", ""),
            "elapsed_sec": kg_sec,
        },
    }


# Mount frontend directory at root "/" after API routes so /health, /ingest, and /query take priority.
# Performance note: StaticFiles serves index.html and app.js directly from port 8000 with zero extra server.
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")


def _run_cli():
    # Allows running ingestion and querying both RAG pipelines directly from the terminal.
    parser = argparse.ArgumentParser(description="Dual RAG CLI (Hybrid RAG + Neo4j AuraDB KG RAG)")
    parser.add_argument("--pdf", default=settings.default_pdf_path.name, help="PDF filename in data/")
    parser.add_argument(
        "--question",
        default="What components, services, and relationships make up the Spotify web app architecture?",
        help="Question to ask both pipelines",
    )
    parser.add_argument("--no-ingest", action="store_true", help="Skip PDF ingestion step")
    parser.add_argument(
        "--no-graph",
        action="store_true",
        help="Skip Neo4j AuraDB graph extraction during ingestion",
    )
    parser.add_argument("--max-chunks", type=int, default=10, help="Max chunks for KG extraction")
    args = parser.parse_args()

    if not args.no_ingest:
        ingest_res = ingest_document(
            IngestRequest(
                pdf_filename=args.pdf,
                build_graph=not args.no_graph,
                max_chunks=args.max_chunks,
            )
        )
        print("=== INGESTION SUMMARY ===")
        print(ingest_res)

    query_res = query_both_pipelines(
        QueryRequest(question=args.question, ingest_first=False)
    )

    print("\n=== TRACK A: HYBRID RAG (BM25 + FAISS + CROSS-ENCODER) ===")
    print(f"Elapsed: {query_res['track_a_hybrid_rag']['elapsed_sec']}s")
    print(query_res["track_a_hybrid_rag"]["answer"])

    print("\n=== TRACK B: KNOWLEDGE GRAPH RAG (NEO4J AURA) ===")
    print(f"Elapsed: {query_res['track_b_kg_rag']['elapsed_sec']}s")
    if query_res["track_b_kg_rag"].get("cypher"):
        print("Generated Cypher:")
        print(query_res["track_b_kg_rag"]["cypher"])
    print(query_res["track_b_kg_rag"]["answer"])


if __name__ == "__main__":
    _run_cli()