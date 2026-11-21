from __future__ import annotations  # Defers type annotations so `X | None` works cleanly on Python 3.9 with zero runtime overhead.

# Track A: Hybrid RAG (BM25 lexical + FAISS dense embeddings + local Cross-Encoder reranker).
# Performance summary:
# - Ingestion: Embeds chunks into FAISS (main cost/time) and saves both FAISS and a lightweight
#   chunks pickle file on disk; BM25 builds in RAM in milliseconds with zero API cost.
# - Reload (--no-ingest): Reloads persisted FAISS index + chunks in ~50ms-200ms and rebuilds BM25
#   in memory without re-calling OpenAI embeddings.
# - Querying: Pulls top candidate_k from BM25 + FAISS, deduplicates, reranks locally via Cross-Encoder,
#   and synthesizes a grounded answer in ~1s-3s without graph traversal.

import pickle  # Standard-library serializer for persisting split chunks alongside FAISS; zero extra install cost.
from pathlib import Path  # Cross-platform path management; zero runtime overhead.
from typing import Optional  # Explicit typing helper for Python 3.9 compatibility.

from langchain_community.document_loaders import PyPDFLoader  # Parses PDF pages into Document objects; memory scales with pages.
from langchain_community.retrievers import BM25Retriever  # Keyword/exact-term scorer; runs in RAM, no API cost.
from langchain_community.vectorstores import FAISS  # Dense vector similarity index; supports local disk persistence.
from langchain_core.output_parsers import StrOutputParser  # Converts LLM response into plain string; negligible cost.
from langchain_core.prompts import ChatPromptTemplate  # Prompt template for grounded answers; negligible cost.
from langchain_openai import ChatOpenAI, OpenAIEmbeddings  # Remote OpenAI API calls; primary source of latency & token spend.
from langchain_text_splitters import RecursiveCharacterTextSplitter  # Splits text on CPU; controls chunk count & prompt tokens.
from sentence_transformers import CrossEncoder  # Jointly scores (query, chunk) pairs for higher ranking precision.


class HybridRAGPipeline:
    def __init__(
        self,
        index_dir: Path | str,
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L6-v2",
        candidate_k: int = 8,
        top_n: int = 5,
        chunk_size: int = 800,
        chunk_overlap: int = 150,
        chat_model: str = "gpt-4o-mini",
        embedding_model: str = "text-embedding-3-small",
    ):
        # Directory storing persisted FAISS index files and serialized chunks for fast reloads.
        self.index_dir = Path(index_dir)
        # Path for persisting split chunks so BM25 can be reconstructed after process restarts (--no-ingest).
        self.chunks_cache_path = self.index_dir / "chunks.pkl"
        # Initialize embedding client once; reusing one instance avoids repeated setup overhead.
        self.embeddings = OpenAIEmbeddings(model=embedding_model)
        # temperature=0 reduces randomness and keeps answers grounded in retrieved evidence.
        self.llm = ChatOpenAI(model=chat_model, temperature=0)
        # Load Cross-Encoder once during __init__; reloading per query would add seconds of model-load overhead.
        self.reranker = CrossEncoder(reranker_model)
        # Number of candidates pulled per retriever before deduplication and Cross-Encoder scoring.
        self.candidate_k = candidate_k
        # Number of top reranked chunks sent to the LLM prompt (balances accuracy against prompt token cost).
        self.top_n = top_n
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        # In-memory document chunks used to build/query BM25.
        self.chunks = []
        self.vectorstore: Optional[FAISS] = None
        self.bm25_retriever: Optional[BM25Retriever] = None

    def ingest_pdf(self, pdf_path: Path | str) -> dict:
        # Load PDF pages from disk.
        loader = PyPDFLoader(str(pdf_path))
        pages = loader.load()

        # Split pages into overlapping chunks so sentences/lists aren't cut off at boundaries.
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        self.chunks = splitter.split_documents(pages)

        # Build FAISS dense index (main ingestion API time & token cost) and persist to disk.
        self.vectorstore = FAISS.from_documents(self.chunks, self.embeddings)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.vectorstore.save_local(str(self.index_dir))

        # Persist split chunks on disk so --no-ingest runs can reload BM25 immediately without re-ingesting.
        with self.chunks_cache_path.open("wb") as f:
            pickle.dump(self.chunks, f)

        # Build BM25 sparse keyword index in RAM from the split chunks.
        self.bm25_retriever = BM25Retriever.from_documents(self.chunks)
        self.bm25_retriever.k = self.candidate_k

        return {
            "pages": len(pages),
            "chunks": len(self.chunks),
            "faiss_dir": str(self.index_dir),
        }

    def _ensure_loaded_from_disk(self) -> bool:
        # Attempt to reload a previously saved FAISS index + chunks cache when ingest_pdf() wasn't called.
        # Performance note: Avoids re-embedding the entire PDF on every --no-ingest query run.
        faiss_file = self.index_dir / "index.faiss"
        if not faiss_file.exists() or not self.chunks_cache_path.exists():
            return False

        if self.vectorstore is None:
            # Safe here because index_dir is generated locally by this pipeline in backend/storage/faiss_index.
            self.vectorstore = FAISS.load_local(
                str(self.index_dir),
                self.embeddings,
                allow_dangerous_deserialization=True,
            )

        if not self.chunks:
            with self.chunks_cache_path.open("rb") as f:
                self.chunks = pickle.load(f)

        if self.bm25_retriever is None and self.chunks:
            self.bm25_retriever = BM25Retriever.from_documents(self.chunks)
            self.bm25_retriever.k = self.candidate_k

        return self.vectorstore is not None and self.bm25_retriever is not None

    def _rerank_candidates(self, query: str, candidates):
        if not candidates:
            return []

        # Jointly score (query, chunk_text) pairs; latency scales with candidate count and text length.
        pairs = [[query, doc.page_content] for doc in candidates]
        scores = self.reranker.predict(pairs)

        # Sort highest Cross-Encoder score first and return only top_n chunks.
        ranked = sorted(
            zip(candidates, scores),
            key=lambda item: float(item[1]),
            reverse=True,
        )
        return [doc for doc, _ in ranked[: self.top_n]]

    def _hybrid_retrieve(self, query: str):
        # If indexes aren't in memory yet, reload them from disk before failing.
        if self.vectorstore is None or self.bm25_retriever is None:
            loaded = self._ensure_loaded_from_disk()
            if not loaded:
                raise RuntimeError("Hybrid RAG index not ready. Ingest a PDF first.")

        # Stage 1a: Dense semantic search via FAISS.
        dense_docs = self.vectorstore.as_retriever(
            search_kwargs={"k": self.candidate_k}
        ).invoke(query)

        # Stage 1b: Sparse keyword search via BM25.
        self.bm25_retriever.k = self.candidate_k
        sparse_docs = self.bm25_retriever.invoke(query)

        # Deduplicate merged candidates before reranking to avoid redundant Cross-Encoder scoring.
        seen = set()
        candidates = []
        for doc in dense_docs + sparse_docs:
            key = (doc.page_content, doc.metadata.get("page"))
            if key not in seen:
                seen.add(key)
                candidates.append(doc)

        # Stage 2: Rerank deduplicated candidates locally and keep top_n highest-relevance chunks.
        return self._rerank_candidates(query, candidates)

    @staticmethod
    def _format_docs(docs) -> str:
        # Format reranked chunks with page numbers for clear grounding and traceability.
        return "\n\n---\n\n".join(
            f"[Page {d.metadata.get('page', '?')}] {d.page_content}" for d in docs
        )

    def ask(self, question: str) -> dict:
        top_docs = self._hybrid_retrieve(question)
        prompt = ChatPromptTemplate.from_template(
            """Answer the question using the provided context. Summarize all relevant details found
(such as databases, protocols, services, or components) even if phrased slightly differently.
Only say you don't know if the context contains no relevant information at all.

Context:
{context}

Question: {question}
"""
        )
        chain = prompt | self.llm | StrOutputParser()
        answer = chain.invoke({"context": self._format_docs(top_docs), "question": question})
        return {
            "answer": answer,
            "sources": [
                {
                    "page": d.metadata.get("page", "?"),
                    "preview": d.page_content[:180],
                }
                for d in top_docs
            ],
        }