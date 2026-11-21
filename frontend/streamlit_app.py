from __future__ import annotations  # Defers type annotation evaluation so `X | None` works cleanly on Python 3.9 with zero runtime cost.

# Streamlit UI for Dual RAG Explorer:
# - Triggers PDF ingestion (/ingest) into Hybrid RAG (BM25 + FAISS + Cross-Encoder)
#   and optionally Neo4j AuraDB KG RAG using your configured OpenAI API / LLM stack.
# - Queries both pipelines (/query) and compares answers, elapsed times, sources, and Cypher.
# Performance summary:
# - Uses @st.cache_data(ttl=15) for /health PDF list lookups so UI reruns don't spam backend requests.
# - Guards /ingest and /query behind button clicks and session_state result caching so Streamlit
#   widget reruns keep prior answers visible without re-running expensive LLM/AuraDB calls.

import os  # Allows optional runtime override of OPENAI_API_KEY in the current Streamlit process if entered in UI.
from typing import Optional  # Explicit Optional type for Python 3.9 compatibility.
import requests  # Lightweight synchronous HTTP client for calling FastAPI endpoints; low overhead.
import streamlit as st  # Interactive Python web UI framework.

# Default FastAPI backend base URL; override via API_BASE_URL env var if needed.
API_BASE = os.getenv("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")


@st.cache_data(ttl=15, show_spinner=False)
def fetch_health(api_base: str) -> dict:
    # Cache /health for 15s to avoid redundant HTTP calls across Streamlit UI interactions.
    resp = requests.get(f"{api_base}/health", timeout=10)
    resp.raise_for_status()
    return resp.json()


def main() -> None:
    st.set_page_config(
        page_title="Dual RAG Explorer (Streamlit)",
        layout="wide",
    )
    st.title("Dual RAG Explorer — Hybrid RAG + Neo4j AuraDB KG RAG")
    st.caption(f"Backend API: `{API_BASE}` | Track A: BM25 + FAISS + Cross-Encoder | Track B: Neo4j AuraDB + LLM")

    # Optional sidebar OpenAI API key override (normally loaded from project root .env by backend).
    # Performance note: Only sets os.environ when non-empty; avoids touching credentials unnecessarily.
    with st.sidebar:
        st.header("Ingestion & API Settings")
        api_key_override = st.text_input(
            "OpenAI API Key (optional override)",
            type="password",
            help="Leave blank to use OPENAI_API_KEY loaded from live_demo_rag/.env",
        )
        if api_key_override.strip():
            os.environ["OPENAI_API_KEY"] = api_key_override.strip()

        build_graph = st.checkbox("Build Neo4j Graph on Ingest (Track B)", value=False)
        max_chunks = st.number_input("Max KG Chunks", min_value=1, max_value=200, value=10, step=1)
        ingest_first = st.checkbox("Run Ingestion Before Query", value=False)

    # Fetch available PDFs from FastAPI /health.
    available_pdfs = ["spotify_web_app_architecture.pdf"]
    try:
        health_data = fetch_health(API_BASE)
        pdfs = health_data.get("available_pdfs") or []
        if pdfs:
            available_pdfs = pdfs
    except Exception as exc:
        st.warning(f"Could not reach FastAPI at `{API_BASE}/health`: {exc}")

    default_idx = (
        available_pdfs.index("spotify_web_app_architecture.pdf")
        if "spotify_web_app_architecture.pdf" in available_pdfs
        else 0
    )

    col_pdf, col_ingest = st.columns([3, 1])
    with col_pdf:
        selected_pdf = st.selectbox("Select PDF from `data/`", options=available_pdfs, index=default_idx)
    with col_ingest:
        st.write("")  # Vertical spacer for button alignment.
        st.write("")
        ingest_clicked = st.button("Ingest PDF", use_container_width=True)

    # Persist last ingestion and query outputs in session_state so UI reruns don't wipe results.
    if "ingest_result" not in st.session_state:
        st.session_state.ingest_result = None
    if "query_result" not in st.session_state:
        st.session_state.query_result = None

    if ingest_clicked:
        # Runs Track A (FAISS + BM25) and optionally Track B (LLMGraphTransformer -> Neo4j AuraDB).
        with st.spinner(f"Ingesting `{selected_pdf}` (build_graph={build_graph})..."):
            try:
                resp = requests.post(
                    f"{API_BASE}/ingest",
                    json={
                        "pdf_filename": selected_pdf,
                        "build_graph": build_graph,
                        "max_chunks": int(max_chunks),
                    },
                    timeout=300,
                )
                data = resp.json()
                if resp.status_code != 200:
                    st.error(f"Ingest failed: {data.get('detail', resp.text)}")
                else:
                    st.session_state.ingest_result = data
                    fetch_health.clear()
            except Exception as exc:
                st.error(f"Ingest error: {exc}")

    if st.session_state.ingest_result:
        ing = st.session_state.ingest_result
        a_info = ing.get("track_a_hybrid_rag", {})
        b_info = ing.get("track_b_kg_rag", {})
        b_msg = "skipped" if b_info.get("skipped") else f"{b_info.get('elapsed_sec', '?')}s"
        st.success(
            f"Ingested `{ing.get('pdf')}` — "
            f"Track A: {a_info.get('stats', {}).get('chunks', '?')} chunks in {a_info.get('elapsed_sec', '?')}s | "
            f"Track B: {b_msg}"
        )

    question = st.text_input(
        "Ask a question across both RAG pipelines",
        value="Which databases, protocols, and services connect with the frontend and backend?",
    )
    query_clicked = st.button("Query Both Pipelines", type="primary")

    if query_clicked:
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Querying Hybrid RAG and Neo4j AuraDB KG RAG..."):
                try:
                    resp = requests.post(
                        f"{API_BASE}/query",
                        json={
                            "question": question.strip(),
                            "ingest_first": ingest_first,
                            "pdf_filename": selected_pdf,
                            "build_graph": build_graph,
                            "max_chunks": int(max_chunks),
                        },
                        timeout=300,
                    )
                    data = resp.json()
                    if resp.status_code != 200:
                        st.error(f"Query failed: {data.get('detail', resp.text)}")
                    else:
                        st.session_state.query_result = data
                except Exception as exc:
                    st.error(f"Query error: {exc}")

    if st.session_state.query_result:
        qres = st.session_state.query_result
        track_a = qres.get("track_a_hybrid_rag", {})
        track_b = qres.get("track_b_kg_rag", {})

        left_col, right_col = st.columns(2)

        with left_col:
            st.subheader("Track A: Hybrid RAG (BM25 + FAISS + Cross-Encoder)")
            st.caption(f"Latency: `{track_a.get('elapsed_sec', '?')}s`")
            st.markdown(track_a.get("answer") or "No answer returned.")

            sources = track_a.get("sources") or []
            if sources:
                with st.expander("Top Reranked Sources"):
                    for src in sources:
                        st.write(f"**Page {src.get('page', '?')}:** {src.get('preview', '')}...")

        with right_col:
            st.subheader("Track B: Knowledge Graph RAG (Neo4j AuraDB)")
            st.caption(f"Latency: `{track_b.get('elapsed_sec', '?')}s`")
            st.markdown(track_b.get("answer") or "No answer returned.")

            cypher_text = track_b.get("cypher") or ""
            if cypher_text:
                with st.expander("Generated Cypher Query", expanded=True):
                    st.code(cypher_text, language="cypher")


if __name__ == "__main__":
    main()