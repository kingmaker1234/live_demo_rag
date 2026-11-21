// Frontend client calling FastAPI backend (/health, /ingest, /query) for Track A and Track B.
// Performance summary:
// - Uses same-origin relative URLs when served by FastAPI on port 8000 (zero cross-origin overhead)
//   and updates only targeted DOM nodes on response.

const API_BASE = window.location.port === "8000" ? "" : "[http://127.0.0.1:8000](http://127.0.0.1:8000)";

const pdfSelect = document.getElementById("pdfSelect");
const buildGraphCheck = document.getElementById("buildGraphCheck");
const maxChunksInput = document.getElementById("maxChunksInput");
const ingestBtn = document.getElementById("ingestBtn");
const questionInput = document.getElementById("questionInput");
const askBtn = document.getElementById("askBtn");
const statusBar = document.getElementById("statusBar");

const hybridMeta = document.getElementById("hybridMeta");
const hybridAnswer = document.getElementById("hybridAnswer");
const hybridSources = document.getElementById("hybridSources");
const kgMeta = document.getElementById("kgMeta");
const kgAnswer = document.getElementById("kgAnswer");
const kgCypher = document.getElementById("kgCypher");

function setBusy(busy, message = "") {
  // Prevent overlapping /ingest or /query calls while LLM or AuraDB requests are running.
  ingestBtn.disabled = busy;
  askBtn.disabled = busy;
  if (message) {
    statusBar.textContent = message;
  }
}

async function loadHealthAndPdfs() {
  // Lightweight GET /health check; populates available PDFs from backend data/ folder.
  try {
    const res = await fetch(`${API_BASE}/health`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();

    if (Array.isArray(data.available_pdfs) && data.available_pdfs.length > 0) {
      pdfSelect.innerHTML = "";
      for (const name of data.available_pdfs) {
        const opt = document.createElement("option");
        opt.value = name;
        opt.textContent = name;
        if (name === "spotify_web_app_architecture.pdf") {
          opt.selected = true;
        }
        pdfSelect.appendChild(opt);
      }
    }
    statusBar.textContent = `Connected to backend (${data.available_pdfs?.length || 0} PDFs in data/).`;
  } catch (err) {
    statusBar.textContent = `Backend offline. Start uvicorn on port 8000 first (${err.message}).`;
  }
}

async function handleIngest() {
  const pdfFilename = pdfSelect.value;
  const buildGraph = buildGraphCheck.checked;
  const maxChunks = Number(maxChunksInput.value) || 10;

  setBusy(true, `Ingesting ${pdfFilename} (buildGraph=${buildGraph})...`);
  try {
    const res = await fetch(`${API_BASE}/ingest`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        pdf_filename: pdfFilename,
        build_graph: buildGraph,
        max_chunks: maxChunks,
      }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || `HTTP ${res.status}`);

    const aSec = data.track_a_hybrid_rag?.elapsed_sec ?? 0;
    const aChunks = data.track_a_hybrid_rag?.stats?.chunks ?? 0;
    const bSec = data.track_b_kg_rag?.elapsed_sec ?? 0;
    const bSkipped = data.track_b_kg_rag?.skipped;

    statusBar.textContent =
      `Ingested ${data.pdf}: Track A (${aChunks} chunks in ${aSec}s)` +
      (bSkipped ? " | Track B skipped" : ` | Track B (${bSec}s)`);
  } catch (err) {
    statusBar.textContent = `Ingest error: ${err.message}`;
  } finally {
    setBusy(false);
  }
}

async function handleQuery() {
  const question = questionInput.value.trim();
  if (!question) {
    statusBar.textContent = "Please enter a question first.";
    return;
  }

  setBusy(true, "Querying Track A (Hybrid RAG) and Track B (Neo4j AuraDB KG RAG)...");
  try {
    const res = await fetch(`${API_BASE}/query`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question,
        ingest_first: false,
        pdf_filename: pdfSelect.value,
        build_graph: false,
      }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || `HTTP ${res.status}`);

    const a = data.track_a_hybrid_rag || {};
    const b = data.track_b_kg_rag || {};

    hybridMeta.textContent = `Latency: ${a.elapsed_sec ?? "?"}s (BM25 + FAISS + Cross-Encoder)`;
    hybridAnswer.textContent = a.answer || "No answer returned.";

    if (Array.isArray(a.sources) && a.sources.length > 0) {
      hybridSources.innerHTML =
        "<strong>Top Reranked Sources:</strong><br>" +
        a.sources.map((s) => `Page ${s.page}: ${s.preview}...`).join("<br>");
    } else {
      hybridSources.innerHTML = "";
    }

    kgMeta.textContent = `Latency: ${b.elapsed_sec ?? "?"}s (Neo4j AuraDB Cypher QA)`;
    kgAnswer.textContent = b.answer || "No answer returned.";
    kgCypher.textContent = b.cypher || "(No Cypher query captured)";

    statusBar.textContent = `Completed query in Track A (${a.elapsed_sec}s) and Track B (${b.elapsed_sec}s).`;
  } catch (err) {
    statusBar.textContent = `Query error: ${err.message}`;
  } finally {
    setBusy(false);
  }
}

ingestBtn.addEventListener("click", handleIngest);
askBtn.addEventListener("click", handleQuery);
questionInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter") handleQuery();
});

loadHealthAndPdfs();