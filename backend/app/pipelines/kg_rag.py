from __future__ import annotations  # Defers type annotation evaluation so `X | None` works cleanly on Python 3.9 with zero runtime overhead.

# Track B: Schema-constrained Knowledge Graph RAG backed by Neo4j AuraDB.
# Performance summary:
# - Ingestion: Runs LLM entity/relationship extraction per chunk and writes nodes/edges over TLS
#   to Neo4j AuraDB; capping max_chunks controls extraction time and LLM token spend.
# - Querying: Generates Cypher using undirected pair matching `(a)-[r]-(b)` with `LIMIT 40`
#   to avoid rigid starting-node or direction misses, executes over TLS against AuraDB, synthesizes
#   the answer, and surfaces the generated Cypher query without extra LLM calls.

from pathlib import Path  # Cross-platform file path handling; zero runtime overhead.
from typing import Any, Optional, Tuple, Union  # Python 3.9 typing helpers.

from langchain_community.document_loaders import PyPDFLoader  # Loads PDF pages into memory.
from langchain_core.callbacks import BaseCallbackHandler  # Captures generated Cypher from chain events without extra LLM calls.
from langchain_core.prompts import PromptTemplate  # Formats Cypher generation instructions; negligible overhead.
from langchain_experimental.graph_transformers import LLMGraphTransformer  # Extracts nodes & edges via LLM structured output.
from langchain_neo4j import GraphCypherQAChain, Neo4jGraph  # Connects to Neo4j AuraDB and executes Cypher QA.
from langchain_openai import ChatOpenAI  # LLM used for graph extraction, Cypher generation, and final answer synthesis.
from langchain_text_splitters import RecursiveCharacterTextSplitter  # Splits text on CPU; controls chunk & LLM call count.

DEFAULT_ALLOWED_NODES = [
    "Frontend",
    "Backend",
    "Service",
    "Database",
    "Protocol",
]

DEFAULT_ALLOWED_RELATIONSHIPS: list[Tuple[str, str, str]] = [
    ("Frontend", "COMMUNICATES_WITH", "Backend"),
    ("Frontend", "USES_PROTOCOL", "Protocol"),
    ("Backend", "INVOKES", "Service"),
    ("Backend", "USES_PROTOCOL", "Protocol"),
    ("Service", "READS_FROM", "Database"),
    ("Service", "WRITES_TO", "Database"),
    ("Service", "COMMUNICATES_WITH", "Service"),
    ("Service", "USES_PROTOCOL", "Protocol"),
]

# Undirected `(a)-[r]-(b)` matching with `LIMIT 40` avoids empty results caused by mandatory
# `Frontend` anchors or reversed edge directions while keeping Neo4j result sizes compact.
CYPHER_GENERATION_TEMPLATE = """Task: Generate a read-only Cypher query for Neo4j using ONLY the provided schema.
Rules:
- Nodes store their name in the `id` property (e.g., `n.id`). Do NOT reference `n.name`.
- Allowed node labels: Frontend, Backend, Service, Database, Protocol.
- Allowed relationship types: COMMUNICATES_WITH, USES_PROTOCOL, INVOKES, READS_FROM, WRITES_TO.
- Do NOT require `(f:Frontend)` as a mandatory anchor; match any connected pair `(a)-[r]-(b)` across Frontend, Backend, Service, Database, and Protocol.
- Return `DISTINCT labels(a)[0] AS source_type, a.id AS source, type(r) AS relation, labels(b)[0] AS target_type, b.id AS target LIMIT 40`.

Schema:
{schema}

Question: {query}
Cypher Query:"""


class _CypherCaptureCallback(BaseCallbackHandler):
    # Lightweight in-memory callback capturing the generated Cypher string during chain execution.
    def __init__(self):
        self.cypher: Optional[str] = None

    def on_text(self, text: str, **kwargs: Any) -> None:
        cleaned = text.strip()
        if cleaned and ("MATCH " in cleaned or "OPTIONAL MATCH " in cleaned or "RETURN " in cleaned):
            # Strip markdown code fences so the UI/CLI displays clean Cypher.
            lines = [
                line for line in cleaned.splitlines()
                if not line.strip().startswith("```") and line.strip().lower() != "cypher"
            ]
            if lines:
                self.cypher = "\n".join(lines).strip()


class KnowledgeGraphRAGPipeline:
    def __init__(
        self,
        neo4j_uri: str,
        neo4j_username: str,
        neo4j_password: str,
        neo4j_database: str = "",
        chat_model: str = "gpt-4o-mini",
        allowed_nodes: Optional[list[str]] = None,
        allowed_relationships: Optional[list[Union[Tuple[str, str, str], str]]] = None,
        strict_mode: bool = True,
        additional_instructions: str = (
            "Extract clean, canonical architecture component names (e.g., 'Next.js', 'NestJS', "
            "'PostgreSQL', 'Redis', 'Elasticsearch', 'ClickHouse', 'S3', 'HLS.js', 'OAuth2 / OIDC', "
            "'REST', 'gRPC', 'Kafka'). Map each item strictly to one of the allowed node labels: "
            "Frontend, Backend, Service, Database, or Protocol."
        ),
    ):
        # Connect to Neo4j AuraDB over TLS (neo4j+s://).
        # Performance note: Network round-trips occur during schema refresh and query execution.
        self.graph = Neo4jGraph(
            url=neo4j_uri,
            username=neo4j_username,
            password=neo4j_password,
            database=neo4j_database or None,
        )

        # temperature=0 minimizes randomness so entity extraction and Cypher queries stay deterministic.
        self.llm = ChatOpenAI(model=chat_model, temperature=0)

        # strict_mode=True restricts extracted nodes/edges to the defined architecture schema.
        # node_properties=False and relationship_properties=False keep extraction prompts lean and fast.
        self.transformer = LLMGraphTransformer(
            llm=self.llm,
            allowed_nodes=allowed_nodes if allowed_nodes is not None else DEFAULT_ALLOWED_NODES,
            allowed_relationships=(
                allowed_relationships
                if allowed_relationships is not None
                else DEFAULT_ALLOWED_RELATIONSHIPS
            ),
            strict_mode=strict_mode,
            node_properties=False,
            relationship_properties=False,
            additional_instructions=additional_instructions,
        )

        # Lazily initialized after schema refresh.
        self.qa_chain: Optional[GraphCypherQAChain] = None

    def ingest_pdf_to_aura(self, pdf_path: Union[Path, str], max_chunks: Optional[int] = 10) -> dict:
        # Step 1: Load PDF pages into memory.
        loader = PyPDFLoader(str(pdf_path))
        pages = loader.load()

        # Step 2: Split into 1200-char chunks with 150-char overlap to balance extraction context vs. LLM call count.
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200,
            chunk_overlap=150,
        )
        chunks = splitter.split_documents(pages)
        if max_chunks is not None:
            chunks = chunks[:max_chunks]

        # Step 3: Extract GraphDocuments via LLM (primary ingestion latency/token bottleneck).
        graph_documents = self.transformer.convert_to_graph_documents(chunks)

        # Step 4: Write extracted nodes and edges into Neo4j AuraDB over TLS.
        self.graph.add_graph_documents(graph_documents, include_source=True)

        # Step 5: Refresh graph schema and compile Cypher QA chain.
        self.graph.refresh_schema()
        self._init_qa_chain()

        return {
            "pages": len(pages),
            "processed_chunks": len(chunks),
            "graph_documents": len(graph_documents),
        }

    def _init_qa_chain(self):
        # Refresh schema metadata from AuraDB before building the Cypher prompt chain.
        self.graph.refresh_schema()
        cypher_prompt = PromptTemplate(
            input_variables=["schema", "query"],
            template=CYPHER_GENERATION_TEMPLATE,
        )
        # return_intermediate_steps=True captures the generated Cypher query for UI and debugging.
        self.qa_chain = GraphCypherQAChain.from_llm(
            llm=self.llm,
            graph=self.graph,
            cypher_prompt=cypher_prompt,
            verbose=True,
            return_intermediate_steps=True,
            allow_dangerous_requests=True,
        )

    def ask(self, question: str) -> dict:
        if self.qa_chain is None:
            self._init_qa_chain()

        cb = _CypherCaptureCallback()
        result = self.qa_chain.invoke({"query": question}, config={"callbacks": [cb]})

        cypher_query = cb.cypher
        steps = result.get("intermediate_steps") or []
        for step in steps:
            if isinstance(step, dict) and step.get("query"):
                cypher_query = str(step["query"]).strip()
                break

        return {
            "answer": result.get("result", "I don't know based on the knowledge graph."),
            "cypher": cypher_query or "",
        }