"""
Enterprise Search
==================
Production-grade RAG with hybrid retrieval and multi-tenant access control.

Retrieval pipeline -- the same one Azure AI Search, Elasticsearch, and Weaviate
run internally under a single config flag:

    Your question
        |
    +---+---+
    |       |
  BM25    Vector search
  (rank-  (Qdrant +
   bm25)   text-embedding-004)
    |       |
    +---+---+
        |
    RRF merge (1 / (rank + k))
        |
    Access control filter
    (tenant_id + per-doc allowed_users)
        |
    Top-5 chunks -> Gemini Flash
        |
    Streamed answer + source citations

What is free:
    - BM25 keyword index:            fully local, no cost
    - Qdrant vector store:           in-memory, no cost
    - Embeddings (text-embedding-004): Google free tier
    - Answer (Gemini Flash):           Google free tier

    One GEMINI_API_KEY covers everything.

What managed services abstract away:
    Azure AI Search   -- hybrid search + semantic ranker in one API call
    Elasticsearch     -- BM25 built-in, kNN vectors added per-field
    Weaviate          -- hybrid search with RRF enabled via a flag
    Qdrant (hosted)   -- sparse+dense hybrid in a single query

This project shows you each step so you know what those flags actually do.
"""

import os
import string
import uuid
from collections import defaultdict
from dataclasses import dataclass

import gradio as gr
import google.generativeai as genai
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)
import pypdf

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY", ""))

# ── Config ─────────────────────────────────────────────────────────────────────
EMBED_MODEL   = "models/text-embedding-004"   # free, 768 dims
CHAT_MODEL    = "gemini-1.5-flash-latest"      # free tier
VECTOR_DIM    = 768
CHUNK_SIZE    = 900     # characters per chunk
CHUNK_OVERLAP = 100     # overlap between consecutive chunks
BM25_POOL     = 20      # candidates from BM25 before ACL filter
VECTOR_POOL   = 20      # candidates from vector search before ACL filter
FINAL_TOP_K   = 5       # chunks passed to the LLM
RRF_K         = 60      # RRF constant (standard value, reduces position bias)


# ── Data model ─────────────────────────────────────────────────────────────────

@dataclass
class DocChunk:
    id: str
    text: str
    source: str
    tenant_id: str
    allowed_users: list   # empty list = every user in the tenant can read it


# ── Storage ────────────────────────────────────────────────────────────────────

qdrant = QdrantClient(":memory:")
qdrant.create_collection(
    collection_name="kb",
    vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
)

# All indexed chunks keyed by chunk_id
chunk_store: dict[str, DocChunk] = {}

# BM25 index per tenant: {tenant_id -> {"chunks": [...], "index": BM25Okapi}}
bm25_store: dict[str, dict] = {}


# ── Sample documents ───────────────────────────────────────────────────────────
# Loaded at startup so the UI works immediately without uploading anything.
# Each document has an allowed_users list that controls who can see it.
# An empty list means everyone in the tenant can access it.

SAMPLE_DOCS = [
    {
        "text": (
            "ACME Corp Employee Salary Bands 2024\n\n"
            "Level 1 (Junior):  28,000 to 38,000\n"
            "Level 2 (Mid):     38,000 to 55,000\n"
            "Level 3 (Senior):  55,000 to 80,000\n"
            "Level 4 (Lead):    80,000 to 110,000\n"
            "Level 5 (Staff):  110,000 to 150,000\n\n"
            "Salary reviews run every January. Band adjustments require VP approval.\n"
            "Benefits include 25 days PTO, private healthcare, and 5% pension matching.\n"
            "Performance bonuses range from 5% to 20% of base salary."
        ),
        "source": "HR: Salary Bands 2024",
        "tenant_id": "acme-corp",
        "allowed_users": ["alice", "admin"],
    },
    {
        "text": (
            "ADR-007: Vector Database Selection\n"
            "Engineering Architecture Decision Record\n\n"
            "Decision: Adopt Qdrant as the primary vector store for all ML search features.\n\n"
            "Context: The team evaluated Pinecone, Weaviate, Milvus, and Qdrant.\n"
            "Qdrant was selected because:\n"
            "  1. Native hybrid search: sparse + dense vectors in a single query\n"
            "  2. On-premise deployment option for data sovereignty requirements\n"
            "  3. Superior payload filtering, essential for per-tenant access control\n"
            "  4. Rust implementation gives better p99 latency than Python-based alternatives\n\n"
            "Implications: All new RAG pipelines must use Qdrant.\n"
            "Legacy Pinecone collections to be migrated by Q2 2025.\n"
            "Local development uses QdrantClient(':memory:').\n\n"
            "Status: Accepted. Owner: Bob Chen. Date: 2024-03-12."
        ),
        "source": "Eng: ADR-007 Vector DB",
        "tenant_id": "acme-corp",
        "allowed_users": ["bob", "admin"],
    },
    {
        "text": (
            "ACME Corp Company Handbook\n\n"
            "ACME Corp builds AI-powered search tools for the enterprise market.\n"
            "Founded in 2019, we are now 350 people across London, Berlin, and Singapore.\n\n"
            "Values: Move fast. Be honest. Build for reliability.\n\n"
            "Working hours: Core hours are 10am to 4pm in your local timezone.\n"
            "We are fully remote-first. Offices are available for collaboration days.\n\n"
            "Meetings: All-hands every second Friday at 3pm UTC.\n"
            "One-on-ones with your manager: weekly, 30 minutes.\n\n"
            "Contacts:\n"
            "  HR questions: people@acmecorp.com\n"
            "  Engineering questions: #eng-help on Slack\n"
            "  IT support: it@acmecorp.com"
        ),
        "source": "Company Handbook",
        "tenant_id": "acme-corp",
        "allowed_users": [],  # empty = all users in the tenant
    },
    {
        "text": (
            "Q3 2024 Sales Targets and Pipeline Review\n\n"
            "Regional targets:\n"
            "  UK and Ireland: 2.4M  (110% of Q2 actuals)\n"
            "  DACH:           1.8M  (new region, conservative target)\n"
            "  Nordics:        900K  (existing accounts + 3 new enterprise logos)\n\n"
            "Key deals in pipeline:\n"
            "  1. NatWest Group: 420K ARR. Security review in progress. Close expected August.\n"
            "  2. Deutsche Bank: 380K ARR. Champion identified. Procurement engaged.\n"
            "  3. Volvo Cars:    220K ARR. POC complete. Awaiting board sign-off.\n\n"
            "Churn risk accounts: Two accounts flagged due to product gaps in mobile search.\n"
            "Customer Success to deliver retention plan by July 15."
        ),
        "source": "Sales: Q3 2024 Targets",
        "tenant_id": "acme-corp",
        "allowed_users": ["charlie", "alice", "admin"],
    },
    {
        "text": (
            "P0 Incident Report: Search Outage, June 14 2024\n\n"
            "Duration: 47 minutes (02:13 to 03:00 UTC)\n"
            "Impact: 100% of search queries failed for all tenants.\n"
            "Root cause: Qdrant OOM. A rogue batch indexing job consumed all container "
            "memory, triggering an OOM kill.\n\n"
            "Timeline:\n"
            "  02:13  Alert: search p99 latency over 30 seconds\n"
            "  02:18  On-call engineer paged\n"
            "  02:31  Root cause identified: Qdrant OOM-killed\n"
            "  02:40  Qdrant restarted with memory limit raised to 16GB\n"
            "  03:00  All health checks passing. Incident resolved.\n\n"
            "Action items:\n"
            "  Per-job memory quotas for indexing pipeline (Bob, due Jul 1)\n"
            "  Qdrant memory utilisation added to runbook (Bob, due Jun 21)\n"
            "  Synthetic search latency monitoring (Alice, due Jun 28)"
        ),
        "source": "Eng: P0 Incident Report",
        "tenant_id": "acme-corp",
        "allowed_users": ["bob", "admin"],
    },
]

# Who can see what -- shown in the UI sidebar for reference
USER_ACCESS_GUIDE = {
    "alice":   "Salary bands, Sales targets, Company handbook",
    "bob":     "Engineering ADR, Incident report, Company handbook",
    "charlie": "Sales targets, Company handbook",
    "admin":   "All documents",
}


# ── Core utilities ──────────────────────────────────────────────────────────────

def embed(text: str) -> list[float]:
    """Call Google text-embedding-004. Free tier, 768 dimensions."""
    result = genai.embed_content(model=EMBED_MODEL, content=text[:8000])
    return result["embedding"]


def tokenize(text: str) -> list[str]:
    """Lowercase + strip punctuation before BM25 tokenisation."""
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    return text.split()


def chunk_text(text: str) -> list[str]:
    """Split text into overlapping character-level chunks."""
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start: start + CHUNK_SIZE])
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return [c for c in chunks if c.strip()]


# ── Indexing ────────────────────────────────────────────────────────────────────

def index_document(text: str, source: str, tenant_id: str, allowed_users: list[str]) -> int:
    """
    Chunk, embed, and store a document.
    Rebuilds the BM25 index for the tenant after every upload.
    """
    chunks = chunk_text(text)
    for raw in chunks:
        cid = str(uuid.uuid4())
        emb = embed(raw)
        doc = DocChunk(id=cid, text=raw, source=source,
                       tenant_id=tenant_id, allowed_users=allowed_users)
        chunk_store[cid] = doc

        qdrant.upsert(
            collection_name="kb",
            points=[
                PointStruct(
                    id=cid,
                    vector=emb,
                    payload={
                        "chunk_id": cid,
                        "text": raw,
                        "source": source,
                        "tenant_id": tenant_id,
                        "allowed_users": allowed_users,
                    },
                )
            ],
        )

    # Rebuild BM25 for this tenant so new documents are immediately searchable.
    # O(n) per upload -- acceptable for a learning project; in production you
    # would use an incremental BM25 library or a dedicated search engine.
    tenant_chunks = [c for c in chunk_store.values() if c.tenant_id == tenant_id]
    bm25_store[tenant_id] = {
        "chunks": tenant_chunks,
        "index": BM25Okapi([tokenize(c.text) for c in tenant_chunks]),
    }
    return len(chunks)


# ── Retrieval ───────────────────────────────────────────────────────────────────

def bm25_retrieve(query: str, tenant_id: str, user_id: str) -> list[str]:
    """
    Score all tenant chunks with BM25, filter by user ACL, return top chunk IDs.
    BM25 is strong on exact keywords: part numbers, names, error codes, legal terms.
    """
    if tenant_id not in bm25_store:
        return []

    store  = bm25_store[tenant_id]
    scores = store["index"].get_scores(tokenize(query))

    ranked = sorted(
        zip(scores, store["chunks"]),
        key=lambda x: x[0],
        reverse=True,
    )
    return [
        c.id for sc, c in ranked
        if sc > 0 and (not c.allowed_users or user_id in c.allowed_users)
    ][:BM25_POOL]


def vector_retrieve(query: str, tenant_id: str, user_id: str) -> list[str]:
    """
    Dense vector search in Qdrant, filtered by tenant_id.
    Post-filter by user ACL since Qdrant cannot filter on list membership directly.
    Vector search is strong on meaning, paraphrasing, and conceptual queries.
    """
    query_emb = embed(query)
    results = qdrant.search(
        collection_name="kb",
        query_vector=query_emb,
        limit=VECTOR_POOL * 3,   # fetch extra to compensate for post-filtering loss
        query_filter=Filter(
            must=[FieldCondition(key="tenant_id", match=MatchValue(value=tenant_id))]
        ),
    )
    accessible = [
        r for r in results
        if not r.payload.get("allowed_users") or user_id in r.payload["allowed_users"]
    ]
    return [r.payload["chunk_id"] for r in accessible[:VECTOR_POOL]]


def rrf_merge(bm25_ids: list[str], vector_ids: list[str]) -> list[str]:
    """
    Reciprocal Rank Fusion.

    For each ranked list: score(chunk) += 1 / (rank + RRF_K)

    A chunk appearing near the top of both lists scores much higher than one
    appearing in only one list. RRF_K=60 is the standard value from the original
    Cormack et al. 2009 paper and is used unchanged by Elasticsearch and Qdrant.

    This is exactly what Azure AI Search applies when you set 'hybridSearch' in
    your index configuration.
    """
    scores: dict[str, float] = defaultdict(float)
    for rank, cid in enumerate(bm25_ids):
        scores[cid] += 1.0 / (rank + RRF_K)
    for rank, cid in enumerate(vector_ids):
        scores[cid] += 1.0 / (rank + RRF_K)
    return sorted(scores, key=lambda c: scores[c], reverse=True)


def hybrid_search(query: str, tenant_id: str, user_id: str):
    """Orchestrate the full hybrid retrieval pipeline."""
    bm25_ids  = bm25_retrieve(query, tenant_id, user_id)
    vec_ids   = vector_retrieve(query, tenant_id, user_id)
    fused_ids = rrf_merge(bm25_ids, vec_ids)[:FINAL_TOP_K]
    chunks    = [chunk_store[cid] for cid in fused_ids if cid in chunk_store]

    def srcs(ids):
        return list({chunk_store[c].source for c in ids if c in chunk_store})

    debug = {
        "bm25_hits":    len(bm25_ids),
        "vector_hits":  len(vec_ids),
        "bm25_sources": srcs(bm25_ids),
        "vec_sources":  srcs(vec_ids),
        "final":        [c.source for c in chunks],
    }
    return chunks, debug


# ── Generation ──────────────────────────────────────────────────────────────────

def respond(message: str, history: list, tenant_id: str, user_id: str):
    """Stream an answer from Gemini Flash using hybrid-retrieved context."""
    if not message.strip():
        yield "", history
        return

    if not chunk_store:
        history = history + [
            {"role": "user",      "content": message},
            {"role": "assistant", "content": "No documents indexed yet. Upload a PDF to get started."},
        ]
        yield "", history
        return

    chunks, debug = hybrid_search(message, tenant_id, user_id)

    if not chunks:
        history = history + [
            {"role": "user",      "content": message},
            {"role": "assistant", "content": (
                f"No documents found that **{user_id}** has access to for this query.\n\n"
                f"Try switching to **admin** or uploading a document with access for {user_id}."
            )},
        ]
        yield "", history
        return

    context = "\n\n---\n\n".join(f"[Source: {c.source}]\n{c.text}" for c in chunks)

    prompt = (
        f"You are an enterprise knowledge assistant for '{tenant_id}'.\n"
        "Answer using only the provided context. "
        "If the answer is not in the context, say so clearly.\n"
        "Always mention which source document your answer comes from.\n\n"
        f"Context:\n{context}\n\nQuestion: {message}"
    )

    debug_md = (
        f"\n\n---\n"
        f"**Retrieval breakdown:**\n"
        f"- BM25 (keyword) matched **{debug['bm25_hits']}** chunks "
        f"from: {', '.join(debug['bm25_sources']) or 'none'}\n"
        f"- Vector (semantic) matched **{debug['vector_hits']}** chunks "
        f"from: {', '.join(debug['vec_sources']) or 'none'}\n"
        f"- RRF merged both lists. Top **{FINAL_TOP_K}** used for this answer.\n"
        f"- **Sources cited:** {', '.join(set(debug['final']))}"
    )

    history = history + [
        {"role": "user",      "content": message},
        {"role": "assistant", "content": ""},
    ]

    model    = genai.GenerativeModel(CHAT_MODEL)
    response = model.generate_content(prompt, stream=True)
    full     = ""

    for part in response:
        full += part.text
        history[-1]["content"] = full
        yield "", history

    history[-1]["content"] = full + debug_md
    yield "", history


# ── Upload handler ───────────────────────────────────────────────────────────────

def upload_pdf(file, source_name: str, tenant_id: str, users_str: str):
    if file is None:
        return "No file selected."
    if not source_name.strip():
        return "Please enter a document name."

    reader  = pypdf.PdfReader(file.name)
    text    = "\n".join(page.extract_text() or "" for page in reader.pages)
    if not text.strip():
        return "Could not extract text from this PDF."

    allowed = [u.strip() for u in users_str.split(",") if u.strip()] if users_str.strip() else []
    n       = index_document(text, source_name.strip(), tenant_id.strip(), allowed)
    access  = ", ".join(allowed) if allowed else "all users in tenant"
    return f"Indexed {n} chunk(s) from '{source_name}'.\nAccess granted to: {access}."


def corpus_stats(tenant_id: str) -> str:
    chunks  = [c for c in chunk_store.values() if c.tenant_id == tenant_id]
    sources = sorted({c.source for c in chunks})
    if not chunks:
        return "No documents indexed yet."
    lines = [f"{len(chunks)} chunks across {len(sources)} source(s):"]
    lines += [f"  {s}" for s in sources]
    return "\n".join(lines)


# ── CSS ──────────────────────────────────────────────────────────────────────────

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
*, *::before, *::after { box-sizing: border-box; }
body, .gradio-container {
    font-family: 'Inter', system-ui, sans-serif !important;
    background: #f1f5f9 !important;
}
.header-container {
    background: white;
    border-radius: 14px;
    padding: 20px 28px;
    margin-bottom: 12px;
    border: 1px solid #e2e8f0;
    box-shadow: 0 1px 6px rgba(0,0,0,.06);
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 12px;
}
.header-title-section h1 { margin: 0 0 4px 0; font-size: 22px; font-weight: 700; color: #0f172a; }
.header-title-section p  { margin: 0; font-size: 13px; color: #64748b; }
.badge-container { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }
.badge {
    padding: 4px 12px; border-radius: 20px;
    font-size: 11px; font-weight: 600; letter-spacing: .3px;
    background: #fdf4ff; color: #7c3aed; border: 1px solid #e9d5ff;
}
.badge.local  { background: #f0fdf4; color: #166534; border-color: #bbf7d0; }
.badge.api    { background: #fffbeb; color: #92400e; border-color: #fde68a; }
.panel-block {
    background: white; border-radius: 12px; padding: 16px;
    border: 1px solid #e2e8f0; box-shadow: 0 1px 4px rgba(0,0,0,.05);
}
.panel-label {
    font-size: 11px !important; font-weight: 700 !important;
    letter-spacing: 1.2px !important; color: #64748b !important;
    text-transform: uppercase !important; margin: 16px 0 6px 0 !important;
}
.panel-label:first-child { margin-top: 0 !important; }
.access-guide {
    background: #f8fafc; border: 1px solid #e2e8f0;
    border-radius: 8px; padding: 10px 14px;
    font-size: 12px; color: #475569; line-height: 1.8;
}
.access-guide b { color: #0f172a; }
.access-guide .tip {
    margin-top: 8px; padding-top: 8px;
    border-top: 1px solid #e2e8f0; font-style: italic; color: #94a3b8;
}
"""

# ── UI ───────────────────────────────────────────────────────────────────────────

def load_samples():
    for doc in SAMPLE_DOCS:
        index_document(doc["text"], doc["source"], doc["tenant_id"], doc["allowed_users"])

load_samples()

TENANT = "acme-corp"
USERS  = list(USER_ACCESS_GUIDE.keys())

with gr.Blocks(title="Enterprise Search", css=CSS) as demo:

    tenant_state = gr.State(TENANT)

    gr.HTML("""
    <div class="header-container">
        <div class="header-title-section">
            <h1>Enterprise Search</h1>
            <p>Hybrid retrieval (BM25 + vector + RRF) with multi-tenant access control.
               The same pipeline Azure AI Search, Elasticsearch, and Weaviate run internally.</p>
        </div>
        <div class="badge-container">
            <span class="badge">Retrieval: BM25 + Vector + RRF</span>
            <span class="badge local">Embeddings: text-embedding-004 (local free)</span>
            <span class="badge api">LLM: Gemini Flash (free tier)</span>
            <span class="badge local">Access: per-doc ACL</span>
        </div>
    </div>
    """)

    with gr.Row(equal_height=False):

        # LEFT PANEL
        with gr.Column(scale=1, min_width=290, elem_classes=["panel-block"]):

            gr.HTML('<p class="panel-label">Current User</p>')
            user_dd = gr.Dropdown(choices=USERS, value="alice", label="", interactive=True)
            gr.HTML("""
            <div class="access-guide">
                <b>alice</b> &mdash; HR + Sales docs<br>
                <b>bob</b>   &mdash; Engineering docs<br>
                <b>charlie</b> &mdash; Sales docs only<br>
                <b>admin</b> &mdash; All documents<br>
                <div class="tip">
                    Switch users and ask the same question to see how
                    access control changes what gets retrieved.
                </div>
            </div>
            """)

            gr.HTML('<p class="panel-label">Corpus</p>')
            stats_box = gr.Textbox(
                value=corpus_stats(TENANT),
                label="", interactive=False, lines=7,
            )
            refresh_btn = gr.Button("Refresh", size="sm")
            refresh_btn.click(fn=lambda: corpus_stats(TENANT), outputs=[stats_box])

            gr.HTML('<p class="panel-label">Upload document</p>')
            pdf_file   = gr.File(label="PDF file", file_types=[".pdf"])
            source_in  = gr.Textbox(label="Document name", placeholder="e.g. Q4 Finance Report")
            users_in   = gr.Textbox(
                label="Allowed users (comma-separated, empty = all)",
                placeholder="alice, bob, admin",
            )
            upload_btn = gr.Button("Index document", variant="primary", size="sm")
            upload_out = gr.Textbox(label="Status", interactive=False, lines=2)

            upload_btn.click(
                fn=upload_pdf,
                inputs=[pdf_file, source_in, tenant_state, users_in],
                outputs=[upload_out],
            )

        # MAIN CHAT
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                type="messages",
                height=510,
                label="",
                show_label=False,
                bubble_full_width=False,
                render_markdown=True,
            )
            with gr.Row():
                msg_in   = gr.Textbox(
                    placeholder="Ask anything about company documents...",
                    show_label=False, scale=5, lines=1,
                )
                send_btn = gr.Button("Send", variant="primary", scale=1, min_width=80)

            clear_btn = gr.Button("Clear conversation", size="sm", variant="secondary")
            clear_btn.click(fn=lambda: [], outputs=[chatbot])

            def _submit(msg, hist, tenant, user):
                yield from respond(msg, hist, tenant, user)

            send_btn.click(
                fn=_submit,
                inputs=[msg_in, chatbot, tenant_state, user_dd],
                outputs=[msg_in, chatbot],
            )
            msg_in.submit(
                fn=_submit,
                inputs=[msg_in, chatbot, tenant_state, user_dd],
                outputs=[msg_in, chatbot],
            )

if __name__ == "__main__":
    demo.launch()
