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

# Always configure Gemini for embeddings & OCR fallback
genai.configure(api_key=os.getenv("GEMINI_API_KEY", ""))

# ── Config ─────────────────────────────────────────────────────────────────────
EMBED_MODEL   = "models/gemini-embedding-001"   # free, 768 dims
PROVIDER      = os.getenv("LLM_PROVIDER", "gemini").lower()
CHAT_MODEL    = os.getenv("CHAT_MODEL", "gemini-1.5-flash-latest")
VECTOR_DIM    = 3072
CHUNK_SIZE    = 900     # characters per chunk
CHUNK_OVERLAP = 100     # overlap between consecutive chunks
BM25_POOL     = 20      # candidates from BM25 before ACL filter
VECTOR_POOL   = 20      # candidates from vector search before ACL filter
FINAL_TOP_K   = 5       # chunks passed to the LLM
RRF_K         = 60      # RRF constant (standard value, reduces position bias)

# Initialize OpenAI/OpenRouter client if needed
openai_client = None
if PROVIDER == "openai":
    from openai import OpenAI
    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))
elif PROVIDER == "openrouter":
    from openai import OpenAI
    openai_client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY", ""),
        default_headers={
            "HTTP-Referer": "https://github.com/enterprise-search-bot",
            "X-Title": "Enterprise Search Bot"
        }
    )


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
# An empty list means everyone in the cabinet can access it.

SAMPLE_DOCS = [
    {
        "text": (
            "My Private Journal - June 2024\n\n"
            "June 10: Thinking of getting Mom a nice gardening set for her birthday next month. Need to keep it secret.\n"
            "June 14: Went to the dentist today. Need to pay the remaining invoice of $85.00 by next week.\n"
            "June 18: Started learning Python. It's really fun! The Agentic AI lessons make so much sense."
        ),
        "source": "My Personal Journal",
        "tenant_id": "my-cabinet",
        "allowed_users": ["self"],
    },
    {
        "text": (
            "Biology Class Group Project Notes\n"
            "Topic: Photosynthesis and Cellular Respiration.\n"
            "Group members: Self, Study Partner.\n\n"
            "Key Details:\n"
            "  1. Photosynthesis converts light energy into chemical energy stored in glucose.\n"
            "  2. Cellular respiration breaks down glucose to produce ATP (energy).\n"
            "  3. Presentation date: July 12. We need to submit the slides by July 10.\n\n"
            "Tasks:\n"
            "  - Self: Research Light-Independent Reactions (Calvin Cycle).\n"
            "  - Study Partner: Design the PowerPoint slides and write the abstract."
        ),
        "source": "Biology Group Project",
        "tenant_id": "my-cabinet",
        "allowed_users": ["self", "study_partner"],
    },
    {
        "text": (
            "Alice's Adventures in Wonderland - Public Summary\n"
            "Written by Lewis Carroll in 1865.\n\n"
            "Key Characters:\n"
            "  - Alice: A young girl who falls down a rabbit hole into a fantasy world.\n"
            "  - The White Rabbit: The prompt and anxious rabbit who leads Alice down the hole.\n"
            "  - The Cheshire Cat: A grinning cat who can disappear and reappear at will.\n"
            "  - The Queen of Hearts: The hot-tempered ruler who frequently orders executions ('Off with their heads!').\n\n"
            "Famous Scenes:\n"
            "  1. The Mad Tea-Party with the Mad Hatter and the March Hare.\n"
            "  2. The caucus-race and Alice swimming in a pool of her own tears."
        ),
        "source": "Alice in Wonderland Summary",
        "tenant_id": "my-cabinet",
        "allowed_users": [],  # empty = public (all users)
    },
]

# Who can see what -- shown in the UI sidebar for reference
USER_ACCESS_GUIDE = {
    "self":          "Personal journal, Biology project, Alice in Wonderland book",
    "study_partner": "Biology project, Alice in Wonderland book",
    "guest":         "Alice in Wonderland book only",
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
    response = qdrant.query_points(
        collection_name="kb",
        query=query_emb,
        limit=VECTOR_POOL * 3,   # fetch extra to compensate for post-filtering loss
        query_filter=Filter(
            must=[FieldCondition(key="tenant_id", match=MatchValue(value=tenant_id))]
        ),
    )
    # Log scores to console
    print(f"\n--- [Vector Search] Query: '{query}' ---")
    for r in response.points:
        print(f"  - Document: '{r.payload.get('source')}' | Similarity Score: {r.score:.4f}")

    accessible = [
        r for r in response.points
        if r.score >= 0.50 and (not r.payload.get("allowed_users") or user_id in r.payload["allowed_users"])
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
        yield "", history, ""
        return

    # 1. Immediately append the user message to history and yield
    # so the user sees their query instantly on the UI!
    history = history + [
        {"role": "user",      "content": message},
        {"role": "assistant", "content": "Thinking..."},
    ]
    yield "", history, "🔍 *Searching database...*"

    if not chunk_store:
        history[-1]["content"] = "No documents indexed yet. Upload a PDF to get started."
        yield "", history, "⚠️ No documents indexed yet."
        return

    # 2. Run retrieval inside a try-except block
    try:
        chunks, debug = hybrid_search(message, tenant_id, user_id)
    except Exception as e:
        history[-1]["content"] = f"Retrieval Error: {str(e)}"
        yield "", history, f"❌ Retrieval Error: {str(e)}"
        return

    if not chunks:
        history[-1]["content"] = (
            f"No documents found that **{user_id}** has access to for this query.\n\n"
            f"Try switching user roles or uploading a new document."
        )
        yield "", history, "⚠️ No matching chunks found."
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
        f"### 🔍 Retrieval Insights & Metrics\n\n"
        f"📊 **BM25 (Keyword Index)**\n"
        f"- Matches: **{debug['bm25_hits']}** chunks\n"
        f"- Sources matched: `{', '.join(debug['bm25_sources']) or 'None'}`\n\n"
        f"🧠 **Vector (Semantic Index)**\n"
        f"- Matches: **{debug['vector_hits']}** chunks\n"
        f"- Sources matched: `{', '.join(debug['vec_sources']) or 'None'}`\n\n"
        f"🔀 **Reciprocal Rank Fusion (RRF)**\n"
        f"- Merged candidates: `{', '.join(set(debug['final'])) or 'None'}`\n"
        f"- Top **{FINAL_TOP_K}** highest-ranked chunks sent as LLM context."
    )

    # 3. Stream from model
    full = ""
    if PROVIDER in ("openai", "openrouter"):
        if not openai_client:
            history[-1]["content"] = f"Error: Provider '{PROVIDER}' selected, but client is not initialized. Please verify your API keys."
            yield "", history, "❌ Configuration Error"
            return

        try:
            response = openai_client.chat.completions.create(
                model=CHAT_MODEL,
                messages=[
                    {"role": "system", "content": "You are a helpful knowledge assistant."},
                    {"role": "user", "content": prompt}
                ],
                stream=True
            )
            for chunk in response:
                delta = chunk.choices[0].delta.content if chunk.choices and chunk.choices[0].delta else None
                if delta:
                    full += delta
                    history[-1]["content"] = full
                    yield "", history, debug_md
        except Exception as e:
            history[-1]["content"] = f"API Error ({PROVIDER}): {str(e)}"
            yield "", history, f"❌ API Error ({PROVIDER})"
            return
    else:
        # Default Gemini flow
        try:
            model    = genai.GenerativeModel(CHAT_MODEL)
            response = model.generate_content(prompt, stream=True)
            for part in response:
                full += part.text
                history[-1]["content"] = full
                yield "", history, debug_md
        except Exception as e:
            history[-1]["content"] = f"API Error (gemini): {str(e)}"
            yield "", history, f"❌ API Error (gemini)"
            return

    history[-1]["content"] = full
    yield "", history, debug_md


def ocr_pdf_with_gemini(file_path: str) -> str:
    """Upload a scanned PDF to Google's Files API and extract text using Gemini."""
    if not os.getenv("GEMINI_API_KEY"):
        raise ValueError("GEMINI_API_KEY is not set. Cannot run OCR fallback.")
    
    # Upload the file to Gemini Files API
    uploaded_file = genai.upload_file(path=file_path)
    
    try:
        model = genai.GenerativeModel("gemini-1.5-flash-latest")
        prompt = (
            "Perform OCR on this document. Transcribe all text, tables, and handwritten notes page-by-page. "
            "Preserve the original layout as much as possible, converting tables to Markdown format."
        )
        response = model.generate_content([uploaded_file, prompt])
        return response.text
    finally:
        # Clean up the file from the Google files hosting
        uploaded_file.delete()


# ── Upload handler ───────────────────────────────────────────────────────────────

def upload_pdf(file, source_name: str, tenant_id: str, users_str: str):
    if file is None:
        return "No file selected."
    if not source_name.strip():
        return "Please enter a document name."

    reader  = pypdf.PdfReader(file.name)
    text    = "\n".join(page.extract_text() or "" for page in reader.pages)
    
    status_msg = ""
    # Fallback to Gemini OCR if text is empty or too short (scanned PDF detection)
    if len(text.strip()) < 50:
        try:
            text = ocr_pdf_with_gemini(file.name)
            status_msg = " [OCR Fallback used]"
        except Exception as e:
            return f"Failed to extract text using standard parser, and OCR failed: {str(e)}"

    if not text.strip():
        return "Could not extract text from this PDF."

    allowed = [u.strip() for u in users_str.split(",") if u.strip()] if users_str.strip() else []
    n       = index_document(text, source_name.strip(), tenant_id.strip(), allowed)
    access  = ", ".join(allowed) if allowed else "all users in tenant"
    return f"Indexed {n} chunk(s) from '{source_name}'.{status_msg}\nAccess granted to: {access}."


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
    padding: 8px 16px;
    margin-bottom: 6px;
    border: 1px solid #e2e8f0;
    box-shadow: 0 1px 6px rgba(0,0,0,.06);
    display: flex;
    align-items: center;
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

def generate_sample_pdfs():
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet
    
    os.makedirs("sample_files", exist_ok=True)
    styles = getSampleStyleSheet()
    
    # 1. My Personal Journal PDF
    pdf_path1 = "sample_files/My_Personal_Journal.pdf"
    if not os.path.exists(pdf_path1):
        doc = SimpleDocTemplate(pdf_path1, pagesize=letter)
        story = [
            Paragraph("<b>My Private Journal - June 2024</b>", styles["Title"]),
            Spacer(1, 15),
            Paragraph("<b>June 10:</b> Thinking of getting Mom a nice gardening set for her birthday next month. Need to keep it secret.", styles["Normal"]),
            Spacer(1, 10),
            Paragraph("<b>June 14:</b> Went to the dentist today. Need to pay the remaining invoice of $85.00 by next week.", styles["Normal"]),
            Spacer(1, 10),
            Paragraph("<b>June 18:</b> Started learning Python. It's really fun! The Agentic AI lessons make so much sense.", styles["Normal"]),
        ]
        doc.build(story)

    # 2. Biology Group Project PDF
    pdf_path2 = "sample_files/Biology_Group_Project.pdf"
    if not os.path.exists(pdf_path2):
        doc = SimpleDocTemplate(pdf_path2, pagesize=letter)
        story = [
            Paragraph("<b>Biology Class Group Project Notes</b>", styles["Title"]),
            Spacer(1, 15),
            Paragraph("<b>Topic:</b> Photosynthesis and Cellular Respiration.<br/><b>Group members:</b> Self, Study Partner.", styles["Normal"]),
            Spacer(1, 12),
            Paragraph("<b>Key Details:</b>", styles["Heading3"]),
            Paragraph("1. Photosynthesis converts light energy into chemical energy stored in glucose.", styles["Normal"]),
            Paragraph("2. Cellular respiration breaks down glucose to produce ATP (energy).", styles["Normal"]),
            Paragraph("3. Presentation date: July 12. We need to submit the slides by July 10.", styles["Normal"]),
            Spacer(1, 10),
            Paragraph("<b>Tasks:</b>", styles["Heading3"]),
            Paragraph("- Self: Research Light-Independent Reactions (Calvin Cycle).", styles["Normal"]),
            Paragraph("- Study Partner: Design the PowerPoint slides and write the abstract.", styles["Normal"]),
        ]
        doc.build(story)

    # 3. Alice in Wonderland Summary PDF
    pdf_path3 = "sample_files/Alice_in_Wonderland_Summary.pdf"
    if not os.path.exists(pdf_path3):
        doc = SimpleDocTemplate(pdf_path3, pagesize=letter)
        story = [
            Paragraph("<b>Alice's Adventures in Wonderland - Public Summary</b>", styles["Title"]),
            Spacer(1, 15),
            Paragraph("Written by Lewis Carroll in 1865.", styles["Normal"]),
            Spacer(1, 12),
            Paragraph("<b>Key Characters:</b>", styles["Heading3"]),
            Paragraph("- <b>Alice:</b> A young girl who falls down a rabbit hole into a fantasy world.", styles["Normal"]),
            Paragraph("- <b>The White Rabbit:</b> The prompt and anxious rabbit who leads Alice down the hole.", styles["Normal"]),
            Paragraph("- <b>The Cheshire Cat:</b> A grinning cat who can disappear and reappear at will.", styles["Normal"]),
            Paragraph("- <b>The Queen of Hearts:</b> The hot-tempered ruler who frequently orders executions ('Off with their heads!').", styles["Normal"]),
            Spacer(1, 10),
            Paragraph("<b>Famous Scenes:</b>", styles["Heading3"]),
            Paragraph("1. The Mad Tea-Party with the Mad Hatter and the March Hare.", styles["Normal"]),
            Paragraph("2. The caucus-race and Alice swimming in a pool of her own tears.", styles["Normal"]),
        ]
        doc.build(story)

# ── UI ───────────────────────────────────────────────────────────────────────────

def load_samples():
    generate_sample_pdfs()
    for doc in SAMPLE_DOCS:
        index_document(doc["text"], doc["source"], doc["tenant_id"], doc["allowed_users"])

load_samples()

TENANT = "my-cabinet"
USERS  = list(USER_ACCESS_GUIDE.keys())

with gr.Blocks(title="Smart File Cabinet") as demo:

    tenant_state = gr.State(TENANT)

    gr.HTML("""
    <div class="header-container">
        <div class="header-title-section">
            <h1>Smart File Cabinet</h1>
            <p>Hybrid retrieval (BM25 + vector + RRF) with user-level access control.
               Securely chat with your personal diaries, shared class project notes, and public reference files.</p>
        </div>
        <div class="badge-container">
            <span class="badge">Retrieval: BM25 + Vector + RRF</span>
            <span class="badge local">Embeddings: gemini-embedding-001 (free)</span>
            <span class="badge api">LLM: Gemini 2.5 Flash</span>
            <span class="badge local">Access: User-Level ACL</span>
        </div>
    </div>
    """)

    with gr.Row(equal_height=False):

        # LEFT PANEL (Vault & Uploads)
        with gr.Column(scale=1, min_width=290, elem_classes=["panel-block"]):
            with gr.Tabs():
                with gr.Tab("📁 Vault Corpus"):
                    gr.HTML('<p class="panel-label">Active Corpus</p>')
                    stats_box = gr.Textbox(
                        value=corpus_stats(TENANT),
                        label="", interactive=False, lines=4,
                    )
                    with gr.Row(variant="compact"):
                        with gr.Column(scale=3):
                            gr.HTML("""
                            <div style="font-size: 11px; line-height: 1.3; color: #475569; padding-top: 2px;">
                                💡 These 3 documents are pre-loaded in your cabinet. Download their physical PDFs below to inspect.
                            </div>
                            """)
                        with gr.Column(scale=1, min_width=75):
                            refresh_btn = gr.Button("Refresh", size="sm")
                    
                    gr.HTML('<p class="panel-label" style="margin-top: 10px !important;">Download Sample PDFs</p>')
                    gr.File(
                        value=[
                            "sample_files/My_Personal_Journal.pdf",
                            "sample_files/Biology_Group_Project.pdf",
                            "sample_files/Alice_in_Wonderland_Summary.pdf"
                        ],
                        label="Local Sample PDFs",
                        file_count="multiple",
                        interactive=False
                    )

                with gr.Tab("📤 Upload New PDF"):
                    gr.HTML("""
                    <div style="margin-bottom: 8px; font-size: 0.8em; line-height: 1.3; color: #475569; background-color: #f8fafc; padding: 6px; border: 1px solid #e2e8f0; border-radius: 4px;">
                        <b>Need test files?</b> Download online PDFs:
                        <ul style="margin: 2px 0 0 12px; padding: 0;">
                            <li><a href="https://bitcoin.org/bitcoin.pdf" target="_blank" style="color: #ea580c; text-decoration: underline;">Bitcoin Whitepaper</a></li>
                            <li><a href="https://www.gutenberg.org/files/1661/1661-pdf.pdf" target="_blank" style="color: #ea580c; text-decoration: underline;">Sherlock Holmes</a></li>
                        </ul>
                    </div>
                    """)
                    pdf_file   = gr.File(label="PDF file", file_types=[".pdf"])
                    source_in  = gr.Textbox(label="Document name", placeholder="e.g. Textbook")
                    users_in   = gr.Textbox(
                        label="Allowed users (comma-separated)",
                        placeholder="self, study_partner",
                    )
                    upload_btn = gr.Button("Index document", variant="primary", size="sm")
                    upload_out = gr.Textbox(label="Status", interactive=False, lines=2)

            # Wiring Left Column Actions
            refresh_btn.click(fn=lambda: corpus_stats(TENANT), outputs=[stats_box])
            upload_btn.click(
                fn=upload_pdf,
                inputs=[pdf_file, source_in, tenant_state, users_in],
                outputs=[upload_out],
            )

        # RIGHT PANEL: MAIN CHAT & DETAILS
        with gr.Column(scale=3):
            # Compact User Selector and Guide inline above Chatbot
            with gr.Row(variant="compact"):
                with gr.Column(scale=1, min_width=90):
                    gr.HTML("""
                    <div style="font-size: 12px; font-weight: 600; color: #475569; margin-top: 8px; text-align: right;">
                        👤 Active User:
                    </div>
                    """)
                with gr.Column(scale=2, min_width=120):
                    user_dd = gr.Dropdown(
                        choices=USERS, value="guest", 
                        show_label=False, 
                        interactive=True
                    )
                with gr.Column(scale=6):
                    gr.HTML("""
                    <div style="font-size: 10.5px; line-height: 1.3; color: #475569; padding: 6px 10px; background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 6px;">
                        🔑 <b>Access Matrix:</b> <b>self</b> (all files) | <b>study_partner</b> (shared notes & public) | <b>guest</b> (public only)
                    </div>
                    """)

            with gr.Tabs():
                with gr.Tab("💬 Cabinet Chatbot"):
                    chatbot = gr.Chatbot(
                        height=280,
                        label="",
                        show_label=False,
                        render_markdown=True,
                    )
                    with gr.Row():
                        msg_in   = gr.Textbox(
                            placeholder="Ask anything about your documents...",
                            show_label=False, scale=5, lines=1,
                        )
                        send_btn = gr.Button("Send", variant="primary", scale=1, min_width=80)

                    clear_btn = gr.Button("Clear conversation", size="sm", variant="secondary")

                with gr.Tab("🔍 Retrieval Logs & Metrics"):
                    debug_output = gr.Markdown(
                        value="*Ask a question to see the step-by-step mathematical retrieval details (BM25 vs. Vector scores).* "
                    )

            # Wiring Right Column Actions
            clear_btn.click(
                fn=lambda: ([], "*Ask a question to see the step-by-step mathematical retrieval details (BM25 vs. Vector scores).* "), 
                outputs=[chatbot, debug_output]
            )

            def _submit(msg, hist, tenant, user):
                yield from respond(msg, hist, tenant, user)

            send_btn.click(
                fn=_submit,
                inputs=[msg_in, chatbot, tenant_state, user_dd],
                outputs=[msg_in, chatbot, debug_output],
            )
            msg_in.submit(
                fn=_submit,
                inputs=[msg_in, chatbot, tenant_state, user_dd],
                outputs=[msg_in, chatbot, debug_output],
            )

if __name__ == "__main__":
    demo.launch(css=CSS)
