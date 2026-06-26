"""
BM25 Document Q&A Bot (Vector-less RAG)
========================================
Ingests PDFs, indexes them with BM25 keyword search (no embeddings,
no vector database), and answers questions using Google Gemini Flash.

How it works:
  1. PDFs are parsed and split into overlapping text chunks.
  2. Chunks are tokenised and indexed in a BM25Okapi index (in-memory).
  3. At query time, BM25 ranks chunks by keyword relevance — no vectors needed.
  4. The top-K chunks form the context sent to Gemini for answer generation.

No vector DB. No embeddings API. No GPU. Runs entirely on CPU.

Setup:
    pip install -r requirements.txt
    cp .env.example .env   # add GOOGLE_API_KEY
    python main.py

UI launches at http://localhost:7860
Get a free API key at: https://aistudio.google.com/apikey
"""

import os
import re
import string
import time
from pathlib import Path
from typing import Generator

import gradio as gr
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pypdf import PdfReader
from rank_bm25 import BM25Okapi

load_dotenv()

gemini = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

CHAT_MODEL = "gemini-flash-latest"
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 150

# ── In-memory index (no vector DB) ───────────────────────────────────────────

# Each entry: {"text": str, "source": str, "chunk": int}
_corpus: list[dict] = []
_bm25: BM25Okapi | None = None


def _tokenise(text: str) -> list[str]:
    """Lowercase, remove punctuation, split on whitespace — classic BM25 tokens."""
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    return text.split()


def _rebuild_index() -> None:
    """Rebuild the BM25 index from the current corpus."""
    global _bm25
    if _corpus:
        _bm25 = BM25Okapi([_tokenise(doc["text"]) for doc in _corpus])
    else:
        _bm25 = None


# ── Ingestion ────────────────────────────────────────────────────────────────

def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping character-level chunks."""
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start : start + size])
        start += size - overlap
    return [c.strip() for c in chunks if c.strip()]


def ingest_pdf(pdf_path: str) -> int:
    """Parse PDF, chunk text, append to corpus, rebuild BM25 index."""
    global _corpus
    reader = PdfReader(pdf_path)
    full_text = "\n".join(page.extract_text() or "" for page in reader.pages)
    chunks = chunk_text(full_text)
    filename = Path(pdf_path).name

    _corpus.extend(
        {"text": chunk, "source": filename, "chunk": i}
        for i, chunk in enumerate(chunks)
    )
    _rebuild_index()
    return len(chunks)


def clear_index() -> None:
    """Wipe the in-memory corpus and index."""
    global _corpus, _bm25
    _corpus = []
    _bm25 = None


# ── Retrieval ────────────────────────────────────────────────────────────────

def retrieve(question: str, top_k: int = 5) -> list[dict]:
    """Return top-K chunks ranked by BM25 score."""
    if _bm25 is None or not _corpus:
        return []
    tokens = _tokenise(question)
    scores = _bm25.get_scores(tokens)
    ranked = sorted(
        zip(scores, _corpus), key=lambda x: x[0], reverse=True
    )
    return [
        {**doc, "score": float(score)}
        for score, doc in ranked[:top_k]
        if score > 0
    ]


# ── Generation ───────────────────────────────────────────────────────────────

def answer_stream(question: str, top_k: int = 5) -> Generator[str, None, None]:
    """Stream a Gemini answer grounded in BM25-retrieved context."""
    chunks = retrieve(question, top_k=top_k)

    if not chunks:
        if _bm25 is None:
            yield "No documents have been indexed yet. Please upload and index a PDF first."
        else:
            yield (
                "No relevant matches found in the document.\n\n"
                "BM25 search requires exact keyword matches. Try rephrasing with key terms from the document."
            )
        return

    context = "\n\n---\n\n".join(
        f"[Source: {c['source']}, chunk {c['chunk']}, BM25={c['score']:.2f}]\n{c['text']}"
        for c in chunks
    )
    sources = list({c["source"] for c in chunks})

    system_prompt = (
        "You are a helpful assistant that answers questions based strictly on the "
        "provided context. If the context doesn't contain the answer, say so clearly. "
        "Always cite the source document.\n\n"
        f"CONTEXT:\n{context}"
    )

    full_response = ""
    for attempt in range(5):
        try:
            for chunk in gemini.models.generate_content_stream(
                model=CHAT_MODEL,
                contents=question,
                config=types.GenerateContentConfig(system_instruction=system_prompt),
            ):
                if chunk.text:
                    full_response += chunk.text
                    yield full_response
            break
        except Exception as e:
            msg = str(e)
            delay_match = re.search(r"retryDelay.*?(\d+)s", msg)
            wait = int(delay_match.group(1)) + 2 if delay_match else 30
            is_transient = "429" in msg or "503" in msg or "UNAVAILABLE" in msg or "experiencing high demand" in msg
            if attempt < 4 and is_transient:
                err_type = "Rate limit hit" if "429" in msg else "Service busy (503)"
                for remaining in range(wait, 0, -1):
                    yield f"[ {err_type} - retrying in {remaining}s... ]"
                    time.sleep(1)
                full_response = ""
            else:
                if "429" in msg:
                    yield "[ Rate limit exceeded. Please wait a minute and try again. ]"
                elif "503" in msg or "UNAVAILABLE" in msg:
                    yield "[ Service temporarily unavailable. Please try again shortly. ]"
                else:
                    yield f"[ Error: {msg[:300]} ]"
                return

    yield full_response + f"\n\n---\n**Sources:** {', '.join(sources)}"


# ── Gradio UI ────────────────────────────────────────────────────────────────

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&family=Quicksand:wght@600;700;800&display=swap');

body, .gradio-container, button, input, select, textarea, span, p {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}

h1, h2, h3, .panel-label, .title-block h1 {
    font-family: 'Quicksand', sans-serif !important;
    font-weight: 700 !important;
}

.gradio-container {
    background-color: #fafbfc !important;
    background-image: 
        radial-gradient(at 0% 0%, rgba(99, 102, 241, 0.05) 0, transparent 40%),
        radial-gradient(at 100% 0%, rgba(6, 182, 212, 0.05) 0, transparent 40%),
        radial-gradient(at 50% 100%, rgba(244, 63, 94, 0.04) 0, transparent 50%) !important;
    color: #1e293b !important;
    max-width: 1200px !important;
    padding: 12px 24px !important;
}

.panel-block {
    background: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 20px !important;
    box-shadow: 0 10px 30px -5px rgba(100, 116, 139, 0.08) !important;
    padding: 20px !important;
    transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1) !important;
}
.panel-block:hover {
    border-color: #6366f1 !important;
    box-shadow: 0 16px 35px -8px rgba(99, 102, 241, 0.12) !important;
    transform: translateY(-1px) !important;
}

.header-container {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 12px 24px;
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 20px;
    margin-bottom: 16px;
    box-shadow: 0 8px 24px -4px rgba(100, 116, 139, 0.05);
}
.header-title-section {
    text-align: left;
}
.header-title-section h1 {
    font-size: 1.6rem;
    font-weight: 800;
    margin: 0 0 2px 0;
    background: linear-gradient(90deg, #6366f1 0%, #3b82f6 35%, #ec4899 70%, #f43f5e 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -0.02em;
}
.header-title-section p {
    color: #64748b;
    font-size: 0.85rem;
    font-weight: 500;
    margin: 0;
}

.badge-container {
    display: flex;
    justify-content: flex-end;
    flex-wrap: wrap;
    gap: 8px;
}
.badge {
    display: inline-block;
    background: #f1f5f9 !important;
    border: 1px solid #e2e8f0 !important;
    color: #475569 !important;
    border-radius: 9999px !important;
    padding: 4px 14px !important;
    font-size: 0.75rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.02em;
    transition: all 0.2s ease !important;
}
.badge:hover {
    background: rgba(99, 102, 241, 0.08) !important;
    border-color: #6366f1 !important;
    color: #6366f1 !important;
}

.panel-label {
    font-weight: 700;
    font-size: 0.85rem;
    margin-bottom: 12px;
    color: #6366f1;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}

input, textarea, .file-preview, .file-input, .gr-file {
    background: #ffffff !important;
    border: 1px solid #cbd5e1 !important;
    color: #0f172a !important;
    border-radius: 12px !important;
    padding: 10px !important;
    transition: all 0.2s ease !important;
}
input:focus, textarea:focus {
    border-color: #6366f1 !important;
    box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.15) !important;
}

button.primary, button.lg.primary {
    background: linear-gradient(135deg, #6366f1 0%, #3b82f6 100%) !important;
    border: none !important;
    color: #ffffff !important;
    font-weight: 600 !important;
    border-radius: 12px !important;
    box-shadow: 0 4px 14px rgba(99, 102, 241, 0.2) !important;
    transition: all 0.2s ease !important;
    cursor: pointer !important;
}
button.primary:hover {
    box-shadow: 0 6px 20px rgba(99, 102, 241, 0.35) !important;
    filter: brightness(1.05) !important;
}
button.primary:active {
    transform: scale(0.98) !important;
}

button.secondary, button.lg.secondary {
    background: #f8fafc !important;
    border: 1px solid #cbd5e1 !important;
    color: #475569 !important;
    border-radius: 12px !important;
    transition: all 0.2s ease !important;
    cursor: pointer !important;
}
button.secondary:hover {
    background: #f1f5f9 !important;
    border-color: #94a3b8 !important;
    color: #1e293b !important;
}

.chatbot-container {
    border-radius: 16px !important;
    background: #f8fafc !important;
    border: 1px solid #e2e8f0 !important;
}
.message.user {
    background: #e0e7ff !important;
    border: 1px solid #c7d2fe !important;
    color: #1e1b4b !important;
    border-radius: 12px 12px 0px 12px !important;
}
.message.bot {
    background: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-left: 4px solid #3b82f6 !important;
    color: #0f172a !important;
    border-radius: 12px 12px 12px 0px !important;
}

.slider-container input[type="range"] {
    accent-color: #3b82f6 !important;
}

.tip-box {
    background: #fefce8 !important;
    border: 1px solid #fef08a !important;
    border-radius: 16px !important;
    padding: 16px !important;
    color: #713f12 !important;
    font-size: 0.85rem !important;
    line-height: 1.5 !important;
}
.tip-box strong {
    color: #a16207 !important;
    font-weight: 700;
    display: block;
    margin-bottom: 4px;
}

textarea[readonly] {
    background: #f8fafc !important;
    border-color: #e2e8f0 !important;
    color: #64748b !important;
}

footer { display: none !important; }
"""


def upload_handler(file):
    if file is None:
        return "No file selected. Please choose a PDF."
    try:
        n = ingest_pdf(file.name)
        total = len(_corpus)
        return f"Successfully indexed {n} chunks from {Path(file.name).name}. Total corpus size: {total} chunks."
    except Exception as e:
        return f"Indexing failed: {str(e)[:200]}"


def clear_handler():
    clear_index()
    return "Index successfully cleared. Ready for a new PDF document."


def corpus_stats() -> str:
    docs = list({c["source"] for c in _corpus})
    if not docs:
        return "No documents indexed."
    return f"{len(_corpus)} chunks across {len(docs)} document(s): " + ", ".join(f"`{d}`" for d in docs)


with gr.Blocks(title="BM25 Q&A Bot", css=CSS) as demo:

    gr.HTML("""
    <div class="header-container">
        <div class="header-title-section">
            <h1>BM25 Document Q&A Bot</h1>
            <p>Vector-less RAG — keyword search with BM25, no embeddings, no vector database.</p>
        </div>
        <div class="badge-container">
            <span class="badge">Retrieval: BM25 (rank-bm25)</span>
            <span class="badge">Language Model: Gemini Flash</span>
            <span class="badge">Architecture: Vector-less & Local</span>
        </div>
    </div>
    """)

    with gr.Row(equal_height=False):

        with gr.Column(scale=1, min_width=300, elem_classes=["panel-block"]):
            gr.HTML('<p class="panel-label">Document Ingestion</p>')
            file_input = gr.File(label="Upload PDF", file_types=[".pdf"])
            with gr.Row():
                upload_btn = gr.Button("Index Document", variant="primary", scale=3)
                clear_btn  = gr.Button("Clear Index", variant="secondary", scale=1)
            upload_status = gr.Textbox(
                label="Status",
                interactive=False,
                lines=2,
                placeholder="Upload a PDF and click Index Document...",
            )
            stats_md = gr.Markdown("No documents indexed.")
            
            top_k_slider = gr.Slider(
                minimum=1, maximum=15, value=5, step=1,
                label="Top-K chunks", scale=1,
            )

            gr.HTML("""
            <div class="tip-box">
              <strong>BM25 Search Tip:</strong> BM25 is keyword-based. It matches exact terms between your query and the document. For best results, use specific words, names, or phrases from the PDF.
            </div>
            """)

            upload_btn.click(upload_handler, inputs=file_input, outputs=upload_status)
            upload_btn.click(corpus_stats, outputs=stats_md)
            clear_btn.click(clear_handler, outputs=upload_status)
            clear_btn.click(corpus_stats, outputs=stats_md)

        with gr.Column(scale=2, elem_classes=["panel-block"]):
            gr.HTML('<p class="panel-label">Assistant Console</p>')
            chatbot = gr.Chatbot(
                height=400,
                show_label=False,
                type="messages",
                placeholder="Upload and index a document on the left, then ask a question here.",
                elem_classes=["chatbot-container"],
            )
            with gr.Row():
                question_input = gr.Textbox(
                    placeholder="Ask a question about the document...",
                    show_label=False,
                    scale=4,
                    container=False,
                )
                ask_btn = gr.Button("Send", variant="primary", scale=1, min_width=80)
                clear_chat_btn = gr.Button("Clear Chat", variant="secondary", scale=1, min_width=80)

            def chat(question, history, top_k):
                if not question.strip():
                    yield history, ""
                    return
                history = history or []
                history.append({"role": "user", "content": question})
                history.append({"role": "assistant", "content": ""})
                for response in answer_stream(question, top_k=int(top_k)):
                    history[-1] = {"role": "assistant", "content": response}
                    yield history, ""

            ask_btn.click(
                chat,
                inputs=[question_input, chatbot, top_k_slider],
                outputs=[chatbot, question_input],
            )
            question_input.submit(
                chat,
                inputs=[question_input, chatbot, top_k_slider],
                outputs=[chatbot, question_input],
            )
            clear_chat_btn.click(lambda: [], outputs=chatbot)


if __name__ == "__main__":
    demo.launch(server_port=7866)
