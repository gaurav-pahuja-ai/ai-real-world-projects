"""
Document Q&A Bot with RAG
=========================
Ingests PDFs, indexes them in ChromaDB, and answers questions
with source citations and streaming responses.

Embeddings: ChromaDB built-in (all-MiniLM-L6-v2 via ONNX, runs locally, no API needed)
Chat:        Google Gemini 1.5 Flash (free tier)

Setup:
    pip install -r requirements.txt
    cp .env.example .env  # add GOOGLE_API_KEY
    python main.py

Usage:
    The Gradio UI launches at http://localhost:7860
    Get a free API key at: https://aistudio.google.com/apikey
"""

import os
import re
import time
from pathlib import Path
from typing import Generator

import chromadb
import gradio as gr
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pypdf import PdfReader

load_dotenv()

gemini = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

chroma = chromadb.PersistentClient(path="./chroma_db")
collection = chroma.get_or_create_collection(
    name="documents_local",
    metadata={"hnsw:space": "cosine"},
)

CHAT_MODEL = "gemini-flash-latest"
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 150


# ── Indexing ────────────────────────────────────────────────────────────────

def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks by character count."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start += size - overlap
    return [c.strip() for c in chunks if c.strip()]


def ingest_pdf(pdf_path: str) -> int:
    """Ingest a PDF file into ChromaDB. ChromaDB auto-embeds using local ONNX model."""
    reader = PdfReader(pdf_path)
    full_text = "\n".join(page.extract_text() or "" for page in reader.pages)
    chunks = chunk_text(full_text)
    filename = Path(pdf_path).name

    collection.add(
        documents=chunks,
        ids=[f"{filename}::chunk_{i}" for i in range(len(chunks))],
        metadatas=[{"source": filename, "chunk": i} for i in range(len(chunks))],
    )
    return len(chunks)


# ── Retrieval + Generation ───────────────────────────────────────────────────

def retrieve(question: str, top_k: int = 5) -> list[dict]:
    """Retrieve top-K chunks using ChromaDB's local semantic search."""
    results = collection.query(
        query_texts=[question],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    return [
        {"text": doc, "source": meta["source"], "chunk": meta["chunk"], "score": 1 - dist}
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        )
    ]


def answer_stream(question: str, top_k: int = 5) -> Generator[str, None, None]:
    """Stream an answer with source citations and auto-retry on rate limit."""
    chunks = retrieve(question, top_k=top_k)
    if not chunks:
        yield "📂 No documents indexed yet. Please upload and index a PDF first."
        return

    context = "\n\n---\n\n".join(
        f"[Source: {c['source']}, chunk {c['chunk']}]\n{c['text']}" for c in chunks
    )
    sources = list({c["source"] for c in chunks})

    system_prompt = (
        "You are a helpful assistant that answers questions based strictly on the "
        "provided context. If the context doesn't contain the answer, say so clearly. "
        "Always cite sources.\n\n"
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
            if attempt < 4 and "429" in msg:
                for remaining in range(wait, 0, -1):
                    yield f"⏳ **Rate limit hit** — free tier quota reached. Retrying in **{remaining}s**..."
                    time.sleep(1)
                full_response = ""
            else:
                if "429" in msg:
                    yield "❌ **Rate limit exhausted** — all retry attempts failed. Please wait a minute and try again."
                else:
                    yield f"❌ **Error** — {msg[:300]}"
                return

    yield full_response + f"\n\n---\n📎 **Sources:** {', '.join(sources)}"


# ── Gradio UI ────────────────────────────────────────────────────────────────

CSS = """
.title-block { text-align: center; padding: 16px 0 8px; }
.title-block h1 { font-size: 2rem; margin-bottom: 4px; }
.title-block p  { color: #6b7280; font-size: 0.95rem; }
.badge { display: inline-block; background: #f3f4f6; border-radius: 6px;
         padding: 2px 10px; font-size: 0.8rem; color: #374151; margin: 2px; }
.panel-label { font-weight: 600; font-size: 1rem; margin-bottom: 4px; color: #111827; }
footer { display: none !important; }
"""

def upload_handler(file):
    if file is None:
        return "⚠️ No file selected. Please choose a PDF."
    try:
        n = ingest_pdf(file.name)
        return f"✅ Indexed **{n} chunks** from `{Path(file.name).name}`"
    except Exception as e:
        return f"❌ Indexing failed: {str(e)[:200]}"

def clear_index_handler():
    global collection
    chroma.delete_collection("documents_local")
    collection = chroma.get_or_create_collection(
        name="documents_local",
        metadata={"hnsw:space": "cosine"},
    )
    return "🗑️ Index cleared. Upload a new PDF to begin."

with gr.Blocks(title="Document Q&A Bot", css=CSS, theme=gr.themes.Soft()) as demo:

    gr.HTML("""
    <div class="title-block">
        <h1>📄 Document Q&A Bot</h1>
        <p>Upload a PDF, index it, then ask questions — answers are grounded in your document.</p>
        <span class="badge">🧠 Embeddings: local ONNX</span>
        <span class="badge">✨ Chat: Gemini Flash (free)</span>
        <span class="badge">🗄️ Vector DB: ChromaDB</span>
    </div>
    """)

    with gr.Row(equal_height=False):

        with gr.Column(scale=1, min_width=280):
            gr.HTML('<p class="panel-label">📁 Document</p>')
            file_input = gr.File(label="Upload PDF", file_types=[".pdf"])
            with gr.Row():
                upload_btn = gr.Button("⚡ Index Document", variant="primary", scale=3)
                clear_btn  = gr.Button("🗑️ Clear", variant="secondary", scale=1)
            upload_status = gr.Textbox(
                label="Status",
                interactive=False,
                lines=2,
                placeholder="Upload a PDF and click Index Document...",
            )
            upload_btn.click(upload_handler, inputs=file_input, outputs=upload_status)
            clear_btn.click(clear_index_handler, outputs=upload_status)

        with gr.Column(scale=2):
            gr.HTML('<p class="panel-label">💬 Chat</p>')
            chatbot = gr.Chatbot(
                height=460,
                show_label=False,
                placeholder="Index a document on the left, then ask a question below.",
                avatar_images=(None, "https://www.gstatic.com/lamda/images/gemini_sparkle_v002_d4735304ff6292a690345.svg"),
            )
            with gr.Row():
                question_input = gr.Textbox(
                    placeholder="Ask a question about your document...",
                    show_label=False,
                    scale=5,
                    container=False,
                )
                ask_btn = gr.Button("Send ➤", variant="primary", scale=1, min_width=80)
            clear_chat_btn = gr.Button("🗑️ Clear chat", variant="secondary", size="sm")

            def chat(question, history):
                if not question.strip():
                    yield history, ""
                    return
                history = history or []
                history.append({"role": "user", "content": question})
                history.append({"role": "assistant", "content": ""})
                for response in answer_stream(question):
                    history[-1] = {"role": "assistant", "content": response}
                    yield history, ""

            ask_btn.click(chat, inputs=[question_input, chatbot], outputs=[chatbot, question_input])
            question_input.submit(chat, inputs=[question_input, chatbot], outputs=[chatbot, question_input])
            clear_chat_btn.click(lambda: [], outputs=chatbot)

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())
