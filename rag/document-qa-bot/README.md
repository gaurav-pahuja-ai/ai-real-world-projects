# Document Q&A Bot

> Upload a PDF. Ask questions. Get answers with source citations — streamed in real time.

A production-ready RAG (Retrieval-Augmented Generation) system built with Google Gemini, ChromaDB, and Gradio. Embeddings run **fully locally** on your CPU via ChromaDB's built-in ONNX model — only the chat model needs an API key.

This project is the hands-on companion to the [RAG Pipeline lesson](https://github.com/MrPahuja/ai-learning-hub) in AI Learning Hub.

---

## Stack

| Layer | Tool | Notes |
|---|---|---|
| PDF parsing | `pypdf` | Extracts text from all pages |
| Embeddings | ChromaDB built-in (`all-MiniLM-L6-v2`) | Runs locally via ONNX — no API key needed |
| Vector store | `ChromaDB` (persistent, local) | Saved to `./chroma_db` |
| LLM | `gemini-flash-latest` (streaming) | Only requires Google API key |
| UI | `Gradio` | Chat interface at `localhost:7860` |

---

## How It Works

### Phase 1 — Indexing (runs once per document)

```
PDF File
   │
   ▼
Extract text with pypdf
   │
   ▼
Chunk into 1500-char pieces with 150-char overlap
   │  (overlap ensures sentences split at boundaries stay findable)
   ▼
ChromaDB auto-embeds each chunk locally
   │  model: all-MiniLM-L6-v2 via ONNX (no API call)
   │  collection.add(documents=chunks, ...)
   ▼
Stored in ChromaDB
   id:       "report.pdf::chunk_0"
   vector:   [0.12, -0.87, 0.34, ...]
   text:     "The quick brown fox..."
   metadata: { source: "report.pdf", chunk: 0 }
```

### Phase 2 — Query (runs live, per question)

```
User question: "What is the return policy?"
   │
   ▼
ChromaDB auto-embeds the question locally
   │  collection.query(query_texts=["What is..."])
   ▼
Cosine similarity search → top 5 matching chunks
   │
   ▼
Build prompt:
   System:   "Answer strictly from context. Cite sources."
   Context:  [chunk_7] [chunk_12] [chunk_3] [chunk_19] [chunk_5]
   Question: "What is the return policy?"
   │
   ▼
gemini-flash-latest (streamed response)
   │  Auto-retries up to 5× on rate limit (429) with countdown timer
   ▼
Answer + "📎 Sources: report.pdf"
```

---

## Key Configuration

| Constant | Value | Why |
|---|---|---|
| `CHUNK_SIZE` | 1500 chars | Larger chunks preserve more context per retrieval |
| `CHUNK_OVERLAP` | 150 chars | 10% overlap prevents boundary cutoffs |
| `CHAT_MODEL` | `gemini-flash-latest` | Fast, free tier, supports streaming |
| `top_k` | 5 | Top 5 chunks sent to the LLM |
| `hnsw:space` | cosine | Similarity metric in ChromaDB |
| Embed model | `all-MiniLM-L6-v2` | ChromaDB's default, runs on CPU via ONNX |

---

## Setup

**1. Get a free Google API key**

Go to [aistudio.google.com/apikey](https://aistudio.google.com/apikey) — no billing required.

**2. Install dependencies**

```bash
cd rag/document-qa-bot
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

> On first run, ChromaDB downloads the `all-MiniLM-L6-v2` ONNX model (~90MB). This happens once and is cached automatically.

**3. Add your API key**

```bash
cp .env.example .env
# Open .env and set GOOGLE_API_KEY=your_key_here
```

**4. Run**

```bash
python main.py
```

The Gradio UI launches at **http://localhost:7860**

---

## Usage

1. Click **Upload PDF** and select a PDF file
2. Click **⚡ Index Document** — wait for the "Indexed N chunks" confirmation
3. Type a question in the chat box and press **Send**
4. The answer streams in real time with source citations at the bottom
5. Use **🗑️ Clear** to wipe the vector index and start fresh with a new document
6. Use **🗑️ Clear chat** to reset the conversation without re-indexing

You can index multiple PDFs. All chunks are stored persistently in `./chroma_db` and survive restarts.

---

## Project Structure

```
document-qa-bot/
├── main.py          # Full app: indexing + retrieval + Gradio UI
├── requirements.txt # Dependencies
├── .env.example     # API key template
├── .env             # Your actual API key (git-ignored)
└── chroma_db/       # Persistent vector store (auto-created on first run)
```

---

## Concepts Demonstrated

- **Chunking** with fixed size (1500) and overlap (150) window
- **Local embeddings** via ChromaDB's built-in ONNX model — no embedding API needed
- **Cosine similarity search** via ChromaDB's HNSW index
- **Prompt construction** with retrieved context injected into system instructions
- **Streaming responses** using `generate_content_stream`
- **Source citations** surfaced from chunk metadata
- **Rate limit retry** with exponential backoff and live countdown in the UI

> To understand the theory behind each step, read the [RAG Pipeline lesson](https://github.com/MrPahuja/ai-learning-hub) in AI Learning Hub.

---

## Difficulty

**Intermediate** — Prerequisites: AI Foundations (embeddings, context windows)
