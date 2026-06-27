# Document Q&A Bot

Upload a PDF. Ask questions about it. Get answers with source citations, streamed word by word in real time.

This project uses semantic search, which means it understands the meaning of your question, not just the exact words. It finds relevant text even if you phrase things differently from how the document was written.

---

## What does this do?

You upload a PDF. The app reads it, cuts it into chunks, and converts each chunk into a list of numbers called a vector. These numbers capture the meaning of the text. When you ask a question, your question is also converted to numbers. The app finds the chunks whose numbers are closest to your question's numbers. Those chunks go to Gemini Flash, which writes your answer.

The embedding step runs fully on your CPU. No embedding API key needed.

---

## How it works, step by step

```
INDEXING (runs once when you upload a PDF)

1. You upload a PDF
        |
        v
2. The app reads all the text from every page
        |
        v
3. The text is cut into overlapping chunks (1500 characters, 150 overlap)
        |
        v
4. ChromaDB converts each chunk into a vector using a local AI model
   (all-MiniLM-L6-v2, runs on your CPU, no API call needed)
        |
        v
5. The vectors and chunk text are saved to disk in ./chroma_db
   (they survive restarts, no need to re-upload)
```

```
ANSWERING (runs every time you ask a question)

1. You type a question
        |
        v
2. ChromaDB converts your question into a vector (same local model)
        |
        v
3. ChromaDB finds the 5 chunks whose vectors are closest to your question
   (this is cosine similarity search, meaning "most similar in meaning")
        |
        v
4. Those 5 chunks are packaged into a prompt for Gemini Flash
        |
        v
5. Gemini writes the answer and streams it to you word by word
        |
        v
6. The source PDF filenames are shown at the bottom of the answer
```

---

## What is a vector?

A vector is a long list of numbers like `[0.12, -0.87, 0.34, ...]`. Each number describes one aspect of the meaning of a piece of text. Two chunks that talk about similar things will have similar numbers. ChromaDB uses this to find the most relevant chunks for your question, even if you used completely different words.

This is why it handles natural language questions better than keyword search.

---

## Tech stack

| Part | Tool |
|---|---|
| PDF reading | pypdf |
| Embeddings model | all-MiniLM-L6-v2 via ONNX (runs locally on CPU) |
| Vector database | ChromaDB (stored on disk in ./chroma_db) |
| Similarity search | Cosine similarity via ChromaDB HNSW index |
| Answer generation | Google Gemini Flash (free tier, streaming) |
| UI | Gradio |

---

## Setup

**Step 1. Get a free Gemini API key**

Go to [aistudio.google.com/apikey](https://aistudio.google.com/apikey) and create a key. No billing required.

**Step 2. Create a virtual environment and install dependencies**

```bash
cd rag/document-qa-bot
python -m venv venv

# On Mac or Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate

pip install -r requirements.txt
```

On the very first run, ChromaDB will download the `all-MiniLM-L6-v2` model (about 90 MB). This happens once and is cached automatically. After that, everything runs offline.

**Step 3. Add your API key**

```bash
cp .env.example .env
```

Open the `.env` file and set your key:

```
GOOGLE_API_KEY=your_key_here
```

**Step 4. Run the app**

```bash
python main.py
```

Open your browser at `http://localhost:7867`

---

## How to use it

1. Click **Upload PDF** and pick a file
2. Click **Index Document** and wait for the confirmation message
3. Type a question in the chat box and press **Send** or hit Enter
4. The answer streams in word by word with source citations at the end
5. Click **Clear** to delete the vector index and start fresh with a new PDF
6. Click **Clear chat** to reset just the conversation, without losing the index

You can index multiple PDFs. All chunks are stored persistently and survive restarts.

---

## Project structure

```
document-qa-bot/
|-- main.py           Full app: indexing, retrieval, Gradio UI
|-- requirements.txt  Python dependencies
|-- .env.example      Template for your API key
|-- .env              Your actual API key (not committed to git)
|-- chroma_db/        Persistent vector store (created automatically on first run)
```

---

## Key settings you can change in main.py

| Setting | Default | What it does |
|---|---|---|
| CHUNK_SIZE | 1500 | How many characters per chunk |
| CHUNK_OVERLAP | 150 | How much chunks overlap at boundaries |
| top_k | 5 | How many chunks are sent to Gemini |
| hnsw:space | cosine | How similarity is measured |

---

## How this is different from the BM25 bot

| Question | BM25 Bot | This Bot |
|---|---|---|
| How does it find relevant text? | Counts exact keyword matches | Compares meaning using vectors |
| Does it understand synonyms? | No | Yes |
| Does the index survive restarts? | No (in memory only) | Yes (saved to disk) |
| Does it need a GPU? | No | No (CPU is enough) |
| Downloads a model? | No | Yes, once (90 MB) |
| Best for | Exact terms, codes, names | Natural language questions |

---

## What you learn from this project

- How text is turned into numbers (embeddings)
- How cosine similarity finds related chunks
- How a persistent vector database stores and retrieves data
- How to build a prompt with retrieved context
- How to stream responses from an LLM
- How to handle API rate limits gracefully with retries
