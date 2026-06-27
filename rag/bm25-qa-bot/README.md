# BM25 Document Q&A Bot

Ask questions about any PDF. Get answers instantly. No cloud APIs. No AI model needed for search. Just pure keyword matching and Gemini Flash for the final answer.

---

## What does this do?

You upload a PDF. The app reads it, chops it into small pieces (called chunks), and builds a keyword index from those pieces. When you ask a question, it scans the index for chunks that contain your keywords, picks the best ones, and hands them to Gemini Flash. Gemini reads those chunks and writes a clean answer.

The whole search step runs on your computer with zero API calls.

---

## How it works, step by step

```
1. You upload a PDF
        |
        v
2. The app reads all the text from every page
        |
        v
3. The text is cut into overlapping chunks (1500 characters each)
        |
        v
4. Every chunk is tokenised (split into lowercase words, punctuation removed)
        |
        v
5. A BM25 index is built from all those word lists (lives in memory)
        |
        v
6. You type a question
        |
        v
7. Your question is also tokenised into words
        |
        v
8. BM25 scores every chunk by how well it matches your words
        |
        v
9. The top-K highest scoring chunks are picked
        |
        v
10. Those chunks are sent to Gemini Flash as context
        |
        v
11. Gemini writes the answer and streams it back to you
```

No vectors. No embeddings. No GPU. No extra API keys beyond Gemini.

---

## What is BM25?

BM25 is a scoring algorithm used by search engines like Elasticsearch and early Google. It works like a librarian who counts how many times your search words appear in each document, gives extra points for rare words, and penalises documents that are unusually long.

It does not understand meaning. It counts words.

That means it is great when you search with exact terms that appear in the document (product codes, names, technical terms). It struggles when you rephrase something or use synonyms.

---

## Tech stack

| Part | Tool |
|---|---|
| PDF reading | pypdf |
| Keyword search | rank-bm25 (BM25Okapi) |
| Index storage | Plain Python list in memory |
| Answer generation | Google Gemini Flash (free tier) |
| UI | Gradio |

---

## Setup

**Step 1. Get a free Gemini API key**

Go to [aistudio.google.com/apikey](https://aistudio.google.com/apikey) and create a key. No billing required.

**Step 2. Create a virtual environment and install dependencies**

```bash
cd rag/bm25-qa-bot
python -m venv venv

# On Mac or Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate

pip install -r requirements.txt
```

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

Open your browser at `http://localhost:7866`

---

## How to use it

1. Click **Upload PDF** and pick a file
2. Click **Index Document** and wait for the confirmation message
3. Type your question in the chat box and press **Send** or hit Enter
4. Read the answer. Sources are listed at the bottom of every reply
5. Click **Clear Index** to remove everything and start fresh with a new file
6. Use the **Top-K** slider to control how many chunks are sent to Gemini (more chunks = more context but slower)

You can upload multiple PDFs. Each one adds its chunks to the same index.

---

## Tips for better results

- Use words that are actually in the PDF. BM25 matches keywords exactly.
- If you get no results, try rephrasing with more specific terms from the document.
- Increase Top-K if the answer feels incomplete.
- Decrease Top-K if the answer feels unfocused.

---

## Project structure

```
bm25-qa-bot/
|-- main.py           Full app: ingestion, BM25 index, retrieval, Gradio UI
|-- requirements.txt  Python dependencies
|-- .env.example      Template for your API key
|-- .env              Your actual API key (not committed to git)
```

---

## When to use BM25 vs a vector database

| Situation | Use |
|---|---|
| Searching for exact names, codes, or technical terms | BM25 (this project) |
| Asking questions in natural language with synonyms | Vector search (document-qa-bot) |
| No GPU, no internet, instant startup needed | BM25 (this project) |
| Permanent storage that survives restarts | Vector search (document-qa-bot) |
| Learning how RAG works without extra complexity | BM25 (this project) |
