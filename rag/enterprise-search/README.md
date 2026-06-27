# Enterprise Search

Ask questions about company documents. Different users see different documents based on their access level.

This project builds the full retrieval pipeline used in production by Azure AI Search, Elasticsearch, and Weaviate. The difference is that here you can see every step. In those managed services the same steps are hidden behind a config flag.

---

## What does this do?

You can upload PDFs and assign them to specific users. When someone asks a question, the app only searches the documents that user is allowed to see, and answers using those documents.

Five sample company documents are pre-loaded at startup so you can try it immediately without uploading anything.

| User | Can see |
|---|---|
| alice | Salary bands, Sales targets, Company handbook |
| bob | Engineering ADR, Incident report, Company handbook |
| charlie | Sales targets, Company handbook |
| admin | All documents |

Switch between users and ask the same question. You will see how the answer and the sources change based on who is asking.

---

## The retrieval pipeline

Most people think search is simple: find documents that contain the keywords. Real production search systems do something more sophisticated.

This project implements the exact pipeline that Azure AI Search calls "hybrid + semantic" when you enable it in your index configuration:

```
Your question
    |
+---+---+
|       |
BM25    Vector search
search  (Qdrant + Google
        text-embedding-004)
|       |
+---+---+
    |
RRF merge
(Reciprocal Rank Fusion)
    |
Access control filter
(tenant + per-document ACL)
    |
Top 5 chunks
    |
Gemini Flash answer
    |
Streamed reply with source citations
```

**Why two retrievers?**

BM25 is great at exact keywords: part numbers, error codes, names, legal terms. If you search for "NatWest Group" or "ADR-007", BM25 will find those chunks reliably.

Vector search understands meaning. If you ask "what went wrong with the database last month?", vector search can match that to the incident report even though it does not contain those exact words.

Neither retriever is always best. Running both and merging the results with RRF consistently outperforms either one alone. This is why every serious production search system uses this pattern.

**What is RRF?**

Reciprocal Rank Fusion combines two ranked lists into one using the formula:

```
score(chunk) += 1 / (rank_in_list + 60)
```

A chunk that appears in the top 5 of both lists scores higher than one that appears in only one list. The constant 60 comes from the original 2009 paper by Cormack et al. and is the default value used by Elasticsearch and Qdrant unchanged.

**What about reranking?**

Reranking is a second-pass neural model that re-scores a shortlist of already-retrieved chunks. It is more accurate but more expensive.

Azure AI Search calls this "semantic ranker". Cohere sells a dedicated reranking API. You can also run a local CrossEncoder model via sentence-transformers.

This project deliberately stops at RRF to keep everything on the free tier. The README for the AI Learning Hub lesson explains how reranking fits in as the natural next step.

---

## What runs where

| Step | Where it runs |
|---|---|
| PDF reading and chunking | Your computer |
| BM25 keyword indexing | Your computer, in memory |
| Qdrant vector store | Your computer, in memory |
| Embedding each chunk | Google text-embedding-004 API (free tier) |
| Embedding your question | Google text-embedding-004 API (free tier) |
| RRF merge | Your computer, pure Python |
| Access control filtering | Your computer |
| Generating the answer | Gemini Flash API (free tier) |

One `GEMINI_API_KEY` covers everything. No paid services, no credit card required.

---

## Why Gemini and not OpenAI?

The original version of this project used OpenAI GPT-4o, which requires a paid account.

This version uses Google Gemini Flash and text-embedding-004, both of which are free up to generous daily limits. The goal of this project is to be usable by anyone learning, not just people with a company credit card.

In production, the choice of LLM and embedding model is separate from the retrieval pipeline. You can drop in any model without changing the BM25, Qdrant, or RRF code.

---

## How managed services compare

Once you understand what this project does, you will recognise it everywhere:

| Service | How it works internally |
|---|---|
| Azure AI Search | BM25 built-in, vector fields per document, hybrid search merges with RRF, semantic ranker adds a second-pass rerank |
| Elasticsearch | BM25 by default, kNN vector search added per field, hybrid with RRF via `hybrid` query type |
| Weaviate | BM25 and vector search run in parallel, hybrid mode uses RRF to merge |
| Qdrant (hosted) | Sparse + dense vectors in one query, RRF fusion built-in |

If you use Azure AI Search at work, this project shows you what is happening behind the scenes when you set `queryType: "semantic"` and add a `vectorQueries` block to your search request.

---

## Setup

```bash
cd rag/enterprise-search
pip install -r requirements.txt
cp .env.example .env
# Add your Gemini API key to .env
python main.py
```

Get a free Gemini API key at https://aistudio.google.com/app/apikey

The app opens in your browser. Five sample documents are loaded automatically. Switch between users and ask questions to explore the access control and retrieval in action.

---

## Stack

| Component | Library | Version |
|---|---|---|
| Keyword search | rank-bm25 | latest |
| Vector store | qdrant-client | 1.9+ |
| Embeddings | google-generativeai (text-embedding-004) | 0.7+ |
| Answer generation | google-generativeai (Gemini Flash) | 0.7+ |
| PDF reading | pypdf | 4.0+ |
| UI | gradio | 4.40+ |

---

## Difficulty

Advanced. You should be comfortable with Python and have worked through the BM25 Keyword Bot and Semantic Vector Bot projects first. Those two projects explain the individual retrieval methods. This project combines them.
