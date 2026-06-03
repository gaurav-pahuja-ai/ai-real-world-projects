"""
Enterprise Knowledge Base Search
==================================
Multi-tenant RAG system with access-control-aware retrieval.
Supports Confluence, Notion, and Google Drive document sources.

Setup:
    pip install -r requirements.txt
    cp .env.example .env
    python main.py

API runs at http://localhost:8000
Docs at  http://localhost:8000/docs
"""

import os
from typing import Optional

import uvicorn
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException
from openai import OpenAI
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

load_dotenv()

app = FastAPI(title="Enterprise Knowledge Base Search")
client = OpenAI()
qdrant = QdrantClient(":memory:")  # use QdrantClient(url="...") for production

COLLECTION = "knowledge_base"
EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o"
VECTOR_SIZE = 1536

qdrant.recreate_collection(
    collection_name=COLLECTION,
    vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
)


# ── Models ───────────────────────────────────────────────────

class IndexRequest(BaseModel):
    text: str
    source: str        # e.g. "confluence", "notion", "gdrive"
    doc_id: str
    tenant_id: str
    allowed_users: list[str]   # access control list


class SearchRequest(BaseModel):
    query: str
    tenant_id: str
    user_id: str
    top_k: int = 5


class SearchResponse(BaseModel):
    answer: str
    sources: list[str]


# ── Helper ───────────────────────────────────────────────────

def embed(text: str) -> list[float]:
    return client.embeddings.create(
        input=text, model=EMBED_MODEL
    ).data[0].embedding


# ── Routes ───────────────────────────────────────────────────

@app.post("/index")
def index_document(req: IndexRequest):
    """Ingest a document with tenant and access metadata."""
    vector = embed(req.text)
    qdrant.upsert(
        collection_name=COLLECTION,
        points=[
            PointStruct(
                id=abs(hash(req.doc_id)) % (10**9),
                vector=vector,
                payload={
                    "text": req.text,
                    "source": req.source,
                    "doc_id": req.doc_id,
                    "tenant_id": req.tenant_id,
                    "allowed_users": req.allowed_users,
                },
            )
        ],
    )
    return {"status": "indexed", "doc_id": req.doc_id}


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest):
    """Search with per-user access control filtering."""
    query_vector = embed(req.query)

    results = qdrant.search(
        collection_name=COLLECTION,
        query_vector=query_vector,
        limit=req.top_k * 3,  # fetch extra, then filter
    )

    # Access control: keep only docs this user can see
    accessible = [
        r for r in results
        if r.payload.get("tenant_id") == req.tenant_id
        and req.user_id in r.payload.get("allowed_users", [])
    ][:req.top_k]

    if not accessible:
        raise HTTPException(status_code=404, detail="No accessible documents found.")

    context = "\n\n---\n\n".join(r.payload["text"] for r in accessible)
    sources = list({r.payload["source"] for r in accessible})

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant. Answer using only the provided context. "
                    f"Context:\n{context}"
                ),
            },
            {"role": "user", "content": req.query},
        ],
    )

    return SearchResponse(
        answer=response.choices[0].message.content,
        sources=sources,
    )


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
