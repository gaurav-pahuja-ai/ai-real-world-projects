# AI Real-World Projects

> Hands-on projects that cement every concept from [AI Learning Hub](https://github.com/MrPahuja/ai-learning-hub).

Each project is self-contained, well-documented, and production-oriented — not toy examples.

---

## Repository structure

```
ai-real-world-projects/
│
├── rag/
│   ├── document-qa-bot/          # PDF Q&A with ChromaDB + Gemini
│   └── enterprise-search/        # Multi-tenant RAG with access control
│
├── mcp/
│   ├── github-server/            # MCP server exposing GitHub operations
│   └── postgres-server/          # MCP server for safe Postgres querying
│
└── agentic/
    ├── research-agent/           # ReAct agent for autonomous research
    └── code-review-agent/        # Multi-agent PR code reviewer
```

---

## Projects

### RAG

#### `rag/document-qa-bot`
A production-ready document Q&A system.
- **Stack:** Python · ChromaDB · Google Gemini · Gradio
- **Concepts:** Semantic chunking, hybrid BM25+vector search, source citation, streaming
- **Difficulty:** Intermediate

#### `rag/enterprise-search`
Multi-tenant knowledge base search with access control.
- **Stack:** Python · Qdrant · FastAPI · LangChain · OAuth2
- **Concepts:** Multi-source ingestion, per-user access-aware retrieval, Cohere re-ranking
- **Difficulty:** Advanced

---

### MCP

#### `mcp/github-server`
Full-featured MCP server exposing GitHub as tools.
- **Stack:** Python · MCP SDK · GitHub API · asyncio
- **Concepts:** MCP tool/resource primitives, stdio + SSE transport, OAuth handling
- **Difficulty:** Intermediate

#### `mcp/postgres-server`
Safe Postgres querying via MCP.
- **Stack:** Python · MCP SDK · PostgreSQL · asyncpg
- **Concepts:** Read-only enforcement, schema discovery as resource, query validation
- **Difficulty:** Intermediate

---

### Agentic AI

#### `agentic/research-agent`
Autonomous multi-step research agent.
- **Stack:** Python · OpenAI · Tavily · LangGraph
- **Concepts:** ReAct loop, web search + PDF tools, source deduplication, structured output
- **Difficulty:** Advanced

#### `agentic/code-review-agent`
Parallel multi-agent code reviewer integrated with GitHub PRs.
- **Stack:** Python · OpenAI · GitHub API · LangGraph
- **Concepts:** Orchestrator-worker pattern, parallel agents, human-approval gate
- **Difficulty:** Advanced

---

## Getting started

Each project has its own `README.md` with setup instructions. Generally:

```bash
cd rag/document-qa-bot
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env  # add your API keys
python main.py
```

## Prerequisites

- Python 3.11+
- A Google API key for `document-qa-bot` (free at [aistudio.google.com/apikey](https://aistudio.google.com/apikey))
- An OpenAI API key (for other projects)
- Docker (optional, for containerised projects)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). All skill levels welcome — each project has clearly labelled "good first issue" improvements.

## License

MIT
