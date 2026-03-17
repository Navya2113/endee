# Endee Semantic Search + Mini RAG Demo (Tap Academy Assignment)

This folder contains a **complete AI/ML mini-project** built **on top of Endee** as the vector database.

It demonstrates:

- **Semantic search** over ingested documents using embeddings
- **RAG-style retrieval** (retrieve top-\(k\) chunks that can be used as context for an LLM)
- **Metadata filtering** (example: search within a specific `source`)

## System Design (High Level)

1. **Endee server (vector DB)** runs on `localhost:8080` (Docker on Windows).
2. **Demo API (FastAPI)** provides:
   - `/setup` create an Endee index (dimension matches embedding model)
   - `/ingest` chunk text → embed → store vectors in Endee
   - `/search` embed query → Endee KNN search → return top matches
3. **Embedding model**: `sentence-transformers/all-MiniLM-L6-v2` (384 dims, runs locally after model download).

### Data stored in Endee

Each chunk is inserted as a vector with:

- `id`: `<doc_id>::chunk::<chunk_idx>`
- `vector`: embedding array (float list)
- `meta`: JSON string containing `{source, chunk_idx, text, ingested_at}`
- `filter`: JSON string (example: `{"source":"notes.md"}`)

## Setup

### 1) Start Endee (Windows: Docker required)

From the repo root:

```bash
docker compose up -d
```

Verify:

```bash
curl http://localhost:8080/api/v1/health
```

### 2) Run the demo API

From `demo/`:

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
uvicorn app:app --reload --port 8000
```

## Quick Demo

### Create the index

```bash
curl -X POST http://localhost:8000/setup
```

### Ingest some text

```bash
curl -X POST http://localhost:8000/ingest ^
  -H "Content-Type: application/json" ^
  -d "{\"documents\":[{\"source\":\"sample.txt\",\"text\":\"Endee is a high-performance vector database for semantic search and RAG.\"}]}"
```

### Search

```bash
curl -X POST http://localhost:8000/search ^
  -H "Content-Type: application/json" ^
  -d "{\"query\":\"What is Endee used for?\",\"k\":5}"
```

### Optional: filter by source

```bash
curl -X POST http://localhost:8000/search ^
  -H "Content-Type: application/json" ^
  -d "{\"query\":\"vector database\",\"k\":5,\"source\":\"sample.txt\"}"
```

## Configuration

Environment variables:

- `ENDEE_BASE_URL` (default `http://localhost:8080`)
- `ENDEE_AUTH_TOKEN` (optional; only if Endee auth is enabled)
- `ENDEE_INDEX_NAME` (default `tapacademy_rag_demo`)
- `EMBED_MODEL_NAME` (default `sentence-transformers/all-MiniLM-L6-v2`)

