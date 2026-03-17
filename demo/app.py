from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

from endee_client import EndeeClient, chunk_text


DEFAULT_INDEX_NAME = os.getenv("ENDEE_INDEX_NAME", "tapacademy_rag_demo")
ENDEE_BASE_URL = os.getenv("ENDEE_BASE_URL", "http://localhost:8080")
ENDEE_AUTH_TOKEN = os.getenv("ENDEE_AUTH_TOKEN") or None

EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")


app = FastAPI(title="Endee RAG Demo", version="1.0.0")
model = SentenceTransformer(EMBED_MODEL_NAME)
client = EndeeClient(base_url=ENDEE_BASE_URL, auth_token=ENDEE_AUTH_TOKEN)


class IngestRequest(BaseModel):
    documents: list[dict[str, Any]] = Field(
        ...,
        description="List of {id?, source, text}. id is optional; will be auto-generated per chunk.",
    )
    chunk_size: int = 800
    overlap: int = 120


class SearchRequest(BaseModel):
    query: str
    k: int = 5
    source: str | None = Field(default=None, description="Optional filter: only search within this source.")


@app.get("/health")
def health() -> dict[str, Any]:
    return {"app": "ok", "endee": client.health()}


@app.post("/setup")
def setup() -> dict[str, Any]:
    dim = int(model.get_sentence_embedding_dimension())
    client.create_index(index_name=DEFAULT_INDEX_NAME, dim=dim, space_type="cosine", precision="int16")
    return {"status": "ok", "index_name": DEFAULT_INDEX_NAME, "dim": dim}


@app.post("/ingest")
def ingest(req: IngestRequest) -> dict[str, Any]:
    dim = int(model.get_sentence_embedding_dimension())
    client.create_index(index_name=DEFAULT_INDEX_NAME, dim=dim, space_type="cosine", precision="int16")

    vectors: list[dict[str, Any]] = []
    now = datetime.utcnow().isoformat() + "Z"

    for doc in req.documents:
        source = str(doc.get("source") or "unknown")
        text = str(doc.get("text") or "")
        base_id = str(doc.get("id") or source)

        for chunk_idx, chunk in enumerate(chunk_text(text, chunk_size=req.chunk_size, overlap=req.overlap)):
            chunk_id = f"{base_id}::chunk::{chunk_idx}"
            emb = model.encode(chunk, normalize_embeddings=True).tolist()
            meta = json.dumps(
                {"source": source, "chunk_idx": chunk_idx, "text": chunk, "ingested_at": now},
                ensure_ascii=False,
            )
            vectors.append(
                {
                    "id": chunk_id,
                    "vector": emb,
                    "meta": meta,
                    "filter": json.dumps({"source": source}),
                    "norm": 1.0,
                }
            )

    if vectors:
        client.insert_vectors_json(index_name=DEFAULT_INDEX_NAME, vectors=vectors)

    return {"status": "ok", "index_name": DEFAULT_INDEX_NAME, "inserted": len(vectors)}


@app.post("/search")
def search(req: SearchRequest) -> dict[str, Any]:
    q = req.query.strip()
    emb = model.encode(q, normalize_embeddings=True).tolist()

    filter_array = None
    if req.source:
        filter_array = [{"source": {"$eq": req.source}}]

    results = client.search_dense(
        index_name=DEFAULT_INDEX_NAME,
        vector=emb,
        k=req.k,
        include_vectors=False,
        filter=filter_array,
    )

    for r in results:
        meta = r.get("meta")
        if isinstance(meta, str):
            try:
                r["meta"] = json.loads(meta)
            except Exception:
                pass

    return {"query": q, "k": req.k, "results": results}

