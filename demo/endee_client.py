from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import msgpack
import requests


@dataclass(frozen=True)
class EndeeClient:
    base_url: str = "http://localhost:8080"
    auth_token: str | None = None
    username: str = "admin"

    def _headers(self, *, content_type: str | None = None) -> dict[str, str]:
        headers: dict[str, str] = {}
        if self.auth_token:
            headers["Authorization"] = self.auth_token
        if content_type:
            headers["Content-Type"] = content_type
        return headers

    def health(self) -> dict[str, Any]:
        r = requests.get(f"{self.base_url}/api/v1/health", timeout=30)
        r.raise_for_status()
        return r.json()

    def list_indexes(self) -> dict[str, Any]:
        r = requests.get(
            f"{self.base_url}/api/v1/index/list",
            headers=self._headers(),
            timeout=30,
        )
        r.raise_for_status()
        return r.json()

    def create_index(
        self,
        *,
        index_name: str,
        dim: int,
        space_type: str = "cosine",
        precision: str = "int16",
        M: int | None = None,
        ef_con: int | None = None,
    ) -> None:
        body: dict[str, Any] = {
            "index_name": index_name,
            "dim": dim,
            "space_type": space_type,
            "precision": precision,
        }
        if M is not None:
            body["M"] = M
        if ef_con is not None:
            body["ef_con"] = ef_con

        r = requests.post(
            f"{self.base_url}/api/v1/index/create",
            headers=self._headers(content_type="application/json"),
            data=json.dumps(body),
            timeout=60,
        )
        if r.status_code in (200, 409):
            return
        r.raise_for_status()

    def insert_vectors_json(self, *, index_name: str, vectors: list[dict[str, Any]]) -> None:
        r = requests.post(
            f"{self.base_url}/api/v1/index/{index_name}/vector/insert",
            headers=self._headers(content_type="application/json"),
            data=json.dumps(vectors),
            timeout=120,
        )
        r.raise_for_status()

    def search_dense(
        self,
        *,
        index_name: str,
        vector: list[float],
        k: int = 5,
        ef: int | None = None,
        include_vectors: bool = False,
        filter: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        body: dict[str, Any] = {
            "k": k,
            "vector": vector,
            "include_vectors": include_vectors,
        }
        if ef is not None:
            body["ef"] = ef
        if filter is not None:
            body["filter"] = json.dumps(filter)

        r = requests.post(
            f"{self.base_url}/api/v1/index/{index_name}/search",
            headers=self._headers(content_type="application/json"),
            data=json.dumps(body),
            timeout=60,
        )
        r.raise_for_status()

        if r.headers.get("Content-Type", "").startswith("application/msgpack"):
            payload = msgpack.unpackb(r.content, raw=False)
        else:
            payload = r.json()

        results: Any = payload
        if isinstance(payload, dict) and "results" in payload:
            results = payload["results"]
        elif isinstance(payload, (list, tuple)) and len(payload) == 1:
            results = payload[0]
        if not isinstance(results, list):
            results = []

        out: list[dict[str, Any]] = []
        for item in results:
            meta = item.get("meta")
            if isinstance(meta, (bytes, bytearray)):
                try:
                    meta = meta.decode("utf-8")
                except Exception:
                    meta = None
            out.append(
                {
                    "id": item.get("id"),
                    "similarity": item.get("similarity"),
                    "filter": item.get("filter"),
                    "meta": meta,
                    "vector": item.get("vector") if include_vectors else None,
                }
            )
        return out


def chunk_text(text: str, *, chunk_size: int = 800, overlap: int = 120) -> list[str]:
    if chunk_size <= 0:
        return [text]
    if overlap < 0:
        overlap = 0

    chunks: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        end = min(n, i + chunk_size)
        chunks.append(text[i:end])
        if end >= n:
            break
        i = max(0, end - overlap)
    return chunks
