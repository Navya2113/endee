<div align="center">

# 🔷 AI Resume Analyzer

**Semantic resume-to-job-description matching via RAG**  
*Built for the Endee.io engineering assessment*

[![Endee](https://img.shields.io/badge/Vector%20DB-Endee-6366f1?style=flat-square)](https://endee.io)
[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688?style=flat-square)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/Frontend-React%2018-61dafb?style=flat-square)](https://react.dev)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue?style=flat-square)](LICENSE)

</div>

---

## 📋 Overview

AI Resume Analyzer is a full-stack **Retrieval-Augmented Generation (RAG)** application that compares a candidate's resume against a job description and instantly returns:

| Output | Description |
|--------|-------------|
| **Match %** | Semantic similarity score derived from cosine search in Endee |
| **Matched Skills** | Skills present in *both* the resume and job description |
| **Missing Skills** | Skills required by the JD but absent from the resume |
| **Summary** | Human-readable verdict with actionable advice |

Unlike keyword counters, the app encodes entire documents as dense vectors — capturing *meaning*, not just word overlap — making it robust to synonyms, rephrasing, and varied terminology.

---

## ✨ Features

- 🧠 **Dense semantic embeddings** — `all-MiniLM-L6-v2` (384-dim), runs fully locally
- 🗄️ **Endee as the vector store** — official Python SDK, cosine index, persistent across sessions
- 🔍 **RAG pipeline** — embed → upsert → query → blend scores → return report
- 🏷️ **Skill taxonomy** — 60+ curated tech and soft skills with regex extraction
- 🎯 **Filtered queries** — resume vectors tagged with `doc_type` filter for future analytics
- ⚡ **Fast** — typical end-to-end latency < 800 ms after model warm-up
- 🐳 **Docker Compose** — one command spins up Endee + backend + frontend
- 📊 **History endpoint** — `GET /history` returns Endee index stats

---

## 🛠 Tech Stack

```
Layer          Technology
─────────────────────────────────────────────────────
Frontend       React 18 · CSS (Space Grotesk + JetBrains Mono)
Backend        FastAPI (Python 3.11+) · Uvicorn ASGI
Embeddings     sentence-transformers — all-MiniLM-L6-v2 (384d)
Vector DB      Endee  ←  official Python SDK  (pip install endee)
Containerise   Docker Compose
```

---

## 🏗 Architecture

```
┌──────────────────────────────────────────────────────────┐
│                      Browser (React)                     │
│  ┌──────────────┐          ┌──────────────────────────┐  │
│  │  Resume text │          │  Job Description text    │  │
│  └──────┬───────┘          └────────────┬─────────────┘  │
└─────────┼────────────────────────────── ┼ ───────────────┘
          │        POST /analyze          │
          └───────────────┬───────────────┘
                          ▼
┌──────────────── FastAPI Backend ─────────────────────────┐
│                                                          │
│  1.  SentenceTransformer.encode(resume)  → vec_r [384]   │
│  2.  SentenceTransformer.encode(jd)      → vec_j [384]   │
│                                                          │
│  3.  endee_index.upsert([{               ──────────────► │──┐
│         id: uuid,                                        │  │
│         vector: vec_r,                                   │  │
│         meta: {preview, skills},                         │  │
│         filter: {doc_type: "resume"}                     │  │
│      }])                                                 │  │
│                                                          │  │
│  4.  results = endee_index.query(        ◄──────────────── ┘│
│         vector=vec_j, top_k=5, ef=128)                   │
│                                                          │  Endee
│  5.  blended_score =                                     │  Vector DB
│        cosine(vec_r, vec_j) * 0.65                       │  (localhost:8080)
│        + results[0].similarity * 0.35                    │
│                                                          │
│  6.  match_pct = (blended + 1) / 2 * 100                 │
│  7.  matched  = resume_skills ∩ jd_skills                │
│  8.  missing  = jd_skills ─ resume_skills                │
│                                                          │
│  9.  return AnalyzeResponse(...)         ─────────────► │
└──────────────────────────────────────────────────────────┘
                          │
                          ▼
              Browser renders score ring,
              matched/missing skill tags
```

---

## 🔷 How Endee Is Used

This project uses **Endee as the primary vector store** in the RAG pipeline:

### 1. Index Creation
At startup the backend calls `client.list_indexes()`. If the `resume_analyzer` index doesn't exist it creates one:
```python
client.create_index(
    name="resume_analyzer",
    dimension=384,           # all-MiniLM-L6-v2 output dim
    space_type="cosine",     # cosine similarity
    precision=Precision.INT8 # quantised for speed
)
```

### 2. Vector Upsert
Every analyzed resume is stored in Endee with metadata and a filter field:
```python
index.upsert([{
    "id":     str(uuid.uuid4()),
    "vector": embedder.encode(resume_text).tolist(),
    "meta":   {"preview": resume_text[:300], "skills": "python, aws …"},
    "filter": {"doc_type": "resume"},   # enables filtered queries
}])
```

### 3. Similarity Query
The job description is embedded and queried against Endee's HNSW index:
```python
results = index.query(vector=jd_vector, top_k=5, ef=128)
# results[i] → {"id": "…", "similarity": 0.87, "meta": {…}}
```

### 4. Score Blending
Endee's score and the direct cosine are blended for best accuracy:
```python
blended = direct_cosine * 0.65 + endee_similarity * 0.35
match_pct = (blended + 1) / 2 * 100          # → 0–100 %
```

> **Why Endee?**  
> Endee handles up to 1B vectors on a single node with HNSW indexing and SIMD-optimised distance computation — ideal for high-throughput resume screening or building a candidate shortlisting engine over millions of resumes.

---

## 📁 Project Structure

```
resume-analyzer/
├── backend/
│   ├── main.py            ← FastAPI app (Endee SDK, embeddings, analysis)
│   ├── requirements.txt   ← Python deps including `endee`
│   └── Dockerfile
├── frontend/
│   ├── src/
│   │   ├── App.js         ← React UI with pipeline indicator
│   │   ├── App.css        ← Deep-space dark theme
│   │   └── index.js
│   ├── public/index.html
│   ├── nginx.conf         ← Reverse proxy for API routes
│   ├── package.json
│   └── Dockerfile
├── docker-compose.yml     ← Endee + backend + frontend
└── README.md
```

---

## 🚀 Setup Instructions

### Prerequisites

| Tool | Version |
|------|---------|
| Docker | 20.10+ |
| Docker Compose | v2 |
| Python | 3.11+ (local dev only) |
| Node.js | 18+ (local dev only) |

---

### Option A — Docker Compose (Recommended)

```bash
# 1. Clone your fork
git clone https://github.com/<YOUR-USERNAME>/endee
cd endee

# 2. Start all three services
docker compose up --build

# Services:
#   Frontend  →  http://localhost:3000
#   Backend   →  http://localhost:8000
#   Endee     →  http://localhost:8080
```

---

### Option B — Local Development

**Step 1: Start Endee**
```bash
docker run -d \
  -p 8080:8080 \
  -v endee-data:/data \
  --name endee-server \
  endeeio/endee-server:latest
```

**Step 2: Backend**
```bash
cd backend
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

> First run downloads `all-MiniLM-L6-v2` (~90 MB) from HuggingFace — cached after that.

**Step 3: Frontend**
```bash
cd frontend
npm install
npm start
# Opens http://localhost:3000
```

---

## 📡 API Reference

### `POST /analyze`

```json
// Request
{
  "resume_text": "John Doe … Python, React, AWS, Docker …",
  "job_description": "We are hiring a senior Python engineer with AWS and Kubernetes …"
}

// Response
{
  "match_percentage": 74.2,
  "relevant_keywords": ["python", "aws", "docker", "react"],
  "missing_skills":   ["kubernetes", "terraform", "ci/cd"],
  "summary": "Good match ✅ A few gaps to address before applying. 4 skills matched, 3 skills missing.",
  "resume_id": "3f2d8c1a-…"
}
```

### `GET /health`
```json
{ "status": "ok", "embedding_model": "all-MiniLM-L6-v2", "vector_db": "http://localhost:8080/api/v1", "index": "resume_analyzer" }
```

### `GET /history`
Returns Endee index stats (total vectors stored, dimension, etc.)

---

## 🔮 Future Extensions

- **Batch screening** — upsert hundreds of resumes, query with a single JD to rank candidates
- **Filtered search** — query only resumes for a specific tech stack using Endee's `$eq` / `$in` filters
- **Hybrid search** — combine dense embeddings with Endee BM25 sparse vectors for higher precision
- **Multi-JD comparison** — store multiple JDs and find the best-fit role for a given resume

---

## 📜 License

Apache 2.0 — see [LICENSE](LICENSE)

---

<div align="center">
Built with ❤ using <a href="https://endee.io">Endee</a> · FastAPI · sentence-transformers · React
</div>
