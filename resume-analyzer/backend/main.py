"""
============================================================
 AI Resume Analyzer — FastAPI Backend
 Vector DB : Endee  (https://github.com/endee-io/endee)
 Embeddings: sentence-transformers / all-MiniLM-L6-v2
============================================================
"""

import re
import uuid
import logging
from typing import List

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

# Official Endee Python SDK
from endee import Endee, Precision

# ─── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ─── FastAPI app ─────────────────────────────────────────────────────────────
app = FastAPI(
    title="AI Resume Analyzer",
    description=(
        "Semantic resume ↔ job-description matching using RAG. "
        "Embeddings stored in and queried from Endee vector database."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Config ──────────────────────────────────────────────────────────────────
ENDEE_URL       = "http://localhost:8080/api/v1"
ENDEE_TOKEN     = ""               # set if you started Endee with NDD_AUTH_TOKEN
INDEX_NAME      = "resume_analyzer"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM   = 384              # dimension for all-MiniLM-L6-v2

# ─── Load sentence-transformer once at startup ───────────────────────────────
log.info(f"Loading embedding model: {EMBEDDING_MODEL} …")
embedder = SentenceTransformer(EMBEDDING_MODEL)
log.info("Embedding model ready.")

# ─── Endee client (initialised at startup) ───────────────────────────────────
endee_client: Endee = None
endee_index          = None


def get_endee():
    """Return (creating if necessary) the Endee client and index."""
    global endee_client, endee_index
    if endee_client is None:
        endee_client = Endee(ENDEE_TOKEN)          # token="" → no-auth mode
        endee_client.set_base_url(ENDEE_URL)
        log.info(f"Endee client connected → {ENDEE_URL}")

    if endee_index is None:
        # Create the index if it doesn't already exist
        existing = [idx["name"] for idx in endee_client.list_indexes()]
        if INDEX_NAME not in existing:
            endee_client.create_index(
                name=INDEX_NAME,
                dimension=EMBEDDING_DIM,
                space_type="cosine",
                precision=Precision.INT8,
            )
            log.info(f"Created Endee index '{INDEX_NAME}' (dim={EMBEDDING_DIM}, cosine)")
        else:
            log.info(f"Using existing Endee index '{INDEX_NAME}'")
        endee_index = endee_client.get_index(name=INDEX_NAME)

    return endee_index


# ─── Pydantic models ─────────────────────────────────────────────────────────
class AnalyzeRequest(BaseModel):
    resume_text: str
    job_description: str

class AnalyzeResponse(BaseModel):
    match_percentage: float
    missing_skills: List[str]
    relevant_keywords: List[str]
    summary: str
    resume_id: str

# ─── Skill keyword taxonomy ───────────────────────────────────────────────────
SKILL_TAXONOMY = [
    # Languages
    "python", "javascript", "typescript", "java", "golang", "go", "rust",
    "c++", "c#", "ruby", "php", "swift", "kotlin", "scala", "r",
    # Frontend
    "react", "angular", "vue", "next.js", "svelte", "html", "css", "tailwind",
    # Backend / Frameworks
    "node", "django", "fastapi", "flask", "spring", "express", "laravel",
    # Cloud
    "aws", "gcp", "azure", "cloud", "s3", "ec2", "lambda", "cloudflare",
    # DevOps / Infra
    "docker", "kubernetes", "terraform", "ansible", "ci/cd", "jenkins",
    "github actions", "linux", "bash", "nginx",
    # Data / ML
    "sql", "postgresql", "mysql", "mongodb", "redis", "elasticsearch", "kafka",
    "machine learning", "deep learning", "nlp", "computer vision",
    "pytorch", "tensorflow", "scikit-learn", "pandas", "numpy", "spark",
    "data pipeline", "etl", "data warehouse", "dbt",
    # Practices
    "rest api", "graphql", "grpc", "microservices", "agile", "scrum",
    "tdd", "git", "system design", "distributed systems",
    # Soft skills
    "communication", "leadership", "problem solving", "teamwork",
    "project management", "mentoring",
]


def extract_skills(text: str) -> set:
    """Return the subset of SKILL_TAXONOMY found in *text*."""
    lower = text.lower()
    found = set()
    for skill in SKILL_TAXONOMY:
        pattern = r"\b" + re.escape(skill) + r"\b"
        if re.search(pattern, lower):
            found.add(skill)
    return found


def cosine_sim(a: list, b: list) -> float:
    """Pure-NumPy cosine similarity."""
    av, bv = np.array(a, dtype=float), np.array(b, dtype=float)
    denom = np.linalg.norm(av) * np.linalg.norm(bv)
    return float(np.dot(av, bv) / denom) if denom else 0.0


# ─── Endpoints ───────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    try:
        get_endee()
        log.info("Endee index ready.")
    except Exception as exc:
        log.warning(f"Endee not reachable at startup — will retry on first request. ({exc})")


@app.get("/health", tags=["meta"])
def health():
    """Liveness probe."""
    return {
        "status": "ok",
        "embedding_model": EMBEDDING_MODEL,
        "vector_db": ENDEE_URL,
        "index": INDEX_NAME,
    }


@app.post("/analyze", response_model=AnalyzeResponse, tags=["core"])
def analyze(req: AnalyzeRequest):
    """
    RAG pipeline
    ─────────────
    1.  Embed resume  →  384-dim float32 vector
    2.  Upsert into Endee  (with skill filter metadata)
    3.  Embed job description
    4.  Query Endee  → top-5 nearest resume vectors
    5.  Blend Endee similarity with direct cosine score
    6.  Diff skill sets → matched / missing
    7.  Return structured JSON
    """
    if not req.resume_text.strip():
        raise HTTPException(400, "resume_text is empty.")
    if not req.job_description.strip():
        raise HTTPException(400, "job_description is empty.")

    # 1. Generate embeddings ──────────────────────────────────────────────────
    log.info("Generating embeddings …")
    resume_vec = embedder.encode(req.resume_text).tolist()
    jd_vec     = embedder.encode(req.job_description).tolist()

    # 2. Store resume vector in Endee ─────────────────────────────────────────
    index       = get_endee()
    resume_id   = str(uuid.uuid4())
    resume_skills = extract_skills(req.resume_text)

    index.upsert([
        {
            "id":     resume_id,
            "vector": resume_vec,
            "meta": {
                "type":    "resume",
                "preview": req.resume_text[:300],
                "skills":  ", ".join(sorted(resume_skills)),
            },
            # Endee filter field — enables future filtered queries
            "filter": {"doc_type": "resume"},
        }
    ])
    log.info(f"Stored resume vector in Endee  id={resume_id}")

    # 3. Query Endee with JD vector ───────────────────────────────────────────
    results = index.query(vector=jd_vec, top_k=5, ef=128)
    log.info(f"Endee returned {len(results)} results.")

    # 4. Compute similarity ───────────────────────────────────────────────────
    direct_sim   = cosine_sim(resume_vec, jd_vec)   # direct cosine (-1 … 1)

    # Endee returns similarity in [0, 1] for cosine space (normalised)
    endee_sim = results[0]["similarity"] if results else direct_sim

    # Blend: favour direct similarity (gives per-doc accuracy),
    # supplement with Endee corpus score
    blended = direct_sim * 0.65 + endee_sim * 0.35

    # Map cosine [-1, 1] → percentage [0, 100]
    match_pct = round(max(0.0, min(1.0, (blended + 1.0) / 2.0)) * 100, 1)

    # 5. Skill analysis ───────────────────────────────────────────────────────
    jd_skills = extract_skills(req.job_description)

    matched_skills  = sorted(resume_skills & jd_skills)
    missing_skills  = sorted(jd_skills - resume_skills)

    # 6. Summary ──────────────────────────────────────────────────────────────
    if match_pct >= 75:
        verdict = "Strong match 🎯 Your profile aligns very well with this role."
    elif match_pct >= 50:
        verdict = "Good match ✅ A few gaps to address before applying."
    else:
        verdict = "Needs work ⚠️  Significant skill gaps detected."

    summary = (
        f"{verdict}  "
        f"{len(matched_skills)} skills matched, {len(missing_skills)} skills missing."
    )

    return AnalyzeResponse(
        match_percentage=match_pct,
        missing_skills=missing_skills,
        relevant_keywords=matched_skills,
        summary=summary,
        resume_id=resume_id,
    )


@app.get("/history", tags=["meta"])
def history():
    """Return index metadata from Endee (total vectors stored, etc.)."""
    try:
        index = get_endee()
        info  = index.describe()
        return {"index": INDEX_NAME, "info": info}
    except Exception as exc:
        raise HTTPException(500, str(exc))
