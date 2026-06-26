import asyncio
import json
import os
import time
from datetime import datetime
from functools import partial
from typing import Literal, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from rag_pipeline import (
    LLM_MODEL,
    FAISS_INDEX_PATH,
    translate,
    load_corrections,
)

load_dotenv()

LLM_MODEL = os.getenv("LLM_MODEL", LLM_MODEL)
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", FAISS_INDEX_PATH)
FAISS_META_PATH = os.getenv("FAISS_META_PATH",
    os.path.expanduser("~/projects/igbo-rag/data/igbo_metadata.json"))
CORRECTIONS_PATH = os.getenv("CORRECTIONS_PATH",
    os.path.expanduser("~/projects/igbo-rag/data/corrections.jsonl"))

APP_NAME = "Igbo-English RAG Translator"
APP_VERSION = "1.0.0"

_cors_origins_env = os.getenv("CORS_ORIGINS", "")
CORS_ORIGINS = [o.strip() for o in _cors_origins_env.split(",") if o.strip()] or ["*"]

app = FastAPI(title=APP_NAME, version=APP_VERSION)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Pydantic models ---

class RootResponse(BaseModel):
    name: str
    version: str
    docs: str


class HealthResponse(BaseModel):
    status: str
    model: str
    faiss_index: str
    total_pairs: int
    total_corrections: int


class TranslateRequest(BaseModel):
    query: str
    direction: Literal["en_to_igbo", "igbo_to_en"] | None = None


class Citation(BaseModel):
    input: str
    output: str
    direction: str
    similarity: float


class TranslateResponse(BaseModel):
    query: str
    direction: str
    translation: str
    retrieval_quality: str
    citations: list[Citation]
    latency_ms: float


class FeedbackRequest(BaseModel):
    query: str
    direction: Literal["en_to_igbo", "igbo_to_en"]
    correct_translation: str
    wrong_translation: Optional[str] = None
    note: Optional[str] = None


class FeedbackResponse(BaseModel):
    status: str
    message: str
    total_corrections: int


# --- Endpoints ---

@app.get("/", response_model=RootResponse)
def read_root() -> RootResponse:
    return RootResponse(
        name=APP_NAME,
        version=APP_VERSION,
        docs="/docs",
    )


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    from rag_pipeline import _faiss_index, _corrections
    return HealthResponse(
        status="ok",
        model=LLM_MODEL,
        faiss_index=FAISS_INDEX_PATH,
        total_pairs=_faiss_index.ntotal,
        total_corrections=len(_corrections),
    )


@app.post("/translate", response_model=TranslateResponse)
async def translate_text(request: TranslateRequest) -> TranslateResponse:
    if not request.query.strip():
        raise HTTPException(status_code=422, detail="query must not be empty")

    start = time.perf_counter()
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(
        None, partial(translate, request.query, direction=request.direction)
    )
    latency_ms = (time.perf_counter() - start) * 1000

    direction = result["direction"] or "auto"

    return TranslateResponse(
        query=result["query"],
        direction=direction,
        translation=result["response"],
        retrieval_quality=result["retrieval_quality"],
        citations=[Citation(**c) for c in result["citations"]],
        latency_ms=round(latency_ms, 2),
    )


@app.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(request: FeedbackRequest) -> FeedbackResponse:
    """
    Submit a translation correction.
    Saves to corrections.jsonl and immediately updates the in-memory
    reference phrase list so the fix takes effect without a restart.
    """
    if not request.query.strip():
        raise HTTPException(status_code=422, detail="query must not be empty")
    if not request.correct_translation.strip():
        raise HTTPException(status_code=422, detail="correct_translation must not be empty")

    correction = {
        "query": request.query.strip(),
        "direction": request.direction,
        "correct_translation": request.correct_translation.strip(),
        "wrong_translation": request.wrong_translation,
        "note": request.note,
        "timestamp": datetime.utcnow().isoformat(),
    }

    # Save to corrections.jsonl
    os.makedirs(os.path.dirname(CORRECTIONS_PATH), exist_ok=True)
    with open(CORRECTIONS_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(correction, ensure_ascii=False) + "\n")

    # Reload corrections into memory immediately — no restart needed
    load_corrections(CORRECTIONS_PATH)

    from rag_pipeline import _corrections
    total = len(_corrections)

    return FeedbackResponse(
        status="ok",
        message=f"Correction saved: '{request.query}' = '{request.correct_translation}'",
        total_corrections=total,
    )


@app.get("/feedback/list")
async def list_feedback():
    """List all saved corrections."""
    if not os.path.exists(CORRECTIONS_PATH):
        return {"corrections": [], "total": 0}

    corrections = []
    with open(CORRECTIONS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    corrections.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    return {"corrections": corrections, "total": len(corrections)}


@app.get("/debug/faiss")
async def debug_faiss():
    from rag_pipeline import _faiss_index
    import numpy as np
    import faiss
    start = time.time()
    vec = np.random.rand(1, 768).astype(np.float32)
    faiss.normalize_L2(vec)
    distances, indices = _faiss_index.search(vec, 1)
    search_ms = (time.time() - start) * 1000
    return {
        "search_ms": search_ms,
        "total_vectors": _faiss_index.ntotal,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000)