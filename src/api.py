import asyncio
import os
import time
from functools import partial
from typing import Literal

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from rag_pipeline import (
    LLM_MODEL,
    FAISS_INDEX_PATH,
    translate,
)

load_dotenv()

LLM_MODEL = os.getenv("LLM_MODEL", LLM_MODEL)
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", FAISS_INDEX_PATH)
FAISS_META_PATH = os.getenv("FAISS_META_PATH",
    os.path.expanduser("~/projects/igbo-rag/data/igbo_metadata.json"))

APP_NAME = "Igbo-English RAG Translator"
APP_VERSION = "1.0.0"

app = FastAPI(title=APP_NAME, version=APP_VERSION)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class RootResponse(BaseModel):
    name: str
    version: str
    docs: str


class HealthResponse(BaseModel):
    status: str
    model: str
    faiss_index: str
    total_pairs: int


class TranslateRequest(BaseModel):
    query: str
    direction: Literal["en_to_igbo", "igbo_to_en"] | None = None


class Citation(BaseModel):
    input: str
    output: str
    direction: str
    distance: float


class TranslateResponse(BaseModel):
    query: str
    direction: str
    translation: str
    retrieval_quality: str
    citations: list[Citation]
    latency_ms: float


@app.get("/", response_model=RootResponse)
def read_root() -> RootResponse:
    return RootResponse(
        name=APP_NAME,
        version=APP_VERSION,
        docs="/docs",
    )


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    from rag_pipeline import _faiss_index
    return HealthResponse(
        status="ok",
        model=LLM_MODEL,
        faiss_index=FAISS_INDEX_PATH,
        total_pairs=_faiss_index.ntotal,
    )


@app.post("/translate", response_model=TranslateResponse)
async def translate_text(request: TranslateRequest) -> TranslateResponse:
    if not request.query.strip():
        raise HTTPException(status_code=422, detail="query must not be empty")

    start = time.perf_counter()
    loop = asyncio.get_event_loop()
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


@app.get("/debug/faiss")
async def debug_faiss():
    from rag_pipeline import _faiss_index
    start = time.time()
    import numpy as np
    import faiss
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