import asyncio
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from functools import partial
from typing import Any, Literal, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from rag_pipeline import (
    LLM_MODEL,
    FAISS_INDEX_PATH,
    translate,
    load_corrections,
)
from ops import router as ops_router


load_dotenv()

LLM_MODEL = os.getenv("LLM_MODEL", LLM_MODEL)
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", FAISS_INDEX_PATH)
FLOWISE_URL = os.getenv("FLOWISE_URL", "http://localhost:3004")
FLOWISE_AGENTFLOW_ID = os.getenv("FLOWISE_AGENTFLOW_ID", "")
FLOWISE_API_KEY = os.getenv("FLOWISE_API_KEY", "")

APP_NAME = "Igbo-English RAG Translator"
APP_VERSION = "1.3.0"

_cors_origins_env = os.getenv("CORS_ORIGINS", "")
CORS_ORIGINS = [o.strip() for o in _cors_origins_env.split(",") if o.strip()] or ["*"]

app = FastAPI(title=APP_NAME, version=APP_VERSION)
app.include_router(ops_router)
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

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


class FeedbackListResponse(BaseModel):
    corrections: list[dict]
    total: int
    limit: int
    skip: int


class SessionUpdate(BaseModel):
    session_id: str
    last_query: str
    last_direction: str
    last_translation: str


# ---------------------------------------------------------------------------
# Session store — keyed by session_id (in-memory, survives within process)
# ---------------------------------------------------------------------------
_sessions: dict = {}


# ---------------------------------------------------------------------------
# Core endpoints
# ---------------------------------------------------------------------------

@app.get("/", response_model=RootResponse)
def read_root() -> RootResponse:
    return RootResponse(name=APP_NAME, version=APP_VERSION, docs="/docs")


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    from rag_pipeline import _faiss_index, _corrections
    from db import count_corrections
    return HealthResponse(
        status="ok",
        model=LLM_MODEL,
        faiss_index=FAISS_INDEX_PATH,
        total_pairs=_faiss_index.ntotal,
        total_corrections=count_corrections(),
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
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    try:
        from db import save_correction, count_corrections
        save_correction(correction)
        total = count_corrections()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Could not save correction to MongoDB: {exc}")
    load_corrections()
    return FeedbackResponse(
        status="ok",
        message=f"Correction saved: '{request.query}' = '{request.correct_translation}'",
        total_corrections=total,
    )


@app.get("/feedback/list", response_model=FeedbackListResponse)
async def list_feedback(
    limit: int = Query(default=50, ge=1, le=500),
    skip: int = Query(default=0, ge=0),
):
    try:
        from db import list_corrections, count_corrections
        corrections = list_corrections(limit=limit, skip=skip)
        total = count_corrections()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Could not read corrections from MongoDB: {exc}")
    for c in corrections:
        for k, v in c.items():
            if hasattr(v, "isoformat"):
                c[k] = v.isoformat()
    return FeedbackListResponse(corrections=corrections, total=total, limit=limit, skip=skip)


@app.post("/feedback/reload")
async def reload_feedback():
    try:
        load_corrections()
        from rag_pipeline import _corrections
        return {"status": "ok", "total_corrections": len(_corrections)}
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Could not reload corrections: {exc}")


# ---------------------------------------------------------------------------
# Session store endpoints
# ---------------------------------------------------------------------------

@app.post("/session/update")
async def update_session(req: SessionUpdate):
    _sessions[req.session_id] = {
        "last_query": req.last_query,
        "last_direction": req.last_direction,
        "last_translation": req.last_translation,
    }
    print(f"SESSION STORED: {req.session_id} → {req.last_query}")
    return {"status": "ok"}


@app.get("/session/{session_id}")
async def get_session(session_id: str):
    session = _sessions.get(session_id, {
        "last_query": "",
        "last_direction": "en_to_igbo",
        "last_translation": ""
    })
    print(f"SESSION GET: {session_id} → {session}")
    return session


# ---------------------------------------------------------------------------
# OpenAI-compatible proxy — Open WebUI → igbo-rag API → Flowise agentflow
#
# Key design: derive a stable session_id from the Open WebUI conversation
# by hashing all but the last message. This is consistent across turns in
# the same conversation and is passed to Flowise as overrideConfig.sessionId
# so $flow.sessionId in the Custom Function matches what we store.
# ---------------------------------------------------------------------------

def _derive_session_id(messages: list[dict]) -> str:
    # Use the first user message as a stable conversation fingerprint
    first_user = next(
        (m.get("content", "") for m in messages if m.get("role") == "user"),
        "default"
    )
    return hashlib.md5(first_user.encode()).hexdigest()


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": "igbo-rag",
                "object": "model",
                "created": 1700000000,
                "owned_by": "igbo-rag",
                "display_name": "Igbo RAG Translator",
            }
        ]
    }


@app.post("/v1/chat/completions")
async def chat_completions(raw_request: Request):
    import httpx
    request = await raw_request.json()

    messages = request.get("messages", [])
    if not messages:
        raise HTTPException(status_code=422, detail="messages must not be empty")

    last_user = next(
        (m["content"] for m in reversed(messages) if m.get("role") == "user"), ""
    )
    if not last_user:
        raise HTTPException(status_code=422, detail="No user message found")

    # Derive stable session ID from conversation history
    session_id = _derive_session_id(messages)
    print(f"PROXY session_id: {session_id} for message: {last_user[:50]}")

    history = [
        {"role": m["role"], "content": m["content"]}
        for m in messages[:-1]
        if m.get("role") in ("user", "assistant")
    ]

    flowise_url = f"{FLOWISE_URL}/api/v1/prediction/{FLOWISE_AGENTFLOW_ID}"
    headers = {}
    if FLOWISE_API_KEY:
        headers["Authorization"] = f"Bearer {FLOWISE_API_KEY}"

    try:
        async with httpx.AsyncClient(timeout=90) as client:
            resp = await client.post(
                flowise_url,
                json={
                    "question": last_user,
                    "chatHistory": history,
                    "sessionId": session_id,
                    "overrideConfig": {
                        "sessionId": session_id,
                    }
                },
                headers=headers,
            )
            resp.raise_for_status()
            data = resp.json()
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Flowise error: {exc}")

    answer = data.get("text") or data.get("answer") or data.get("output") or str(data)

    return {
        "id": f"igbo-rag-{int(time.time())}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": "igbo-rag",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": answer},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


# ---------------------------------------------------------------------------
# Debug endpoints
# ---------------------------------------------------------------------------

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
    return {"search_ms": search_ms, "total_vectors": _faiss_index.ntotal}


@app.get("/debug/corrections")
async def debug_corrections():
    from rag_pipeline import _corrections
    return {"corrections": _corrections, "total": len(_corrections)}


@app.get("/debug/sessions")
async def debug_sessions():
    return {"sessions": _sessions, "total": len(_sessions)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000)