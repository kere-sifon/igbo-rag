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
    CHROMA_DB_PATH,
    LLM_MODEL,
    get_chroma_collection,
    translate,
)

load_dotenv()

CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", CHROMA_DB_PATH)
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LLM_MODEL = os.getenv("LLM_MODEL", LLM_MODEL)

COLLECTION_NAME = "igbo_translations"
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
    chroma_db: str
    collection: str
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
    try:
        col = get_chroma_collection()
        loop = asyncio.get_event_loop()
        total_pairs = await loop.run_in_executor(None, col.count)
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail="ChromaDB is unreachable",
        ) from exc

    return HealthResponse(
        status="ok",
        model=LLM_MODEL,
        chroma_db=CHROMA_DB_PATH,
        collection=COLLECTION_NAME,
        total_pairs=total_pairs,
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


@app.get("/debug/chroma")
async def debug_chroma():
    import time
    start = time.time()
    col = get_chroma_collection()
    chroma_ms = (time.time() - start) * 1000
    return {
        "chroma_ms": chroma_ms,
        "count": col.count()
    }


@app.get("/debug/llm")
async def debug_llm():
    import time
    from langchain_ollama import ChatOllama
    start = time.time()
    llm = ChatOllama(model=LLM_MODEL, base_url=OLLAMA_BASE_URL, temperature=0.1)
    response = llm.invoke("say hi")
    llm_ms = (time.time() - start) * 1000
    return {
        "llm_ms": llm_ms,
        "model": LLM_MODEL,
        "response": response.content
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.api:app", host="0.0.0.0", port=8000)
