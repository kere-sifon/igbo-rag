# CLAUDE.md

## Project

Igbo ↔ English RAG translator. Fully local: FAISS (1M pairs) + Qwen2.5:7B via Ollama + FastAPI. Live correction system lets a native speaker submit fixes that take effect immediately without a restart.

## Commands

```bash
# Start API (with hot reload)
python src/run.py

# Run evaluation suite
python src/eval.py

# Build FAISS index (one-time, ~5 hours)
python scripts/build_faiss_index.py /path/to/nllb_train.jsonl
```

API docs available at `http://localhost:8000/docs` once running.

## Architecture

```
query
  → correction lookup (_corrections dict — instant if match found)
  → nomic-embed-text embed (~83ms)
  → FAISS IndexFlatIP cosine search over 1M pairs (~3ms)
  → quality assessment (similarity > 0.92 = high, > 0.90 = medium, else low)
  → quality-aware prompt routing (grounded vs fallback)
  → Qwen2.5:7B via Ollama (temperature 0.1)
  → FastAPI response { translation, retrieval_quality, citations, latency_ms }

Persistence layer (MongoDB Atlas):
  feedback collection  — HITL corrections (query, direction, correct_translation,
                         wrong_translation, note, created_at, reingested)
  evals collection     — RAGAS run snapshots for score-drift tracking
```

## Key files

| File | Purpose |
|---|---|
| `src/rag_pipeline.py` | Core RAG logic — embed, retrieve, prompt, generate |
| `src/api.py` | FastAPI endpoints — translate, feedback, health |
| `src/db.py` | MongoDB client — corrections + eval persistence |
| `src/eval.py` | RAGAS evaluation suite |
| `src/run.py` | Uvicorn entry point with hot reload |
| `scripts/build_faiss_index.py` | One-time index build from nllb_train.jsonl |

## Critical invariants

**Corrections store** (`_corrections` in `rag_pipeline.py`):
- Key format: `"query_lowercase||direction"` (e.g. `"i miss you||en_to_igbo"`)
- **Source of truth is MongoDB** (`feedback` collection via `db.py`). The in-memory dict is a read-through cache rebuilt on startup and after every `POST /feedback`.
- `load_corrections()` does an atomic swap — builds a new dict from MongoDB, then replaces in one assignment. Never mutate `_corrections` in place.
- `_DEFAULT_CORRECTIONS` provides baseline corrections that `_corrections` overrides. Both are merged in `format_corrections_for_prompt()` before every LLM call.
- MongoDB connection failure at startup is non-fatal: the API stays up using defaults only, and a WARNING is printed. Fix the connection and restart to restore persistence.

**FAISS ↔ metadata sync** (`build_faiss_index.py`):
- `all_embeddings[i]` must always correspond to `embedded_pairs[i]`. The build script maintains these in lockstep — extend both on batch success, skip both on failure.
- Three checkpoint files kept in sync: `igbo_embeddings.npy`, `igbo_metadata_checkpoint.json`, `igbo_checkpoint_state.json`. All three must be deleted together or none.

**Distance metric**:
- FAISS returns cosine **similarity** (higher = more similar, range 0–1). Stored as `similarity` in citation dicts. Quality thresholds: `> 0.92` = high, `> 0.90` = medium. Do not invert to distance.

**Correction lookup**:
- Always checks corrections before the RAG pipeline. When `direction=None`, checks both `en_to_igbo` and `igbo_to_en` in order.

## Environment variables

See `.env.example` for the full list. Key ones:

```
OLLAMA_BASE_URL   # default: http://localhost:11434
LLM_MODEL         # default: Qwen2.5:7B
EMBED_MODEL       # default: nomic-embed-text
FAISS_INDEX_PATH  # required — path to igbo_faiss.index
FAISS_META_PATH   # required — path to igbo_metadata.json
CORRECTIONS_PATH  # optional — path for JSONL backup (no longer primary store)
MONGODB_URI       # default: mongodb://localhost:27017  (Atlas URI for prod)
MONGODB_DB        # default: igbo_rag
CORS_ORIGINS      # comma-separated; defaults to * for local dev
```

## Data files (not in git)

| File | Size | Notes |
|---|---|---|
| `data/igbo_faiss.index` | ~2.9 GB | FAISS IndexFlatIP, 1M vectors, 768-dim |
| `data/igbo_metadata.json` | ~193 MB | Parallel metadata for each vector |
| `data/corrections.jsonl` | small | Append-only, one JSON object per line |
| `nllb_train.jsonl` | external | Raw NLLB dataset — not in this repo |

## API endpoints

- `POST /translate` — translate a phrase; `direction` optional (auto-detects)
- `POST /feedback` — submit a correction; takes effect immediately
- `GET /feedback/list` — list all corrections
- `GET /health` — model, index size, correction count
- `GET /debug/faiss` — spot-check FAISS search latency
