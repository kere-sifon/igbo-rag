# Igbo-English RAG Translator

![Python](https://img.shields.io/badge/Python-3.14-blue)
![FAISS](https://img.shields.io/badge/FAISS-1M_curated_pairs-green)
![Ollama](https://img.shields.io/badge/LLM-Qwen2.5%3A7B-orange)
![RAGAS](https://img.shields.io/badge/Answer_Relevancy-0.833-brightgreen)
![Local](https://img.shields.io/badge/Inference-100%25_Local-purple)
![MongoDB](https://img.shields.io/badge/Corrections-MongoDB_Atlas-darkgreen)
![Flowise](https://img.shields.io/badge/Agent-Flowise_Agentflow-blue)

A production RAG translation system for Igbo ↔ English, running entirely on local infrastructure — Qwen2.5:7B via Ollama, a FAISS vector index of 1M curated translation pairs, a FastAPI REST API, a MongoDB-backed correction system with nightly re-ingestion into FAISS, a Flowise Agentflow for deterministic routing, and an Open WebUI chat frontend. Zero API costs, zero data leaving the machine.

## Why I built this

Igbo is a low-resource African language. Despite being spoken by tens of millions of people, it is badly underserved by mainstream translation systems — training data is scarce, noisy, and rarely curated, and commercial models treat it as an afterthought.

This project is personal. I was raised in Igbo land and grew up speaking the language fluently. Igbo is part of who I am, and I want to keep that connection alive for my children — with a tool that produces *formal, correct* Igbo rather than transliterated approximations or social media noise.

Growing up speaking the language also means I can directly evaluate whether the system's output is correct — an advantage most NLP researchers working on low-resource African languages don't have. When the model returns "M m i love ya" instead of "A hụrụ m gị n'anya", I know immediately it's wrong, and I can trace exactly where the pipeline failed.

## Architecture

```
Open WebUI (chat UI)
  → igbo-rag FastAPI /v1/chat/completions (OpenAI-compatible proxy)
    → Flowise Agentflow (deterministic routing)
      → Custom Function: isCorrection?
        YES → GET /session/{id} → POST /feedback → POST /feedback/reload
        NO  → POST /translate → POST /session/update
      → Direct Reply
    → OpenAI format response

igbo-rag FastAPI /translate:
  query
    → correction lookup (_corrections dict — instant, from MongoDB)
    → nomic-embed-text embed (~83ms)
    → FAISS IndexFlatIP cosine search over 1M pairs (~3ms)
    → quality assessment (similarity > 0.92 = HIGH, > 0.90 = MEDIUM, else LOW)
    → quality-aware prompt routing (grounded vs fallback)
    → Qwen2.5:7B via Ollama (temperature 0.1)
    → FastAPI response { translation, retrieval_quality, citations, latency_ms }

Nightly cron (2am):
  MongoDB { reingested: false }
    → embed via nomic-embed-text
    → upsert into FAISS index + metadata
    → mark reingested in MongoDB
    → POST /feedback/reload
```

## Key components

| Component | Technology |
|---|---|
| Vector index | FAISS IndexFlatIP (1M+ curated pairs, 768-dim) |
| Source corpus | NLLB dataset (19.5M pairs, `nllb_train.jsonl`) |
| Embeddings | nomic-embed-text (768-dim, via Ollama) |
| LLM | Qwen2.5:7B (via Ollama) |
| API | FastAPI |
| Correction store | MongoDB Atlas (persisted) + in-memory dict (hot path) |
| Agent routing | Flowise Agentflow v2 |
| Chat UI | Open WebUI (via OpenAI-compatible proxy) |
| Re-ingestion | Nightly cron — corrections fold back into FAISS corpus |
| Evaluation | RAGAS |
| Runtime | Python 3.14, fully local |

## Key files

| File | Purpose |
|---|---|
| `src/rag_pipeline.py` | Core RAG logic — embed, retrieve, prompt, generate |
| `src/api.py` | FastAPI — translate, feedback, session store, OpenAI proxy |
| `src/db.py` | MongoDB client — corrections + eval persistence |
| `src/eval.py` | RAGAS evaluation suite |
| `src/run.py` | Uvicorn entry point |
| `scripts/build_faiss_index.py` | One-time index build from nllb_train.jsonl |
| `scripts/reingest_corrections.py` | Nightly re-ingestion of MongoDB corrections into FAISS |

## Key engineering decisions

### 1. FAISS over ChromaDB — a performance migration

The system was originally built on ChromaDB with a 9.7M-pair SQLite store. Under load, ChromaDB retrieval took **60–90 seconds per query** due to SQLite scan overhead at that scale.

The fix was to migrate to FAISS with a curated index extracted from the original 19.5M-pair NLLB dataset:
- Extract and quality-filter 1M pairs from `nllb_train.jsonl`
- Embed with `nomic-embed-text` (~5 hours one-time cost)
- Build a normalised `IndexFlatIP` FAISS index (cosine similarity)
- Result: **~86ms total retrieval** (83ms embed + 3ms search) vs 60–90s with ChromaDB

### 2. MongoDB-backed corrections with nightly FAISS re-ingestion

Corrections are a two-layer system:

**Layer 1 — hot path (instant):** Every correction is saved to MongoDB and immediately loaded into an in-memory `_corrections` dict. The next translation of the same phrase bypasses FAISS entirely and returns the verified correction in ~0.36ms.

**Layer 2 — cold path (nightly):** At 2am, `scripts/reingest_corrections.py` reads all `{ reingested: false }` corrections from MongoDB, embeds them with the same `nomic-embed-text` model, and upserts them directly into the live FAISS index and metadata file. After re-ingestion, the correction exists as a proper corpus pair — so even if the MongoDB record is deleted, the corpus itself has been improved. Corrections compound over time.

This means every correction made by a native Igbo speaker is a permanent improvement to the system.

### 3. Flowise Agentflow for deterministic routing

The original Open WebUI implementation used tool calling (Qwen2.5:7B deciding when to call `translate_igbo` or `correct_translation`). This was unreliable — the model would occasionally hallucinate tool results, ignore the system prompt, or fail to call the tool at all.

The fix was to move routing entirely out of the LLM and into a **Flowise Agentflow** — a visual agent flow where a Custom Function node detects correction phrases deterministically, calls `/translate` or `/feedback` directly via HTTP, and returns the result without any model inference in the routing path. The LLM is only used for translation generation, not for deciding what to do.

The igbo-rag FastAPI exposes an **OpenAI-compatible proxy** (`/v1/models`, `/v1/chat/completions`) that Open WebUI connects to as a custom model. The proxy derives a stable session ID from the conversation's first user message, passes it to Flowise via `overrideConfig.sessionId`, and the session store tracks `lastQuery`/`lastDirection`/`lastTranslation` so corrections always reference the right phrase.

### 4. Quality-aware prompt routing

The naive RAG approach feeds whatever the retriever returns straight into the prompt. That is the **wrong objective when corpus quality is variable.** The NLLB dataset contains noise — transliterated English, internet slang, and mislabeled pairs. Blindly grounding on a weak match produces a confidently wrong translation.

Routing on retrieval similarity lets the system decide *whether the corpus deserves trust* for a given query. When the match is strong (similarity > 0.92), it grounds tightly on the corpus. When the match is weak, it deliberately overrides the corpus and falls back to the model's own knowledge of formal Igbo, backed by a curated reference phrase list validated by a native speaker.

### 5. Native speaker validation

Most NLP projects on low-resource languages are built without a native speaker in the loop. Having grown up speaking Igbo fluently, I can directly evaluate output correctness and trace pipeline failures. The fallback prompt's reference phrase list was hand-curated and verified, not generated.

### 6. Corpus curation strategy

The raw NLLB dataset has 19.5M pairs but significant noise. The quality filter:
- Keeps only `Translate` instruction pairs
- Filters pairs with outputs under 4 characters
- Removes social media noise (`mmmmm`, `ya nice`, `lol` etc.)
- For `en_to_igbo` only: drops pairs where the Igbo output contains no unicode characters (genuine Igbo always uses diacritics)
- Result: ~87% of sampled pairs pass the filter

## RAGAS evaluation results

Evaluated with RAGAS (faithfulness, answer relevancy, context precision), judged by DeepSeek-r1:14b with `nomic-embed-text` embeddings.

| Query | Direction | Retrieval | Faithfulness | Answer Relevancy | Context Precision |
|-------|-----------|-----------|--------------|------------------|-------------------|
| Ụtụtụ ọma | igbo→en | HIGH | 0.833 | 0.875 | 1.000 |
| A hụrụ m gị n'anya | igbo→en | HIGH | 0.250 | 0.733 | 1.000 |
| Kedu ka i mere? | igbo→en | HIGH | 0.500 | 0.745 | 0.000 |
| Biko nọdụ ala | igbo→en | HIGH | 0.750 | 0.851 | 0.950 |
| I love you | en→igbo | LOW | 0.250 | 0.809 | 0.000 |
| Good morning | en→igbo | MEDIUM | 0.000 | 0.919 | 0.000 |
| Thank you | en→igbo | LOW | 0.000 | 0.908 | 0.000 |
| Please sit down | en→igbo | MEDIUM | 0.800 | 0.824 | 0.804 |
| **AVERAGE** | | | **0.423** | **0.833** | **0.469** |

Low faithfulness on `LOW`/`MEDIUM` retrieval rows is expected — the fallback prompt deliberately instructs the model to override noisy corpus matches. RAGAS penalises this as "unfaithful to context" but answer relevancy stays high (0.833 average), which is the metric that reflects actual translation quality.

## Setup and run

### Prerequisites

[Ollama](https://ollama.com/) running locally:

```bash
ollama pull qwen2.5:7b
ollama pull nomic-embed-text
```

[MongoDB Atlas](https://www.mongodb.com/atlas) free tier or local MongoDB instance.

[Flowise](https://flowiseai.com/) running locally on port 3004.

### Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Configure

Copy `.env.example` to `.env` and fill in:

```env
# Ollama
OLLAMA_BASE_URL=http://localhost:11434
LLM_MODEL=Qwen2.5:7B
EMBED_MODEL=nomic-embed-text

# Data paths
FAISS_INDEX_PATH=/path/to/data/igbo_faiss.index
FAISS_META_PATH=/path/to/data/igbo_metadata.json

# MongoDB
MONGODB_URI=mongodb+srv://<user>:<pass>@<cluster>.mongodb.net/?retryWrites=true&w=majority
MONGODB_DB=igbo_rag

# Flowise (for Open WebUI proxy)
FLOWISE_URL=http://localhost:3004
FLOWISE_AGENTFLOW_ID=<your-agentflow-uuid>
FLOWISE_API_KEY=<your-flowise-api-key>

# API
CORS_ORIGINS=http://localhost:3000,http://localhost:8080
```

### Build the FAISS index (one-time, ~5 hours for 1M pairs)

```bash
caffeinate -i python scripts/build_faiss_index.py /path/to/nllb_train.jsonl
```

Outputs `data/igbo_faiss.index` and `data/igbo_metadata.json`. Checkpoints every 5000 pairs — if interrupted, re-run and it resumes automatically.

### Run the API

```bash
python src/run.py
```

Server starts on `http://0.0.0.0:8000`. Swagger docs at `http://localhost:8000/docs`.

### Run nightly re-ingestion manually

```bash
python scripts/reingest_corrections.py
```

### Set up nightly cron (2am)

```bash
crontab -e
```

Add:
```
0 2 * * * cd /Users/kere/igbo-rag && /Users/kere/igbo-rag/.venv/bin/python scripts/reingest_corrections.py >> logs/reingest.log 2>&1
```

### Open WebUI integration

1. Start the API: `python src/run.py`
2. In Open WebUI → **Settings → Connections** → add OpenAI-compatible:
   - URL: `http://localhost:8000/v1`
   - API Key: `igbo-rag`
3. Select **Igbo RAG Translator** from the model dropdown

All routing is handled deterministically by the Flowise Agentflow — no system prompt or tool configuration needed in Open WebUI.

### Flowise Agentflow setup

1. In Flowise → **Agentflows** → create a new agentflow named `RAG`
2. Add nodes: `Start → Custom Function → Direct Reply`
3. Paste the Custom Function code from `src/flowise_custom_function.js`
4. Set `HTTP_SECURITY_CHECK=false` in Flowise environment to allow localhost calls

## API reference

### `POST /translate`

```bash
curl -s -X POST http://localhost:8000/translate \
  -H "Content-Type: application/json" \
  -d '{"query": "good evening", "direction": "en_to_igbo"}'
```

```json
{
  "query": "good evening",
  "direction": "en_to_igbo",
  "translation": "Anyasị ọma",
  "retrieval_quality": "correction",
  "citations": [],
  "latency_ms": 0.36
}
```

### `POST /feedback`

```bash
curl -s -X POST http://localhost:8000/feedback \
  -H "Content-Type: application/json" \
  -d '{
    "query": "good evening",
    "direction": "en_to_igbo",
    "correct_translation": "Anyasị ọma",
    "wrong_translation": "Ehihie ọma"
  }'
```

### `POST /feedback/reload`

Force reload corrections from MongoDB into memory (no restart needed):

```bash
curl -X POST http://localhost:8000/feedback/reload
```

### `GET /feedback/list`

```bash
curl "http://localhost:8000/feedback/list?limit=20"
```

### `GET /health`

```bash
curl http://localhost:8000/health
```

```json
{
  "status": "ok",
  "model": "Qwen2.5:7B",
  "faiss_index": "/path/to/igbo_faiss.index",
  "total_pairs": 1000005,
  "total_corrections": 5
}
```

### `GET /debug/corrections`

Show live in-memory corrections dict:

```bash
curl http://localhost:8000/debug/corrections
```

### `GET /debug/sessions`

Show active session store:

```bash
curl http://localhost:8000/debug/sessions
```


