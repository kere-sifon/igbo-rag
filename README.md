# Igbo-English RAG Translator

![Python](https://img.shields.io/badge/Python-3.14-blue)
![FAISS](https://img.shields.io/badge/FAISS-1M_curated_pairs-green)
![Ollama](https://img.shields.io/badge/LLM-Qwen2.5%3A7B-orange)
![RAGAS](https://img.shields.io/badge/Answer_Relevancy-0.594-brightgreen)
![Local](https://img.shields.io/badge/Inference-100%25_Local-purple)
![MongoDB](https://img.shields.io/badge/Corrections-MongoDB_Atlas-darkgreen)
![Flowise](https://img.shields.io/badge/Agent-Flowise_Agentflow-blue)

A production RAG translation system for Igbo ↔ English, running entirely on local infrastructure — Qwen2.5:7B via Ollama, a FAISS vector index of 1M curated translation pairs, a FastAPI REST API, a MongoDB-backed correction system with nightly re-ingestion into FAISS, a Flowise Agentflow for deterministic routing, and an Open WebUI chat frontend. Zero API costs, zero data leaving the machine.

## Why I built this

Igbo is a low-resource African language. Despite being spoken by tens of millions of people, it is badly underserved by mainstream translation systems — training data is scarce, noisy, and rarely curated, and commercial models treat it as an afterthought.

This project is personal. I was raised in Igbo land and grew up speaking the language fluently. Igbo is part of who I am, and I want to keep that connection alive for my children — with a tool that produces *formal, correct* Igbo rather than transliterated approximations or social media noise.

Growing up speaking the language also means I can directly evaluate whether the system's output is correct — an advantage most NLP researchers working on low-resource African languages don't have. When the model returns "M m i love ya" instead of "A hụrụ m gị n'anya", I know immediately it's wrong, and I can trace exactly where the pipeline failed.

## Demo

![Correction loop in action](docs/demo.png)

The correction loop working end to end in Open WebUI: `"Where do you go to school?"` initially translates to a LOW-quality `"Ebe one ka ị na-eje Akwụkwọ?"`. Typing `"It should be 'Ebe ole ka i na-eje Akwukwo'"` triggers the Flowise correction path, which saves the fix to MongoDB and confirms it immediately — no restart, no rebuild. A few turns earlier, `"Where is the elephant?"` shows what a phrase looks like *after* it's already been corrected once: retrieval quality reads **CORRECTION** (✅) instead of LOW/MEDIUM/HIGH, meaning it's served straight from the verified corrections dict rather than a FAISS similarity match.

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

### 7. Embedding mismatch bug — documents and queries in different vector spaces

The index was originally built by embedding the bilingual concatenation `"{input} | {output}"` per pair, while queries at inference time embedded `"{input}"` alone. This put stored vectors and query vectors in different regions of embedding space — real semantic matches didn't reliably rank, and for short common phrases the mismatch could produce a spuriously high similarity score instead of an honest low one.

The fix: embed documents on the `input` field only — the field actually being searched by — with matching nomic task prefixes (`search_document:` on stored vectors, `search_query:` at query time). This requires a full index rebuild, since all stored vectors were in the wrong format; a rebuilt input-only index queried without the matching prefix (or vice versa) reintroduces a mismatch, just a different one.

A direct check of both the curated metadata and the raw NLLB source confirmed the corpus never contained a clean, standalone entry for common canonical phrases like "good morning" or "thank you" — every occurrence was the phrase embedded inside a longer, unrelated sentence. Pre-fix HIGH-confidence retrieval for these phrases wasn't a real match; it was the embedding bug producing a coincidentally high score. Post-fix, the system correctly reports low retrieval confidence for these phrases and falls back to the model's own knowledge — see [RAGAS evaluation results](#ragas-evaluation-results) for the full before/after comparison.

## RAGAS evaluation results

Evaluated with RAGAS (faithfulness, answer relevancy, context precision), judged by DeepSeek-r1:14b with `nomic-embed-text` embeddings.

| Query | Direction | Retrieval | Faithfulness | Answer Relevancy | Context Precision |
|-------|-----------|-----------|--------------|------------------|-------------------|
| Ụtụtụ ọma | igbo→en | LOW | 0.00 | 0.55 | 0.00 |
| A hụrụ m gị n'anya | igbo→en | LOW | 0.25 | 0.51 | 0.42 |
| Kedu ka i mere? | igbo→en | LOW | 0.33 | 0.63 | 0.00 |
| Biko nọdụ ala | igbo→en | LOW | 0.00 | 0.55 | 0.33 |
| I love you | en→igbo | LOW | 0.00 | 0.59 | 0.64 |
| Good morning | en→igbo | LOW | 0.67 | 0.70 | 1.00 |
| Thank you | en→igbo | LOW | 0.00 | 0.67 | 0.00 |
| Please sit down | en→igbo | LOW | 0.00 | 0.57 | 0.53 |
| **AVERAGE** | | | **0.156** | **0.594** | **0.365** |

All 8 translations were correct, including exact ground-truth matches (`Thank you` → `Daalụ`, `Good morning` → `Ụtụtụ ọma`). Every query routed **LOW**, which at first looks like a regression from an earlier run where these same phrases scored HIGH/MEDIUM retrieval (faithfulness 0.423, relevancy 0.833, context precision 0.469 average). It isn't — it's the result of a bug fix, and it's worth explaining why the "worse" numbers are actually more honest.

**What changed:** the FAISS index was originally built by embedding the bilingual concatenation `"{input} | {output}"` per pair, while queries were embedded as `"{input}"` alone — putting documents and queries in different regions of embedding space (see `src/rag_pipeline.py` / `scripts/build_faiss_index.py` history). For short, common phrases, that mismatch didn't always show up as a bad match — the appended `" | {output}"` contributes little to a short string's embedding, so similarity scores could still spuriously clear the HIGH threshold. The fix re-embeds documents on the `input` field only, with matching `search_document:`/`search_query:` nomic task prefixes on both sides.

**Verifying the corpus, not just the fix:** a direct check of both the curated metadata (1,000,005 pairs) and the raw 19.5M-row NLLB source for exact matches on all 5 canonical phrases in the eval set returned **zero exact matches in either** — every occurrence is the phrase embedded inside a longer, unrelated sentence, frequently with a translation that doesn't map cleanly back to the phrase itself. The corpus was never a phrasebook; it's natural-sentence data with real NLLB translation noise, and it never had a clean standalone entry for "good morning" or "thank you." The pre-fix HIGH-confidence retrieval for these exact phrases wasn't a real match — it was the embedding-mismatch bug producing a coincidentally high similarity score for short strings. Post-fix, the system correctly reports LOW confidence and falls back to the model's own knowledge, which is why every translation stayed correct despite every retrieval being LOW.

Low faithfulness on LOW-retrieval rows is expected and by design — RAGAS scores faithfulness as grounding in the retrieved context, and the fallback prompt deliberately overrides noisy or absent corpus matches rather than grounding on them. A correct translation that ignores a bad context is scored "unfaithful" by RAGAS; that's the intended trade-off documented in [Quality-aware prompt routing](#4-quality-aware-prompt-routing), not a defect. Answer relevancy (0.594) is the more informative signal here for output quality — and even that undersells it, since manual verification confirmed all 8 translations were correct, including exact matches.

**What this means for corpus curation:** the corrections layer exists precisely for this gap. When a native speaker submits the correct canonical translation via `/feedback`, it becomes a clean, standalone `{input, output}` pair in MongoDB and, after the nightly re-ingest, a first-class entry in the FAISS index — something the upstream NLLB corpus never provided for exactly the short, common phrases people are most likely to ask about.

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


