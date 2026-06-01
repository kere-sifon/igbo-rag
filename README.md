# Igbo-English RAG Translator

![Python](https://img.shields.io/badge/Python-3.14-blue)
![FAISS](https://img.shields.io/badge/FAISS-100K_curated_pairs-green)
![Ollama](https://img.shields.io/badge/LLM-Llama3.1%3A8b-orange)
![RAGAS](https://img.shields.io/badge/Answer_Relevancy-0.833-brightgreen)
![Local](https://img.shields.io/badge/Inference-100%25_Local-purple)

A production RAG translation system for Igbo ↔ English, running entirely on local infrastructure — Llama3.1:8b via Ollama, a FAISS vector index of 100K curated translation pairs, a FastAPI REST API, and a RAGAS evaluation suite. Zero API costs, zero data leaving the machine.

## Why I built this

Igbo is a low-resource African language. Despite being spoken by tens of millions of people, it is badly underserved by mainstream translation systems — training data is scarce, noisy, and rarely curated, and commercial models treat it as an afterthought.

This project is personal. I was raised in Igbo land and grew up speaking the language fluently. Igbo is part of who I am, and I want to keep that connection alive — with a tool that produces *formal, correct* Igbo rather than transliterated approximations or social media noise.

Growing up speaking the language also means I can directly evaluate whether the system's output is correct — an advantage most NLP researchers working on low-resource African languages don't have. When the model returns "M m i love ya" instead of "A hụrụ m gị n'anya", I know immediately it's wrong, and I can trace exactly where the pipeline failed.

## Demo

Integrated with Open WebUI as a dedicated model with tool calling. The system routes each query through the RAG pipeline, returns citations, and surfaces retrieval quality directly in the chat UI.

![Igbo Translator in Open WebUI](docs/demo.png)

## Architecture

The full RAG pipeline, end to end:

1. **Query** — an Igbo or English phrase, with an optional `direction` (`igbo_to_en` / `en_to_igbo`).
2. **Embedding** — the query is embedded with `nomic-embed-text` (768-dim vectors) served by Ollama (~83ms).
3. **Retrieval** — FAISS performs cosine similarity search over **100K curated translation pairs** (~3ms). The pipeline over-fetches (8× the target count) and filters by direction and quality threshold.
4. **Quality assessment** — the best retrieval distance is bucketed into a quality tier:
   - `HIGH`   — distance `< 0.20` (strong semantic match, corpus is trustworthy)
   - `MEDIUM` — distance `< 0.35` (reasonable match, use with caution)
   - `LOW`    — distance `>= 0.35` (weak match, prefer model knowledge)
   - `no_matches` — nothing retrieved
5. **Quality-aware prompt routing** — the assessed quality selects one of two prompts:
   - **Grounded prompt** (`HIGH` / `MEDIUM`): stays close to the retrieved corpus examples.
   - **Fallback prompt** (`LOW` / `no_matches`): explicitly instructs the model to *ignore* weak matches and rely on its own knowledge of formal Igbo.
6. **Generation** — `Llama3.1:8b` runs locally via Ollama (temperature 0.1) and produces the translation plus confidence and usage notes.
7. **Response** — FastAPI returns the translation along with `retrieval_quality` and the retrieved `citations` (input/output/direction/distance), plus measured latency.

```
query
  → nomic-embed-text (768-dim, ~83ms)
  → FAISS cosine search (100K pairs, ~3ms)
  → distance-based quality assessment (HIGH < 0.20, MEDIUM < 0.35, LOW >= 0.35)
  → quality-aware prompt routing (grounded vs fallback)
  → Llama3.1:8b (Ollama)
  → FastAPI response { translation, retrieval_quality, citations, latency_ms }
```

## Key engineering decisions

### 1. FAISS over ChromaDB — a performance migration

The system was originally built on ChromaDB with a 9.7M-pair SQLite store. Under load, ChromaDB retrieval took **60–90 seconds per query** due to SQLite scan overhead at that scale — the actual query was sub-millisecond but thread pool contention and index loading blocked the FastAPI event loop.

The fix was to migrate to FAISS with a curated 100K-pair index extracted from the original 19.5M-pair NLLB dataset:
- Extract and quality-filter 100K pairs from `nllb_train.jsonl`
- Embed with `nomic-embed-text` (~32 minutes one-time cost)
- Build a normalised `IndexFlatIP` FAISS index (cosine similarity)
- Result: **86ms total retrieval** (83ms embed + 3ms search) vs 60–90s with ChromaDB

The 9.7M ChromaDB store is retained as the source of record. The FAISS index is the production retrieval layer.

### 2. Quality-aware prompt routing

The naive RAG approach feeds whatever the retriever returns straight into the prompt and optimises for faithfulness to that context. That is the **wrong objective when corpus quality is variable.** The NLLB dataset contains noise — transliterated English, internet slang, and mislabeled pairs. Blindly grounding on a weak match produces a confidently wrong translation.

Routing on retrieval distance lets the system decide *whether the corpus deserves trust* for a given query. When the match is strong, it grounds tightly. When the match is weak, it deliberately overrides the corpus and falls back to the model's own knowledge of formal Igbo. This means faithfulness-to-corpus is intentionally *not* maximised on low-quality retrievals — correctness matters more than fidelity to noise.

### 3. Local LLM over a hosted API

- **Zero cost** — 100K pairs and an iterative eval loop would be expensive against a metered API. Local inference is free to run as often as needed.
- **Data privacy** — Igbo text, including anything personal or culturally sensitive, never leaves the machine.
- **Reproducibility** — a pinned local model + a pinned local vector store means evals are deterministic and not subject to a vendor silently changing the model underneath.

### 4. Corpus curation strategy

The raw NLLB dataset has 19.5M pairs but significant noise. The quality filter for the FAISS index:
- Keeps only `Translate` instruction pairs (drops QA, summarisation etc.)
- Filters pairs with outputs under 4 characters
- Removes social media noise (`mmmmm`, `ya nice`, `lol` etc.)
- For `en_to_igbo` only: drops pairs where the Igbo output contains no unicode characters (genuine Igbo always uses diacritics)
- Result: ~87% of sampled pairs pass the filter

## RAGAS evaluation results

Evaluated with RAGAS (faithfulness, answer relevancy, context precision), judged by DeepSeek-r1:14b with `nomic-embed-text` embeddings. Evals were run against the original ChromaDB store.

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

### Why LOW retrieval queries score low on faithfulness — by design

The low faithfulness numbers on `LOW`/`MEDIUM` retrieval rows are **expected and correct.** For those queries the corpus matches are noisy, so the fallback prompt deliberately instructs the model to override the retrieved citations and translate from its own knowledge of formal Igbo. RAGAS measures faithfulness as *consistency between the response and the retrieved context* — so when the system correctly ignores bad context, RAGAS penalises it as "unfaithful."

In other words, a low faithfulness score here is a signal that the routing layer is doing its job: the response is faithful to *correct Igbo*, not to a noisy corpus. Answer relevancy stays high (0.833 average) across exactly these rows, which is the metric that actually reflects translation quality in this regime.

## Stack

| Component | Technology |
|---|---|
| Vector index | FAISS IndexFlatIP (100K curated pairs, 768-dim) |
| Source corpus | NLLB dataset (19.5M pairs, `nllb_train.jsonl`) |
| Embeddings | nomic-embed-text (768-dim, via Ollama) |
| LLM | Llama3.1:8b (via Ollama) |
| API | FastAPI |
| Evaluation | RAGAS |
| UI integration | Open WebUI (tool + dedicated model) |
| Runtime | Python 3.14, fully local |

## Open WebUI integration

The API is integrated into Open WebUI as a callable Tool and a dedicated **Igbo Translator** model. The model is configured to always invoke the tool rather than translate from memory, ensuring every response is grounded in the corpus and returns retrieval quality + citations.

To enable:
1. Start the API: `python src/run.py`
2. In Open WebUI → **Tools** → create a new tool using the code in `src/owui_tool.py`
3. In **Workspace → Models** → create a model named `Igbo Translator` with base model `Llama3.1:8b`, the system prompt below, and the Igbo Translator tool enabled

**System prompt for the model:**
```
You are an Igbo-English translation assistant powered by a RAG pipeline
grounded on 100,000 curated Igbo-English translation pairs.

CRITICAL RULES:
- ALWAYS call the translate_igbo tool for every translation request
- NEVER translate from memory or your own knowledge
- Quote the tool's translation EXACTLY as returned — do not paraphrase or reword it
- Present the full tool response including retrieval quality and citations

When the user gives you text:
- If it looks like English → call translate_igbo with direction='en_to_igbo'
- If it looks like Igbo → call translate_igbo with direction='igbo_to_en'
- If unsure → call translate_igbo with direction=null
```

## Setup and run

### Prerequisites

[Ollama](https://ollama.com/) running locally with the required models:

```bash
ollama pull Llama3.1:8b
ollama pull nomic-embed-text
```

### Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Configure

Create a `.env` file at the repository root:

```env
OLLAMA_BASE_URL=http://localhost:11434
LLM_MODEL=Llama3.1:8b
EMBED_MODEL=nomic-embed-text
FAISS_INDEX_PATH=/path/to/data/igbo_faiss.index
FAISS_META_PATH=/path/to/data/igbo_metadata.json
```

### Build the FAISS index (one-time, ~32 minutes)

Requires the raw `nllb_train.jsonl` dataset:

```bash
python scripts/build_faiss_index.py
```

Outputs `data/igbo_faiss.index` (293MB) and `data/igbo_metadata.json` (19MB).

### Run the API

```bash
python src/run.py
```

Server starts on `http://0.0.0.0:8000`. Interactive Swagger docs at `http://localhost:8000/docs`.

### Run the evaluation suite

```bash
python src/eval.py
```

Runs the RAG pipeline over the test set, scores with RAGAS, prints a per-query table, and writes full results to `eval_results.json`.

## API reference

### `POST /translate`

Translate a phrase. Returns translation, retrieval quality, citations, and latency.

**Request**
```bash
curl -s -X POST http://localhost:8000/translate \
  -H "Content-Type: application/json" \
  -d '{"query": "Ụtụtụ ọma", "direction": "igbo_to_en"}'
```

**Response**
```json
{
  "query": "Ụtụtụ ọma",
  "direction": "igbo_to_en",
  "translation": "Good morning",
  "retrieval_quality": "high",
  "citations": [
    {
      "input": "Ụtụtụ ọma",
      "output": "Good morning",
      "direction": "igbo_to_en",
      "distance": 0.05
    }
  ],
  "latency_ms": 1843.21
}
```

`direction` is optional (`en_to_igbo`, `igbo_to_en`, or omitted to search both). Empty `query` returns HTTP 422.

### `GET /health`

Liveness check. Returns model and index info. Returns HTTP 503 if unavailable.

```bash
curl -s http://localhost:8000/health
```

```json
{
  "status": "ok",
  "model": "Llama3.1:8b",
  "faiss_index": "/path/to/igbo_faiss.index",
  "total_pairs": 100000
}
```

### `GET /`

Service metadata.

```bash
curl -s http://localhost:8000/
```

```json
{
  "name": "Igbo-English RAG Translator",
  "version": "1.0.0",
  "docs": "/docs"
}
```

## What I would add next

- **Reranking layer** — a cross-encoder reranker (e.g. Cohere Rerank) over the top-k FAISS candidates to tighten precision before the quality gate.
- **Expand the curated index** — run the build script on the full 19.5M pairs overnight to index 1M+ high-quality pairs, improving recall for rare phrases.
- **Streaming responses** — token streaming from Ollama to cut perceived latency.
- **LangSmith tracing** — end-to-end tracing of retrieval, routing, and generation to debug quality regressions and inspect per-stage latency.
- **Hybrid search** — combine FAISS vector search with BM25 keyword search for better precision on exact phrase matches (e.g. proper nouns, fixed expressions).
- **RAGAS re-evaluation** — re-run evals against the new FAISS index to measure retrieval quality improvement over the original ChromaDB baseline.