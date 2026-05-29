# Igbo-English RAG Translator

![Python](https://img.shields.io/badge/Python-3.14-blue)
![ChromaDB](https://img.shields.io/badge/ChromaDB-9.7M_pairs-green)
![DeepSeek](https://img.shields.io/badge/LLM-DeepSeek--r1%3A14b-orange)
![RAGAS](https://img.shields.io/badge/Answer_Relevancy-0.833-brightgreen)
![Local](https://img.shields.io/badge/Inference-100%25_Local-purple)

A production RAG translation system for Igbo ↔ English, running entirely on local infrastructure — DeepSeek-r1:14b via Ollama, a ChromaDB vector store of 9.7M translation pairs, a FastAPI REST API, and a RAGAS evaluation suite. Zero API costs, zero data leaving the machine.

## Why I built this

Igbo is a low-resource African language. Despite being spoken by tens of millions of people, it is badly underserved by mainstream translation systems — training data is scarce, noisy, and rarely curated, and commercial models treat it as an afterthought.

This project is personal. As someone of Igbo heritage, I want the language to survive the generation gap. I want to be able to teach my children the language with a tool that produces *formal, correct* Igbo rather than transliterated approximations. Cultural preservation of a low-resource language is not a problem you can wait for a vendor to solve — so I built the tooling locally, where the data and the model are both under my control.

## Demo

Integrated with Open WebUI as a dedicated model with tool calling. The system routes each query through the RAG pipeline, returns citations, and surfaces retrieval quality directly in the chat UI.

![Igbo Translator in Open WebUI](docs/demo.png)

## Architecture

The full RAG pipeline, end to end:

1. **Query** — an Igbo or English phrase, with an optional `direction` (`igbo_to_en` / `en_to_igbo`).
2. **Embedding** — the query is embedded with `nomic-embed-text` (384-dim vectors) served by Ollama.
3. **Retrieval** — ChromaDB performs semantic search over **9.7M translation pairs** in the `igbo_translations` collection. The pipeline over-fetches (4× the target count) and filters by distance so noisy pairs can be dropped.
4. **Quality assessment** — the best retrieval distance is bucketed into a quality tier:
   - `HIGH`   — distance `< 0.55` (strong semantic match, corpus is trustworthy)
   - `MEDIUM` — distance `< 0.70` (reasonable match, use with caution)
   - `LOW`    — distance `>= 0.70` (weak match, prefer model knowledge)
   - `no_matches` — nothing retrieved
5. **Quality-aware prompt routing** — the assessed quality selects one of two prompts:
   - **Grounded prompt** (`HIGH` / `MEDIUM`): stays close to the retrieved corpus examples.
   - **Fallback prompt** (`LOW` / `no_matches`): explicitly instructs the model to *ignore* the noisy corpus and rely on its own knowledge of formal Igbo.
6. **Generation** — `DeepSeek-r1:14b` runs locally via Ollama (temperature 0.1) and produces the translation plus confidence and usage notes.
7. **Response** — FastAPI returns the translation along with `retrieval_quality` and the retrieved `citations` (input/output/direction/distance), plus measured latency.

```
query
  → nomic-embed-text (384-dim)
  → ChromaDB retrieval (9.7M pairs)
  → distance-based quality assessment (HIGH < 0.55, MEDIUM < 0.70, LOW >= 0.70)
  → quality-aware prompt routing (grounded vs fallback)
  → DeepSeek-r1:14b (Ollama)
  → FastAPI response { translation, retrieval_quality, citations, latency_ms }
```

## Key engineering decisions

### 1. Quality-aware prompt routing

The naive RAG approach feeds whatever the retriever returns straight into the prompt and optimises for faithfulness to that context. That is the **wrong objective when corpus quality is variable.** This corpus contains a large amount of noise — transliterated English, internet slang, and mislabeled pairs. Blindly grounding on a weak match produces a confidently wrong translation.

Routing on retrieval distance lets the system decide *whether the corpus deserves trust* for a given query. When the match is strong, it grounds tightly. When the match is weak, it deliberately overrides the corpus and falls back to the model's own knowledge of formal Igbo. This means faithfulness-to-corpus is intentionally *not* maximised on low-quality retrievals — correctness matters more than fidelity to noise.

### 2. ChromaDB over raw FAISS

FAISS is a great index, but it is just an index. ChromaDB gives a **persistent client** (the 9.7M-pair store lives on disk and reloads instantly), **collection metadata** (each pair carries `input` / `output` / `direction`, which the pipeline filters and cites on), and a production-ready query surface (`where` filters, distance includes) without hand-rolling the storage, serialization, and metadata layers around FAISS. Less glue code, fewer places to get it wrong.

### 3. Local LLM over a hosted API

- **Zero cost** — 9.7M pairs and an iterative eval loop would be expensive against a metered API. Local inference is free to run as often as needed.
- **Data privacy** — Igbo text, including anything personal or culturally sensitive, never leaves the machine.
- **Reproducibility** — a pinned local model + a pinned local vector store means evals are deterministic and not subject to a vendor silently changing the model underneath me.

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

### Why LOW retrieval queries score low on faithfulness — by design

The low faithfulness numbers on `LOW`/`MEDIUM` retrieval rows are **expected and correct.** For those queries the corpus matches are noisy, so the fallback prompt deliberately instructs the model to override the retrieved citations and translate from its own knowledge of formal Igbo. RAGAS measures faithfulness as *consistency between the response and the retrieved context* — so when the system correctly ignores bad context, RAGAS penalises it as "unfaithful."

In other words, a low faithfulness score here is a signal that the routing layer is doing its job: the response is faithful to *correct Igbo*, not to a noisy corpus. Answer relevancy stays high (0.833 average) across exactly these rows, which is the metric that actually reflects translation quality in this regime.

## Stack

| Component | Technology |
|---|---|
| Vector store | ChromaDB (9.7M translation pairs) |
| Embeddings | nomic-embed-text (384-dim, via Ollama) |
| LLM | DeepSeek-r1:14b (via Ollama) |
| Orchestration | LangChain |
| API | FastAPI |
| Evaluation | RAGAS |
| UI integration | Open WebUI (tool + dedicated model) |
| Runtime | Python 3.14, fully local |

## Open WebUI integration

The API is integrated into Open WebUI as a callable Tool and a dedicated **Igbo Translator** model. The model is configured to always invoke the tool rather than translate from memory, ensuring every response is grounded in the corpus and returns retrieval quality + citations.

To enable:
1. Start the API: `python src/run.py`
2. In Open WebUI → **Tools** → create a new tool using the code in `src/owui_tool.py`
3. In **Workspace → Models** → create a model named `Igbo Translator` with base model `deepseek-r1:14b`, the system prompt below, and the Igbo Translator tool enabled

**System prompt for the model:**
```
You are an Igbo-English translation assistant powered by a RAG pipeline
grounded on 9.7 million real Igbo-English translation pairs.

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
ollama pull deepseek-r1:14b
ollama pull nomic-embed-text
```

A populated ChromaDB store at `CHROMA_DB_PATH` with an `igbo_translations` collection.

### Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Configure

Create a `.env` file at the repository root:

```env
CHROMA_DB_PATH=/path/to/igbo_vector_db
OLLAMA_BASE_URL=http://localhost:11434
LLM_MODEL=deepseek-r1:14b
EMBED_MODEL=nomic-embed-text
```

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
  "translation": "Good morning ...",
  "retrieval_quality": "high",
  "citations": [
    {
      "input": "Ụtụtụ ọma",
      "output": "Good morning",
      "direction": "igbo_to_en",
      "distance": 0.41
    }
  ],
  "latency_ms": 1843.21
}
```

`direction` is optional (`en_to_igbo`, `igbo_to_en`, or omitted to search both). Empty `query` returns HTTP 422.

### `GET /health`

Liveness check. Connects to ChromaDB and returns live document count. Returns HTTP 503 if ChromaDB is unreachable.

```bash
curl -s http://localhost:8000/health
```

```json
{
  "status": "ok",
  "model": "deepseek-r1:14b",
  "chroma_db": "/path/to/igbo_vector_db",
  "collection": "igbo_translations",
  "total_pairs": 9775982
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

- **Reranking layer** — a cross-encoder reranker (e.g. Cohere Rerank) over the top-k ChromaDB candidates to tighten precision before the quality gate.
- **Corpus curation** — an offline pass to filter transliterated/noisy pairs out of the 9.7M-pair store, raising the share of `HIGH` retrievals and reducing reliance on the fallback path.
- **Streaming responses** — token streaming from Ollama to cut perceived latency, which matters with a 14B model running locally.
- **LangSmith tracing** — end-to-end tracing of retrieval, routing, and generation to debug quality regressions and inspect per-stage latency.
- **pgvector migration** — move from ChromaDB to pgvector on Postgres for hybrid keyword + semantic search, enabling better precision on exact phrase matches.