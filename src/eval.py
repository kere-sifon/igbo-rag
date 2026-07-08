import json
import os
import re
import sys
import time
from pathlib import Path

_src = os.path.dirname(os.path.abspath(__file__))
if _src not in sys.path:
    sys.path.insert(0, _src)

from dotenv import load_dotenv
from langchain_ollama import ChatOllama, OllamaEmbeddings
from ragas import EvaluationDataset, evaluate
from ragas.metrics._answer_relevance import answer_relevancy
from ragas.metrics._context_precision import context_precision
from ragas.metrics._faithfulness import faithfulness
from ragas.run_config import RunConfig

from rag_pipeline import OLLAMA_BASE_URL, LLM_MODEL, translate
from db import save_eval_run

load_dotenv()

EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
# Judge model is decoupled from the translation model: the pipeline translates
# with LLM_MODEL, but RAGAS scoring is more reliable with a stronger reasoner.
# Override with RAGAS_JUDGE_MODEL (e.g. deepseek-r1:14b) without changing the
# model the pipeline actually uses to translate.
JUDGE_MODEL = os.getenv("RAGAS_JUDGE_MODEL", os.getenv("LLM_MODEL", LLM_MODEL))
RESULTS_PATH = Path(__file__).resolve().parent.parent / "eval_results.json"

THINKING_PATTERN = re.compile(
    r"<think>.*?</think>",
    re.DOTALL | re.IGNORECASE,
)

METRIC_NAMES = ("faithfulness", "answer_relevancy", "context_precision")

test_cases = [
    {"query": "Ụtụtụ ọma", "direction": "igbo_to_en", "ground_truth": "Good morning"},
    {"query": "A hụrụ m gị n'anya", "direction": "igbo_to_en", "ground_truth": "I love you"},
    {"query": "Kedu ka i mere?", "direction": "igbo_to_en", "ground_truth": "How are you?"},
    {"query": "Biko nọdụ ala", "direction": "igbo_to_en", "ground_truth": "Please sit down"},
    {"query": "I love you", "direction": "en_to_igbo", "ground_truth": "A hụrụ m gị n'anya"},
    {"query": "Good morning", "direction": "en_to_igbo", "ground_truth": "Ụtụtụ ọma"},
    {"query": "Thank you", "direction": "en_to_igbo", "ground_truth": "Daalụ"},
    {"query": "Please sit down", "direction": "en_to_igbo", "ground_truth": "Biko nọdụ ala"},
]


def strip_thinking(text: str) -> str:
    """Remove DeepSeek reasoning blocks before RAGAS scoring."""
    return THINKING_PATTERN.sub("", text).strip()


def format_user_input(query: str, direction: str) -> str:
    if direction == "igbo_to_en":
        return f"Translate from Igbo to English: {query}"
    if direction == "en_to_igbo":
        return f"Translate from English to Igbo: {query}"
    return query


def citations_to_contexts(citations: list) -> list[str]:
    if not citations:
        return ["No close matches found in corpus."]
    return [
        f"[{c['direction']}] \"{c['input']}\" → \"{c['output']}\" "
        f"(similarity: {c['similarity']:.4f})"
        for c in citations
    ]


def run_pipeline() -> tuple[list[dict], list[dict]]:
    """Call translate() for each test case; return RAGAS samples and run metadata."""
    samples = []
    run_meta = []

    for i, case in enumerate(test_cases, 1):
        query = case["query"]
        direction = case["direction"]
        print(f"[{i}/{len(test_cases)}] {direction}: {query!r}")

        start = time.perf_counter()
        result = translate(query, direction=direction)
        latency_ms = (time.perf_counter() - start) * 1000

        response = strip_thinking(result["response"])

        samples.append(
            {
                "user_input": format_user_input(query, direction),
                "retrieved_contexts": citations_to_contexts(result["citations"]),
                "response": response,
                "reference": case["ground_truth"],
            }
        )
        run_meta.append(
            {
                "query": query,
                "direction": direction,
                "ground_truth": case["ground_truth"],
                "response": response,
                "retrieval_quality": result["retrieval_quality"],
                "latency_ms": round(latency_ms, 2),
                "citations": result["citations"],
            }
        )

    return samples, run_meta


def merge_results(run_meta: list[dict], eval_result) -> dict:
    scores_per_row = eval_result.scores
    queries = []
    for meta, scores in zip(run_meta, scores_per_row):
        row = {**meta}
        for name in METRIC_NAMES:
            value = scores.get(name)
            row[name] = None if value is None else float(value)
        queries.append(row)

    summary = {}
    for name in METRIC_NAMES:
        values = [q[name] for q in queries if q[name] is not None]
        summary[name] = round(sum(values) / len(values), 4) if values else None

    return {
        "judge_model": JUDGE_MODEL,
        "embed_model": EMBED_MODEL,
        "ollama_base_url": OLLAMA_BASE_URL,
        "summary": summary,
        "queries": queries,
    }


def _fmt_score(value: float | None) -> str:
    if value is None:
        return "   n/a"
    return f"{value:6.3f}"


def print_summary_table(results: dict) -> None:
    queries = results["queries"]
    col_query = max(len("Query"), max(len(q["query"]) for q in queries))
    col_query = min(col_query, 28)

    header = (
        f"{'Query':<{col_query}}  {'Dir':<12}  {'Retr':<8}  "
        f"{'Faith':>6}  {'Relv':>6}  {'CtxP':>6}  {'ms':>8}"
    )
    print("\n" + "=" * len(header))
    print("RAGAS Evaluation — Per-query scores")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for q in queries:
        label = q["query"] if len(q["query"]) <= col_query else q["query"][: col_query - 1] + "…"
        print(
            f"{label:<{col_query}}  {q['direction']:<12}  {q['retrieval_quality']:<8}  "
            f"{_fmt_score(q.get('faithfulness'))}  {_fmt_score(q.get('answer_relevancy'))}  "
            f"{_fmt_score(q.get('context_precision'))}  {q['latency_ms']:>8.0f}"
        )

    print("-" * len(header))
    s = results["summary"]
    print(
        f"{'AVERAGE':<{col_query}}  {'':12}  {'':8}  "
        f"{_fmt_score(s.get('faithfulness'))}  {_fmt_score(s.get('answer_relevancy'))}  "
        f"{_fmt_score(s.get('context_precision'))}"
    )
    print("=" * len(header))


def main() -> None:
    print("Igbo-English RAG — RAGAS evaluation")
    print(f"Judge LLM : {JUDGE_MODEL}")
    print(f"Embeddings: {EMBED_MODEL}")
    print(f"Ollama    : {OLLAMA_BASE_URL}\n")

    samples, run_meta = run_pipeline()

    print("\nRunning RAGAS metrics (faithfulness, answer_relevancy, context_precision)...")
    judge_llm = ChatOllama(
        model=JUDGE_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0.0,
    )
    judge_embeddings = OllamaEmbeddings(
        model=EMBED_MODEL,
        base_url=OLLAMA_BASE_URL,
    )

    eval_dataset = EvaluationDataset.from_list(samples)
    run_config = RunConfig(timeout=300, max_workers=1, max_wait=600)
    results = evaluate(
        dataset=eval_dataset,
        metrics=[faithfulness, answer_relevancy, context_precision],
        llm=judge_llm,
        embeddings=judge_embeddings,
        run_config=run_config,
    )

    results = merge_results(run_meta, results)
    print_summary_table(results)

    RESULTS_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n")
    print(f"\nFull results saved to {RESULTS_PATH}")

    try:
        save_eval_run(results)
        print("Snapshot saved to MongoDB (evals collection) — /ops will pick this up.")
    except Exception as exc:
        print(f"Warning: could not save eval run to MongoDB: {exc}")


if __name__ == "__main__":
    main()