"""
weaviate_benchmark.py

Head-to-head retrieval benchmark: FAISS vs Weaviate (pure vector) vs
Weaviate (hybrid).

Uses the same eval queries as src/eval.py and the same embedding function
(embeddings.embed_query) so the only variable is the retrieval backend.

Run after ingesting into Weaviate:

    python scripts/weaviate_benchmark.py

Environment variables:
    VECTOR_BACKEND is ignored here — the script calls each backend directly.
    WEAVIATE_HYBRID_ALPHA  default 0.5  (0 = keyword, 1 = vector)
"""

import json
import os
import sys
import time
from pathlib import Path

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_src = os.path.join(_project_root, "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

from dotenv import load_dotenv

load_dotenv()

from embeddings import embed_query
from rag_pipeline import _retrieve_from_faiss, assess_retrieval_quality
from weaviate_store import retrieve_translation_pairs as weaviate_retrieve

RESULTS_PATH = Path(__file__).resolve().parent.parent / "weaviate_benchmark_results.json"
HYBRID_ALPHA = float(os.getenv("WEAVIATE_HYBRID_ALPHA", "0.5"))

TEST_CASES = [
    {"query": "Ụtụtụ ọma", "direction": "igbo_to_en"},
    {"query": "A hụrụ m gị n'anya", "direction": "igbo_to_en"},
    {"query": "Kedu ka i mere?", "direction": "igbo_to_en"},
    {"query": "Biko nọdụ ala", "direction": "igbo_to_en"},
    {"query": "I love you", "direction": "en_to_igbo"},
    {"query": "Good morning", "direction": "en_to_igbo"},
    {"query": "Thank you", "direction": "en_to_igbo"},
    {"query": "Please sit down", "direction": "en_to_igbo"},
    {"query": "Where do you go to school?", "direction": "en_to_igbo"},
    {"query": "Where is the elephant?", "direction": "en_to_igbo"},
]


def _top1_similarity(pairs: list) -> float:
    if not pairs:
        return 0.0
    return pairs[0]["similarity"]


def _top1_text(pairs: list) -> str:
    if not pairs:
        return ""
    p = pairs[0]
    return f'{p["input"]} → {p["output"]}'


def benchmark_case(query: str, direction: str):
    # Warm up the shared embedding function once.
    _ = embed_query(query)

    # FAISS
    t0 = time.perf_counter()
    faiss_pairs = _retrieve_from_faiss(query, direction=direction)
    faiss_ms = (time.perf_counter() - t0) * 1000

    # Weaviate vector-only
    t0 = time.perf_counter()
    weaviate_pairs = weaviate_retrieve(
        query, direction=direction, use_hybrid=False
    )
    weaviate_ms = (time.perf_counter() - t0) * 1000

    # Weaviate hybrid
    t0 = time.perf_counter()
    hybrid_pairs = weaviate_retrieve(
        query, direction=direction, use_hybrid=True, hybrid_alpha=HYBRID_ALPHA
    )
    hybrid_ms = (time.perf_counter() - t0) * 1000

    return {
        "query": query,
        "direction": direction,
        "faiss": {
            "latency_ms": round(faiss_ms, 2),
            "top1_similarity": round(_top1_similarity(faiss_pairs), 4),
            "top1_pair": _top1_text(faiss_pairs),
            "retrieval_quality": assess_retrieval_quality(faiss_pairs),
            "citations": faiss_pairs,
        },
        "weaviate_vector": {
            "latency_ms": round(weaviate_ms, 2),
            "top1_similarity": round(_top1_similarity(weaviate_pairs), 4),
            "top1_pair": _top1_text(weaviate_pairs),
            "retrieval_quality": assess_retrieval_quality(weaviate_pairs),
            "citations": weaviate_pairs,
        },
        "weaviate_hybrid": {
            "latency_ms": round(hybrid_ms, 2),
            "top1_similarity": round(_top1_similarity(hybrid_pairs), 4),
            "top1_pair": _top1_text(hybrid_pairs),
            "retrieval_quality": assess_retrieval_quality(hybrid_pairs),
            "citations": hybrid_pairs,
        },
    }


def print_table(results: list[dict]):
    headers = ["Query", "Dir", "FAISS sim", "FAISS ms", "W-v sim", "W-v ms", "W-h sim", "W-h ms"]
    col_query = max(len(h) for h in headers)
    col_query = max(col_query, max(len(r["query"]) for r in results))
    col_query = min(col_query, 28)

    header_line = (
        f"{'Query':<{col_query}}  {'Dir':<12}  "
        f"{'FAISS':>12}  {'Weaviate v':>12}  {'Weaviate h':>12}"
    )
    subheader = (
        f"{'':<{col_query}}  {'':12}  "
        f"{'sim':>6} {'ms':>5}  {'sim':>6} {'ms':>5}  {'sim':>6} {'ms':>5}"
    )
    print("=" * len(header_line))
    print("Retrieval benchmark: FAISS vs Weaviate")
    print("=" * len(header_line))
    print(header_line)
    print(subheader)
    print("-" * len(header_line))

    for r in results:
        label = r["query"] if len(r["query"]) <= col_query else r["query"][: col_query - 1] + "…"
        print(
            f"{label:<{col_query}}  {r['direction']:<12}  "
            f"{r['faiss']['top1_similarity']:>6.3f} {r['faiss']['latency_ms']:>5.1f}  "
            f"{r['weaviate_vector']['top1_similarity']:>6.3f} {r['weaviate_vector']['latency_ms']:>5.1f}  "
            f"{r['weaviate_hybrid']['top1_similarity']:>6.3f} {r['weaviate_hybrid']['latency_ms']:>5.1f}"
        )

    print("=" * len(header_line))


def main():
    print("Igbo-English RAG — Retrieval backend benchmark")
    print(f"Hybrid alpha: {HYBRID_ALPHA}")
    print(f"Test cases: {len(TEST_CASES)}\n")

    results = []
    for i, case in enumerate(TEST_CASES, 1):
        print(f"[{i}/{len(TEST_CASES)}] {case['direction']}: {case['query']!r}")
        results.append(benchmark_case(case["query"], case["direction"]))

    print_table(results)

    summary = {
        "faiss_avg_latency_ms": round(
            sum(r["faiss"]["latency_ms"] for r in results) / len(results), 2
        ),
        "weaviate_vector_avg_latency_ms": round(
            sum(r["weaviate_vector"]["latency_ms"] for r in results) / len(results), 2
        ),
        "weaviate_hybrid_avg_latency_ms": round(
            sum(r["weaviate_hybrid"]["latency_ms"] for r in results) / len(results), 2
        ),
        "faiss_high_quality_count": sum(
            1 for r in results if r["faiss"]["retrieval_quality"] == "high"
        ),
        "weaviate_vector_high_quality_count": sum(
            1 for r in results if r["weaviate_vector"]["retrieval_quality"] == "high"
        ),
        "weaviate_hybrid_high_quality_count": sum(
            1 for r in results if r["weaviate_hybrid"]["retrieval_quality"] == "high"
        ),
    }

    print("\nSummary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    output = {"summary": summary, "results": results, "hybrid_alpha": HYBRID_ALPHA}
    RESULTS_PATH.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n")
    print(f"\nFull results saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
