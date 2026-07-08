"""
Isolated FAISS search latency check — no Ollama, no translate(), no /ops
overhead. Just: load the index once, then run repeated searches and report
the distribution, so we can tell whether ~500ms is real or a symptom of
CPU contention with a concurrently running LLM.

Run from your project root (wherever data/igbo_faiss.index lives):

    python check_faiss_latency.py

Override the index path if needed:

    FAISS_INDEX_PATH=/path/to/igbo_faiss.index python check_faiss_latency.py
"""

import os
import statistics
import time

import faiss
import numpy as np

INDEX_PATH = os.environ.get(
    "FAISS_INDEX_PATH", "data/igbo_faiss.index"
)
N_RUNS = 200
K = 5  # matches the k used in rag_pipeline.py's real search calls


def main():
    if not os.path.exists(INDEX_PATH):
        print(f"Index not found at {INDEX_PATH} — set FAISS_INDEX_PATH.")
        return

    print(f"Loading index from {INDEX_PATH} ...")
    t0 = time.perf_counter()
    index = faiss.read_index(INDEX_PATH)
    load_ms = (time.perf_counter() - t0) * 1000
    print(f"Index loaded in {load_ms:.1f}ms — {index.ntotal:,} vectors, dim={index.d}\n")

    dim = index.d

    # Generate all probe vectors up front so vector generation itself
    # never shows up inside the timed search window.
    vectors = np.random.rand(N_RUNS, dim).astype(np.float32)
    faiss.normalize_L2(vectors)

    # First search separately — cold-cache effects (paging the index into
    # memory, thread pool spin-up) can make the very first call slower
    # than steady-state, which would explain a high number if /ops is
    # hitting a cold index on every request rather than a warm one.
    t0 = time.perf_counter()
    index.search(vectors[0:1], K)
    first_ms = (time.perf_counter() - t0) * 1000
    print(f"First search (cold):  {first_ms:.2f}ms\n")

    # Steady-state distribution.
    timings = []
    for i in range(1, N_RUNS):
        t0 = time.perf_counter()
        index.search(vectors[i:i+1], K)
        timings.append((time.perf_counter() - t0) * 1000)

    print(f"Steady-state over {len(timings)} searches (k={K}):")
    print(f"  min:    {min(timings):.2f}ms")
    print(f"  median: {statistics.median(timings):.2f}ms")
    print(f"  mean:   {statistics.mean(timings):.2f}ms")
    print(f"  p95:    {sorted(timings)[int(len(timings) * 0.95)]:.2f}ms")
    print(f"  max:    {max(timings):.2f}ms")

    print(
        "\nIf steady-state median is single-digit ms but /ops reports ~500ms, "
        "the discrepancy is contention (e.g. Ollama running concurrently during "
        "the /ops request) rather than the index itself. If median here is "
        "*also* consistently high, the index/hardware genuinely got slower "
        "since the original ~3ms measurement."
    )


if __name__ == "__main__":
    main()