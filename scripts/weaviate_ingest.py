"""
weaviate_ingest.py

Ingest translation pairs into Weaviate for benchmarking.

Two modes:

1. Fast mode — copy vectors from the existing FAISS index:

       python scripts/weaviate_ingest.py --from-faiss --limit 100000

   This avoids re-embedding and is the fairest head-to-head comparison.

2. Fresh mode — re-embed each source phrase via Ollama:

       python scripts/weaviate_ingest.py --re-embed --limit 5000

   This matches how you would build a production Weaviate corpus from scratch.

The script assumes Weaviate is running and the collection already exists
(see scripts/weaviate_setup.py).
"""

import argparse
import json
import os
import sys
import time

import numpy as np

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_src = os.path.join(_project_root, "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

from dotenv import load_dotenv

load_dotenv()

import faiss

from embeddings import embed_document
from weaviate_store import COLLECTION_NAME, get_client, ingest_batch

FAISS_INDEX_PATH = os.getenv(
    "FAISS_INDEX_PATH",
    os.path.expanduser("~/projects/igbo-rag/data/igbo_faiss.index"),
)
FAISS_META_PATH = os.getenv(
    "FAISS_META_PATH",
    os.path.expanduser("~/projects/igbo-rag/data/igbo_metadata.json"),
)
DEFAULT_BATCH_SIZE = 100


def load_pairs(limit: int | None = None):
    """Load pairs from the FAISS metadata file."""
    print(f"Loading metadata from {FAISS_META_PATH}")
    with open(FAISS_META_PATH, "r", encoding="utf-8") as f:
        pairs = json.load(f)
    if limit and limit > 0:
        pairs = pairs[:limit]
    print(f"Loaded {len(pairs):,} pairs")
    return pairs


def load_faiss_vectors(limit: int | None = None):
    """Load stored vectors directly from the FAISS IndexFlatIP index."""
    print(f"Loading FAISS index from {FAISS_INDEX_PATH}")
    index = faiss.read_index(FAISS_INDEX_PATH)
    ntotal = index.ntotal
    dim = index.d
    print(f"FAISS index: {ntotal:,} vectors, {dim}-dim")

    # IndexFlatIP stores vectors contiguously; reconstruct_n returns them all.
    vectors = index.reconstruct_n(0, ntotal)
    if limit and limit > 0:
        vectors = vectors[:limit]
    return vectors


def build_pairs_for_ingest(pairs: list[dict]) -> list[dict]:
    """Normalise metadata dicts into the shape Weaviate expects."""
    clean = []
    for p in pairs:
        clean.append({
            "input": p.get("input", "").strip(),
            "output": p.get("output", "").strip(),
            "direction": p.get("direction", "").strip(),
            "source": p.get("source", "nllb").strip(),
            "reingested_at": p.get("reingested_at", ""),
        })
    return clean


def main():
    parser = argparse.ArgumentParser(description="Ingest translation pairs into Weaviate")
    parser.add_argument(
        "--from-faiss",
        action="store_true",
        help="Copy pre-computed vectors from the existing FAISS index (fast).",
    )
    parser.add_argument(
        "--re-embed",
        action="store_true",
        help="Re-embed each source phrase via Ollama (slow, fresh build).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Ingest only the first N pairs (useful for quick benchmarks).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Objects per Weaviate batch (default {DEFAULT_BATCH_SIZE}).",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Number of times to retry failed batches (default 3).",
    )
    args = parser.parse_args()

    if not (args.from_faiss or args.re_embed):
        parser.error("Specify one of --from-faiss or --re-embed")
    if args.from_faiss and args.re_embed:
        parser.error("Specify only one of --from-faiss or --re-embed")

    client = get_client()
    try:
        collection = client.collections.get(COLLECTION_NAME)
        count_before = collection.aggregate.over_all().total_count
        print(f"Collection '{COLLECTION_NAME}' exists with {count_before:,} objects")
    except Exception as exc:
        print(f"ERROR: Could not connect to Weaviate collection: {exc}")
        print("Run scripts/weaviate_setup.py first and make sure Weaviate is running.")
        sys.exit(1)

    pairs = load_pairs(args.limit)
    pairs = build_pairs_for_ingest(pairs)

    if args.from_faiss:
        vectors = load_faiss_vectors(args.limit)
        if len(vectors) != len(pairs):
            print(
                f"ERROR: vector count ({len(vectors):,}) != pair count ({len(pairs):,})"
            )
            sys.exit(1)
        embeddings = vectors.tolist()
    else:
        print(f"Re-embedding {len(pairs):,} source phrases via Ollama...")
        embeddings = []
        t0 = time.time()
        for i, p in enumerate(pairs, 1):
            vec = embed_document(p["input"])
            embeddings.append(vec[0].tolist())
            if i % 100 == 0 or i == len(pairs):
                elapsed = time.time() - t0
                rate = i / elapsed if elapsed > 0 else 0
                eta = (len(pairs) - i) / rate / 60 if rate > 0 else 0
                print(f"  {i:,}/{len(pairs):,}  {rate:.1f}/s  ETA {eta:.1f} min")

    print(f"\nInserting {len(pairs):,} objects into Weaviate (batch={args.batch_size})...")
    t0 = time.time()
    total_inserted = 0
    total_failed = 0
    num_batches = (len(pairs) + args.batch_size - 1) // args.batch_size

    for batch_idx in range(num_batches):
        i = batch_idx * args.batch_size
        batch_pairs = pairs[i : i + args.batch_size]
        batch_vectors = embeddings[i : i + args.batch_size]

        inserted, failed = ingest_batch(
            batch_pairs, batch_vectors, batch_size=args.batch_size
        )

        # Retry individual failed objects up to --retries times.
        retry_attempt = 0
        while failed and retry_attempt < args.retries:
            retry_attempt += 1
            print(
                f"  retrying {len(failed)} failed objects from batch "
                f"{batch_idx + 1}/{num_batches} (attempt {retry_attempt}/{args.retries})..."
            )
            retry_pairs = []
            retry_vectors = []
            for obj in failed:
                props = obj.original.get("properties") or getattr(obj, "properties", {})
                vec = obj.original.get("vector") or getattr(obj, "vector", None)
                if props and vec is not None:
                    retry_pairs.append(props)
                    retry_vectors.append(vec)
            if retry_pairs:
                inserted_retry, failed = ingest_batch(
                    retry_pairs, retry_vectors, batch_size=len(retry_pairs)
                )
                inserted += inserted_retry
            else:
                break

        total_inserted += inserted
        total_failed += len(failed)

        if failed:
            print(
                f"  WARNING: batch {batch_idx + 1}/{num_batches} still has "
                f"{len(failed)} failed objects after {args.retries} retries"
            )

        if (batch_idx + 1) % 10 == 0 or total_inserted == len(pairs):
            elapsed = time.time() - t0
            rate = total_inserted / elapsed if elapsed > 0 else 0
            print(
                f"  {total_inserted:,}/{len(pairs):,} inserted  {rate:.1f}/s  "
                f"(failed: {total_failed:,})"
            )

    count_after = collection.aggregate.over_all().total_count
    print(f"\nDone. Collection now has {count_after:,} objects (added {count_after - count_before:,}).")
    print(f"Successfully inserted: {total_inserted:,} | Failed: {total_failed:,}")
    print(f"Total time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
