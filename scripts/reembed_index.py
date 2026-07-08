"""
reembed_index.py

Rebuilds the FAISS index from the EXISTING igbo_metadata.json, fixing the
query/document embedding-format mismatch.

Background
----------
The original index (build_faiss_index.py) embedded each pair as the bilingual
string "input | output". But queries only ever contain the source phrase, so
`embed_query` embeds "input" alone — a different region of embedding space.
Result: real matches never rank and retrieval returns high-similarity noise.

This script re-embeds every pair using ONLY the source `input` phrase, with the
nomic-embed-text `search_document:` task prefix (queries use `search_query:`).
Metadata order is preserved 1:1, so vectors stay aligned with metadata — the
metadata file itself is NOT modified.

Safety
------
- Reads metadata from FAISS_META_PATH, writes the new index to
  FAISS_INDEX_PATH + ".rebuild". The live index is untouched until you swap.
- Embeddings stream into a disk-backed memmap (flat RAM usage).
- Checkpointed: re-run to resume from where it stopped.

Run:
    /Users/kere/igbo-rag/.venv/bin/python scripts/reembed_index.py

When it finishes it prints the exact swap + restart commands.
"""

import json
import os
import sys
import time
import urllib.request

import numpy as np
from dotenv import load_dotenv

load_dotenv()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
FAISS_INDEX_PATH = os.getenv(
    "FAISS_INDEX_PATH",
    os.path.expanduser("~/projects/igbo-rag/data/igbo_faiss.index"),
)
FAISS_META_PATH = os.getenv(
    "FAISS_META_PATH",
    os.path.expanduser("~/projects/igbo-rag/data/igbo_metadata.json"),
)

DATA_DIR = os.path.dirname(FAISS_INDEX_PATH)
REBUILD_INDEX_PATH = FAISS_INDEX_PATH + ".rebuild"
MEMMAP_PATH = os.path.join(DATA_DIR, "igbo_embeddings_rebuild.dat")
STATE_PATH = os.path.join(DATA_DIR, "igbo_rebuild_state.json")

CHECKPOINT_EVERY = 5000


def log(msg: str) -> None:
    print(f"[reembed {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def embed(text: str) -> list:
    payload = json.dumps({"model": EMBED_MODEL, "prompt": text}).encode("utf-8")
    req = urllib.request.Request(
        f"{OLLAMA_BASE_URL}/api/embeddings",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())["embedding"]


def main():
    import faiss

    log(f"Loading metadata from {FAISS_META_PATH}")
    with open(FAISS_META_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    n = len(metadata)
    log(f"Metadata entries: {n:,}")

    dim = len(embed("search_document: test"))
    log(f"Embedding dim: {dim}")

    # Disk-backed store so RAM stays flat regardless of corpus size.
    mode = "r+" if os.path.exists(MEMMAP_PATH) else "w+"
    embeddings = np.memmap(MEMMAP_PATH, dtype=np.float32, mode=mode, shape=(n, dim))

    start_idx = 0
    if os.path.exists(STATE_PATH):
        with open(STATE_PATH) as f:
            start_idx = json.load(f).get("next_idx", 0)
        log(f"Resuming from checkpoint at index {start_idx:,}")

    if start_idx < n:
        log(f"Embedding {n - start_idx:,} pairs (input-only, search_document prefix)...")
        t0 = time.time()
        done_at_start = start_idx
        for i in range(start_idx, n):
            src = metadata[i]["input"].strip()
            embeddings[i] = np.asarray(embed(f"search_document: {src}"), dtype=np.float32)

            if (i + 1) % 500 == 0 or i + 1 == n:
                elapsed = time.time() - t0
                rate = (i + 1 - done_at_start) / elapsed if elapsed > 0 else 0
                eta = (n - (i + 1)) / rate / 60 if rate > 0 else 0
                log(f"  {i + 1:,}/{n:,}  {rate:.0f}/s  ETA {eta:.1f} min")

            if (i + 1) % CHECKPOINT_EVERY == 0:
                embeddings.flush()
                with open(STATE_PATH, "w") as f:
                    json.dump({"next_idx": i + 1}, f)

        embeddings.flush()
        with open(STATE_PATH, "w") as f:
            json.dump({"next_idx": n}, f)
        log(f"Embedding complete in {(time.time() - t0) / 60:.1f} min")
    else:
        log("All embeddings already present — skipping to index build")

    log("Normalising vectors (cosine similarity)...")
    vecs = np.ascontiguousarray(np.array(embeddings, dtype=np.float32))
    faiss.normalize_L2(vecs)

    log("Building FAISS IndexFlatIP...")
    index = faiss.IndexFlatIP(dim)
    index.add(vecs)
    log(f"Index built: {index.ntotal:,} vectors, {dim}-dim")

    faiss.write_index(index, REBUILD_INDEX_PATH)
    size_mb = os.path.getsize(REBUILD_INDEX_PATH) / (1024 ** 2)
    log(f"Wrote rebuilt index: {REBUILD_INDEX_PATH} ({size_mb:.0f} MB)")

    # Clean up scratch files (keep the .rebuild index for the swap).
    for p in (MEMMAP_PATH, STATE_PATH):
        if os.path.exists(p):
            os.remove(p)

    print("\n" + "=" * 64)
    print("REBUILD DONE. To go live, swap the index and restart the API:")
    print(f'  mv "{FAISS_INDEX_PATH}" "{FAISS_INDEX_PATH}.bak"')
    print(f'  mv "{REBUILD_INDEX_PATH}" "{FAISS_INDEX_PATH}"')
    print("  pm2 restart igbo-rag-api")
    print("=" * 64)


if __name__ == "__main__":
    main()
