"""
reingest_corrections.py

Nightly job: reads unprocessed corrections from MongoDB, embeds them using
nomic-embed-text (same model as the main index), upserts them into the live
FAISS index + metadata, and marks them reingested in MongoDB.

Run manually:
    python scripts/reingest_corrections.py

Run nightly via cron (add with `crontab -e`):
    0 2 * * * cd /Users/kere/projects/igbo-rag && /path/to/python scripts/reingest_corrections.py >> logs/reingest.log 2>&1

Only the latest correction per (query, direction) pair is ingested —
duplicates from multiple corrections of the same phrase are deduplicated.

After ingestion, the running API is notified via POST /feedback/reload
so the in-memory _corrections dict is refreshed without a restart.
"""

import json
import os
import sys
import time
import urllib.request
from datetime import datetime, timezone

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
FAISS_INDEX_PATH = os.getenv(
    "FAISS_INDEX_PATH",
    os.path.expanduser("~/projects/igbo-rag/data/igbo_faiss.index")
)
FAISS_META_PATH = os.getenv(
    "FAISS_META_PATH",
    os.path.expanduser("~/projects/igbo-rag/data/igbo_metadata.json")
)
API_URL = os.getenv("API_URL", "http://localhost:8000")
LOG_PREFIX = f"[reingest {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')} UTC]"


def log(msg: str) -> None:
    print(f"{LOG_PREFIX} {msg}", flush=True)


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def embed_text(text: str) -> list[float]:
    """Embed a single text using nomic-embed-text via Ollama."""
    payload = json.dumps({
        "model": EMBED_MODEL,
        "prompt": text
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{OLLAMA_BASE_URL}/api/embeddings",
        data=payload,
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())["embedding"]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import faiss
    import numpy as np

    log("Starting nightly correction re-ingestion")

    # --- Step 1: Load pending corrections from MongoDB ---
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
        from db import pending_reingestion, mark_reingested
    except ImportError as e:
        log(f"ERROR: Could not import db module: {e}")
        sys.exit(1)

    pending = pending_reingestion()
    if not pending:
        log("No pending corrections — nothing to do.")
        return

    log(f"Found {len(pending)} pending corrections")

    # --- Step 2: Deduplicate — keep latest per (query, direction) ---
    seen: dict = {}
    for c in pending:
        key = f"{c['query'].lower().strip()}||{c['direction']}"
        # pending_reingestion returns newest-first from MongoDB
        if key not in seen:
            seen[key] = c

    deduped = list(seen.values())
    log(f"After deduplication: {len(deduped)} unique pairs to ingest")

    # --- Step 3: Load existing FAISS index + metadata ---
    log(f"Loading FAISS index from {FAISS_INDEX_PATH}")
    index = faiss.read_index(FAISS_INDEX_PATH)
    with open(FAISS_META_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    log(f"Loaded index: {index.ntotal:,} vectors, {len(metadata):,} metadata entries")

    # --- Step 4: Embed and add each correction ---
    new_vectors = []
    new_metadata = []
    failed = []

    for i, correction in enumerate(deduped):
        query = correction["query"].strip()
        correct_translation = correction["correct_translation"].strip()
        direction = correction["direction"]

        # Embed the source phrase only, with the nomic search_document prefix —
        # must match how the main index is built (scripts/reembed_index.py).
        # The query side embeds "search_query: {query}" (see rag_pipeline).
        text = f"search_document: {query}"

        try:
            embedding = embed_text(text)
            new_vectors.append(embedding)
            new_metadata.append({
                "input": query,
                "output": correct_translation,
                "direction": direction,
                "source": "correction",
                "reingested_at": datetime.now(timezone.utc).isoformat(),
            })
            log(f"  [{i+1}/{len(deduped)}] Embedded: '{query}' → '{correct_translation}' ({direction})")
        except Exception as e:
            log(f"  [{i+1}/{len(deduped)}] FAILED embedding '{query}': {e}")
            failed.append(correction["query"])
            continue

        # Small delay to avoid overwhelming Ollama
        time.sleep(0.1)

    if not new_vectors:
        log("No vectors to add — all embeddings failed.")
        return

    # --- Step 5: Add to FAISS index ---
    log(f"Adding {len(new_vectors)} new vectors to FAISS index...")
    vecs_np = np.array(new_vectors, dtype=np.float32)
    faiss.normalize_L2(vecs_np)
    index.add(vecs_np)
    log(f"Index now has {index.ntotal:,} vectors")

    # --- Step 6: Save updated index + metadata ---
    log(f"Saving updated index to {FAISS_INDEX_PATH}")
    faiss.write_index(index, FAISS_INDEX_PATH)

    metadata.extend(new_metadata)
    with open(FAISS_META_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False)
    log(f"Saved metadata: {len(metadata):,} entries")

    # --- Step 7: Mark reingested in MongoDB ---
    ingested_queries = [m["input"] for m in new_metadata]
    mark_reingested(ingested_queries)
    log(f"Marked {len(ingested_queries)} corrections as reingested in MongoDB")

    # --- Step 8: Notify running API to reload corrections ---
    try:
        req = urllib.request.Request(
            f"{API_URL}/feedback/reload",
            data=b"{}",
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            result = json.loads(resp.read())
            log(f"API reloaded corrections: {result.get('total_corrections')} in memory")
    except Exception as e:
        log(f"WARNING: Could not notify API to reload: {e} (restart API manually if needed)")

    # --- Summary ---
    log(f"Re-ingestion complete:")
    log(f"  Ingested : {len(new_metadata)}")
    log(f"  Failed   : {len(failed)}")
    if failed:
        log(f"  Failed queries: {failed}")


if __name__ == "__main__":
    main()