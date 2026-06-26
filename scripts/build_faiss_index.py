"""
build_faiss_index.py

Builds a FAISS index from nllb_train.jsonl:
1. Loads and filters clean translation pairs
2. Embeds with nomic-embed-text via Ollama
3. Saves FAISS index + metadata to ~/projects/igbo-rag/data/

Usage:
    python scripts/build_faiss_index.py /path/to/nllb_train.jsonl

    The JSONL path can also be set via the JSONL_PATH environment variable.

Outputs:
    data/igbo_faiss.index        - FAISS index
    data/igbo_metadata.json      - metadata (input/output/direction) per vector
    data/igbo_embeddings.npy     - raw embeddings checkpoint (deleted after index built)
"""

import json
import os
import sys
import time
import urllib.request
import numpy as np

# --- Config ---
JSONL_PATH = (
    sys.argv[1]
    if len(sys.argv) > 1
    else os.getenv("JSONL_PATH")
)
OUTPUT_DIR = os.getenv("OUTPUT_DIR", os.path.expanduser("~/projects/igbo-rag/data"))
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/embeddings")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
TARGET_PAIRS = int(os.getenv("TARGET_PAIRS", "1000000"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "50"))


# Quality filter
def is_clean(record: dict) -> bool:
    inp = record.get("input", "").strip()
    out = record.get("output", "").strip()
    instruction = record.get("instruction", "")

    # Only keep translate instructions
    if "Translate" not in instruction:
        return False
    # Skip very short outputs
    if len(out) < 4 or len(inp) < 4:
        return False
    # Skip obvious social media noise
    if any(w in out.lower() for w in ["ya nice", "mmmmm", "lol", "haha", "wtf", "omg"]):
        return False
    # Skip all-caps noise
    if out == out.upper() and len(out) > 5 and out.isalpha():
        return False
    # For en_to_igbo ONLY: skip pairs where Igbo output has no unicode
    # (igbo_to_en pairs have English output which legitimately has no unicode)
    direction = "en_to_igbo" if "English sentence to Igbo" in instruction else "igbo_to_en"
    if direction == "en_to_igbo" and all(ord(c) < 128 for c in out):
        return False

    return True


def get_direction(instruction: str) -> str:
    if "English sentence to Igbo" in instruction:
        return "en_to_igbo"
    return "igbo_to_en"


def embed_batch(texts: list) -> list:
    embeddings = []
    for text in texts:
        payload = json.dumps({
            "model": EMBED_MODEL,
            "prompt": text
        }).encode("utf-8")
        req = urllib.request.Request(
            OLLAMA_URL,
            data=payload,
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read())
            embeddings.append(result["embedding"])
    return embeddings


def detect_embed_dim() -> int:
    """Detect actual embedding dimension from Ollama."""
    payload = json.dumps({
        "model": EMBED_MODEL,
        "prompt": "test"
    }).encode("utf-8")
    req = urllib.request.Request(
        OLLAMA_URL,
        data=payload,
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        result = json.loads(resp.read())
        return len(result["embedding"])


def main():
    if not JSONL_PATH:
        print("Error: provide the JSONL path as an argument or set JSONL_PATH env var.")
        print("  Usage: python scripts/build_faiss_index.py /path/to/nllb_train.jsonl")
        sys.exit(1)

    try:
        import faiss
    except ImportError:
        print("Installing faiss-cpu...")
        os.system("pip install faiss-cpu")
        import faiss

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- Detect embedding dimension ---
    print("Detecting embedding dimension...")
    embed_dim = detect_embed_dim()
    print(f"Embedding dim: {embed_dim}")
    print()

    # --- Step 1: Load and filter pairs ---
    print(f"Loading clean pairs from {JSONL_PATH}")
    print(f"Target: {TARGET_PAIRS:,} pairs")
    print("=" * 60)

    pairs = []
    skipped = 0
    with open(JSONL_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if len(pairs) >= TARGET_PAIRS:
                break
            try:
                record = json.loads(line.strip())
            except json.JSONDecodeError:
                continue
            if is_clean(record):
                pairs.append({
                    "input": record["input"].strip(),
                    "output": record["output"].strip(),
                    "direction": get_direction(record["instruction"])
                })
            else:
                skipped += 1

    print(f"Loaded  : {len(pairs):,} clean pairs")
    print(f"Skipped : {skipped:,} noisy pairs")
    print()

    # --- Step 2: Embed in batches ---
    # Three checkpoint files kept in sync:
    #   igbo_embeddings.npy          — raw embedding vectors
    #   igbo_metadata_checkpoint.json — pairs that were successfully embedded
    #   igbo_checkpoint_state.json   — next source index to process (so failed
    #                                  batches don't cause pairs/embeddings to
    #                                  drift out of sync on resume)
    checkpoint_path = os.path.join(OUTPUT_DIR, "igbo_embeddings.npy")
    meta_checkpoint_path = os.path.join(OUTPUT_DIR, "igbo_metadata_checkpoint.json")
    state_checkpoint_path = os.path.join(OUTPUT_DIR, "igbo_checkpoint_state.json")

    all_embeddings = []
    embedded_pairs = []  # always parallel to all_embeddings
    next_idx = 0         # position in the source `pairs` list

    if (os.path.exists(checkpoint_path)
            and os.path.exists(meta_checkpoint_path)
            and os.path.exists(state_checkpoint_path)):
        print(f"Checkpoint found — resuming from {checkpoint_path}")
        all_embeddings = np.load(checkpoint_path).tolist()
        with open(meta_checkpoint_path, "r") as f:
            embedded_pairs = json.load(f)
        with open(state_checkpoint_path, "r") as f:
            next_idx = json.load(f)["next_idx"]
        print(f"Resumed: {len(all_embeddings):,} embeddings, next source idx {next_idx:,}")

    def _save_checkpoint(next_source_idx: int) -> None:
        np.save(checkpoint_path, np.array(all_embeddings, dtype=np.float32))
        with open(meta_checkpoint_path, "w") as f:
            json.dump(embedded_pairs, f, ensure_ascii=False)
        with open(state_checkpoint_path, "w") as f:
            json.dump({"next_idx": next_source_idx}, f)

    if next_idx < len(pairs):
        print(f"Embedding {len(pairs) - next_idx:,} remaining pairs with {EMBED_MODEL}...")
        start = time.time()
        embedded_at_start = len(all_embeddings)

        for i in range(next_idx, len(pairs), BATCH_SIZE):
            batch_pairs = pairs[i:i + BATCH_SIZE]
            texts = [f"{p['input']} | {p['output']}" for p in batch_pairs]
            try:
                embeddings = embed_batch(texts)
                all_embeddings.extend(embeddings)
                embedded_pairs.extend(batch_pairs)
            except Exception as e:
                print(f"  Batch {i//BATCH_SIZE} failed: {e} — skipping")

            done = len(all_embeddings)
            if (i // BATCH_SIZE) % 20 == 0:
                elapsed = time.time() - start
                rate = (done - embedded_at_start) / elapsed if elapsed > 0 else 0
                remaining = len(pairs) - (i + BATCH_SIZE)
                eta = remaining / rate if rate > 0 else 0
                print(f"  [{done:,} embedded / {i + BATCH_SIZE:,} processed / {len(pairs):,} total] "
                      f"{elapsed:.0f}s | {rate:.0f} pairs/s | ETA {eta/60:.1f} min")

            if done % 5000 == 0 and done > 0:
                _save_checkpoint(i + BATCH_SIZE)

        print(f"\nEmbedding complete: {len(all_embeddings):,} vectors in "
              f"{time.time()-start:.0f}s")
        _save_checkpoint(len(pairs))

    # --- Step 3: Build FAISS index ---
    print("\nBuilding FAISS index...")
    embeddings_np = np.array(all_embeddings, dtype=np.float32)
    print(f"Embeddings shape: {embeddings_np.shape}")
    print(f"Using detected dim: {embed_dim}")

    actual_dim = embeddings_np.shape[1]
    if actual_dim != embed_dim:
        print(f"Warning: detected dim {embed_dim} != actual dim {actual_dim}, using actual")
        embed_dim = actual_dim

    # Normalise for cosine similarity
    faiss.normalize_L2(embeddings_np)

    # IndexFlatIP = inner product on normalised vectors = cosine similarity
    index = faiss.IndexFlatIP(embed_dim)
    index.add(embeddings_np)
    print(f"Index built: {index.ntotal:,} vectors, {embed_dim}-dim")

    # --- Step 4: Save ---
    index_path = os.path.join(OUTPUT_DIR, "igbo_faiss.index")
    meta_path = os.path.join(OUTPUT_DIR, "igbo_metadata.json")

    faiss.write_index(index, index_path)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(embedded_pairs, f, ensure_ascii=False)

    index_size_mb = os.path.getsize(index_path) / (1024**2)
    meta_size_mb = os.path.getsize(meta_path) / (1024**2)

    # Clean up checkpoints
    for cp in (checkpoint_path, meta_checkpoint_path, state_checkpoint_path):
        if os.path.exists(cp):
            os.remove(cp)

    print(f"\nSaved:")
    print(f"  FAISS index : {index_path} ({index_size_mb:.1f} MB)")
    print(f"  Metadata    : {meta_path} ({meta_size_mb:.1f} MB)")
    print(f"\nDone. {index.ntotal:,} vectors ready for sub-millisecond retrieval.")


if __name__ == "__main__":
    main()