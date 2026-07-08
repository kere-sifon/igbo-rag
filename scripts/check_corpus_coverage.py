"""
Corpus coverage check: do the canonical greetings actually exist cleanly
in the curated corpus, or in the raw NLLB source?

Run from your project root (wherever data/ and the NLLB jsonl live):

    python check_corpus_coverage.py

Checks both:
  1. The curated FAISS metadata (data/igbo_metadata.json) — what actually
     made it into the index build_faiss_index.py produced.
  2. The raw NLLB source jsonl — whether the phrase exists upstream at all,
     even if the quality filter dropped it.

Adjust METADATA_PATH / NLLB_PATH below to match your actual paths (same
values as FAISS_META_PATH / the argv you pass to build_faiss_index.py).
"""

import json
import os
import unicodedata

METADATA_PATH = os.environ.get("FAISS_META_PATH", "data/igbo_metadata.json")
NLLB_PATH = os.environ.get("NLLB_PATH", "data/nllb_train.jsonl")

# The exact canonical phrases used in the eval set / probe query.
CANONICAL_PHRASES = [
    ("Ụtụtụ ọma", "Good morning"),
    ("A hụrụ m gị n'anya", "I love you"),
    ("Kedu ka i mere?", "How are you?"),
    ("Biko nọdụ ala", "Please sit down"),
    ("Daalụ", "Thank you"),
]


def normalize(s: str) -> str:
    """Strip diacritics-insensitive-ish normalization for loose matching."""
    return unicodedata.normalize("NFKC", s).strip().lower()


def check_metadata():
    if not os.path.exists(METADATA_PATH):
        print(f"[metadata] Not found at {METADATA_PATH} — set FAISS_META_PATH env var.")
        return
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
    # metadata is typically a list of {"input":..., "output":...} aligned to FAISS ids
    pairs = meta if isinstance(meta, list) else meta.get("pairs", meta)
    print(f"[metadata] Loaded {len(pairs)} entries from {METADATA_PATH}\n")

    for igbo, english in CANONICAL_PHRASES:
        exact = [p for p in pairs if normalize(p.get("input", "")) == normalize(igbo)
                 or normalize(p.get("output", "")) == normalize(igbo)]
        near = [p for p in pairs if normalize(igbo) in normalize(p.get("input", "") + " " + p.get("output", ""))]
        print(f"'{igbo}' ({english}):")
        print(f"  exact field match: {len(exact)}")
        print(f"  substring/near match: {len(near)}")
        if exact:
            print(f"  sample: {exact[0]}")
        elif near:
            print(f"  sample near-match: {near[0]}")
        print()


def check_raw_nllb():
    if not os.path.exists(NLLB_PATH):
        print(f"[nllb] Not found at {NLLB_PATH} — set NLLB_PATH env var.")
        return
    counts = {igbo: 0 for igbo, _ in CANONICAL_PHRASES}
    samples = {igbo: None for igbo, _ in CANONICAL_PHRASES}
    normalized_targets = {igbo: normalize(igbo) for igbo, _ in CANONICAL_PHRASES}

    with open(NLLB_PATH, "r", encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            text_blob = normalize(json.dumps(row, ensure_ascii=False))
            for igbo, target in normalized_targets.items():
                if target in text_blob:
                    counts[igbo] += 1
                    if samples[igbo] is None:
                        samples[igbo] = row

    print(f"[nllb raw source] scanned {NLLB_PATH}\n")
    for igbo, english in CANONICAL_PHRASES:
        print(f"'{igbo}' ({english}): {counts[igbo]} occurrences in raw source")
        if samples[igbo]:
            print(f"  sample row: {samples[igbo]}")
        print()


if __name__ == "__main__":
    print("=" * 60)
    print("CURATED METADATA (what's actually in the FAISS index)")
    print("=" * 60)
    check_metadata()

    print("=" * 60)
    print("RAW NLLB SOURCE (before quality filtering)")
    print("=" * 60)
    check_raw_nllb()