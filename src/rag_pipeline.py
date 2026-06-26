import os
import json
import time
import urllib.request
import numpy as np
import faiss
from dotenv import load_dotenv

load_dotenv()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "Qwen2.5:7B")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH",
    os.path.expanduser("~/projects/igbo-rag/data/igbo_faiss.index"))
FAISS_META_PATH = os.getenv("FAISS_META_PATH",
    os.path.expanduser("~/projects/igbo-rag/data/igbo_metadata.json"))

# Kept for optional JSONL export / backup — no longer the primary store.
CORRECTIONS_PATH = os.getenv("CORRECTIONS_PATH",
    os.path.expanduser("~/projects/igbo-rag/data/corrections.jsonl"))

# --- Initialise FAISS index + metadata once at module load ---
print(f"Loading FAISS index from {FAISS_INDEX_PATH}...")
_faiss_index = faiss.read_index(FAISS_INDEX_PATH)
with open(FAISS_META_PATH, "r", encoding="utf-8") as f:
    _metadata = json.load(f)
print(f"FAISS index ready: {_faiss_index.ntotal:,} vectors")

# --- Corrections store — hot path dict, backed by MongoDB ---
_corrections: dict = {}  # key: "query||direction", value: correct_translation

# Baseline corrections baked in at install time; user feedback overrides these.
_DEFAULT_CORRECTIONS: dict = {
    "please sit down||en_to_igbo":  "Biko nọdụ ala",
    "biko nọdụ ala||igbo_to_en":    "Please sit down",
    "i miss you||en_to_igbo":       "A chefuo m gị",
    "my name is||en_to_igbo":       "Aha m bụ",
    "come and eat||en_to_igbo":     "Bịa rie nri",
}


def load_corrections(path: str = None):
    """
    Rebuild the in-memory _corrections dict from MongoDB.

    MongoDB is the primary source of truth.  The `path` argument is
    accepted for backwards-compatibility but is no longer used.

    Falls back to an empty dict (plus _DEFAULT_CORRECTIONS) if MongoDB
    is unavailable — the API stays up, corrections just won't persist
    until the connection is restored.
    """
    global _corrections
    try:
        from db import load_all_corrections
        rows = load_all_corrections()
        new_corrections = {}
        for c in rows:
            try:
                key = f"{c['query'].lower().strip()}||{c['direction']}"
                new_corrections[key] = c["correct_translation"]
            except KeyError:
                continue
        _corrections = new_corrections
        print(f"Loaded {len(_corrections)} corrections from MongoDB")
    except Exception as exc:
        print(f"WARNING: could not load corrections from MongoDB ({exc}). "
              "Using defaults only.")
        _corrections = {}


# Load corrections at startup
load_corrections()


def check_correction(query: str, direction: str) -> str | None:
    """Check if a correction exists for this exact query + direction."""
    key = f"{query.lower().strip()}||{direction}"
    return _corrections.get(key)


def embed_query(text: str) -> np.ndarray:
    """Embed a query string using nomic-embed-text via Ollama."""
    payload = json.dumps({
        "model": EMBED_MODEL,
        "prompt": text
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{OLLAMA_BASE_URL}/api/embeddings",
        data=payload,
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        result = json.loads(resp.read())
        vec = np.array([result["embedding"]], dtype=np.float32)
        faiss.normalize_L2(vec)
        return vec


def retrieve_translation_pairs(
    query: str,
    direction: str = None,
    n_results: int = 5,
    similarity_threshold: float = 0.70
):
    """Retrieve the most relevant translation pairs using FAISS.

    FAISS IndexFlatIP on L2-normalised vectors returns cosine similarity
    scores in [0, 1] — higher means more similar.
    """
    fetch_k = n_results * 8 if direction else n_results * 4
    fetch_k = min(fetch_k, _faiss_index.ntotal)

    vec = embed_query(query)
    scores, indices = _faiss_index.search(vec, fetch_k)

    pairs = []
    for sim, idx in zip(scores[0], indices[0]):
        if idx < 0 or idx >= len(_metadata):
            continue
        m = _metadata[idx]
        if direction and m.get("direction") != direction:
            continue
        if sim < similarity_threshold:
            continue
        pairs.append({
            "input": m["input"],
            "output": m["output"],
            "direction": m["direction"],
            "similarity": float(sim)
        })
        if len(pairs) >= n_results:
            break

    if not pairs:
        for sim, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(_metadata):
                continue
            m = _metadata[idx]
            if direction and m.get("direction") != direction:
                continue
            pairs.append({
                "input": m["input"],
                "output": m["output"],
                "direction": m["direction"],
                "similarity": float(sim)
            })
            if len(pairs) >= min(3, n_results):
                break

    return pairs


def format_context(pairs: list) -> str:
    lines = []
    for i, p in enumerate(pairs, 1):
        lines.append(
            f"{i}. [{p['direction']}] \"{p['input']}\" → \"{p['output']}\""
        )
    return "\n".join(lines)


def assess_retrieval_quality(pairs: list) -> str:
    if not pairs:
        return "no_matches"
    best_similarity = pairs[0]["similarity"]
    if best_similarity > 0.92:
        return "high"
    elif best_similarity > 0.90:
        return "medium"
    else:
        return "low"


def format_corrections_for_prompt() -> str:
    """Build the CRITICAL CORRECTIONS block from live store, falling back to defaults."""
    merged = {**_DEFAULT_CORRECTIONS, **_corrections}
    if not merged:
        return ""
    lines = ["CRITICAL CORRECTIONS — always use these exact translations, no exceptions:"]
    for key, translation in merged.items():
        query, _ = key.rsplit("||", 1)
        lines.append(f'    "{query}" = "{translation}"')
    return "\n".join(lines)


def build_messages(retrieval_quality: str, context: str, query: str):
    corrections_block = format_corrections_for_prompt()

    if retrieval_quality in ("low", "no_matches"):
        system_msg = f"""You are an expert Igbo-English translator with deep knowledge of
formal Igbo as spoken in southeastern Nigeria (Owerri, Onitsha, Enugu dialects).

IMPORTANT: The corpus examples are LOW QUALITY or not relevant.
DO NOT use them. Rely on your own linguistic knowledge of formal Igbo.

Rules:
- Use standard formal Igbo only
- Never use transliterated English as Igbo
- Keep response concise: translation, confidence, one-line note

{corrections_block}

Common reference phrases:
    Daalụ / Imeela          = Thank you
    Daalụ nke ukwuu         = Thank you very much (emphatic)
    A hụrụ m gị n'anya      = I love you
    Aha m bụ [name]         = My name is [name]
    Biko                    = Please
    Ụtụtụ ọma               = Good morning (NOT good luck)
    Ehihie ọma              = Good afternoon
    Anyasị ọma              = Good evening
    Kedu                    = How are you? (NOT I am fine)
    Kedu ka ị mere?         = How are you? (formal)
    Ọ dị mma                = I am fine / It is good
    Nnọọ                    = Welcome
    Njem ọma                = Safe journey
    Chukwu gozie gị         = God bless you
    A chefuo m gị           = I miss you
    Amara m                 = Congratulations"""
    else:
        system_msg = f"""You are an expert Igbo-English translator.
Use the corpus examples to ground your translation.
If examples are noisy, use your own knowledge instead.
Keep response concise: translation, confidence, one-line note.

{corrections_block}"""

    user_msg = f"""Corpus examples:
{context}

Translate: {query}

Reply with:
1. Translation
2. Confidence (high/medium/low)
3. Note (one sentence)"""

    return system_msg, user_msg


def call_ollama(system_msg: str, user_msg: str, timeout: int = 60) -> str:
    payload = json.dumps({
        "model": LLM_MODEL,
        "stream": False,
        "options": {
            "temperature": 0.1,
            "num_predict": 200,
        },
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
    }).encode("utf-8")

    req = urllib.request.Request(
        f"{OLLAMA_BASE_URL}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())["message"]["content"]


def translate(query: str, direction: str = None) -> dict:
    """
    Main translation function.
    1. Check corrections store first — if exact match found, return immediately
    2. Otherwise retrieve from FAISS + generate via Ollama
    """
    # --- Correction lookup (highest priority) ---
    directions_to_check = [direction] if direction else ["en_to_igbo", "igbo_to_en"]
    for d in directions_to_check:
        correction = check_correction(query, d)
        if correction:
            return {
                "query": query,
                "direction": d,
                "response": correction,  # clean translation only, no numbered format
                "retrieval_quality": "correction",
                "citations": []
            }

    # --- Normal RAG pipeline ---
    pairs = retrieve_translation_pairs(
        query, direction, n_results=5, similarity_threshold=0.70
    )

    retrieval_quality = assess_retrieval_quality(pairs)
    context = format_context(pairs) if pairs else "No close matches found."

    system_msg, user_msg = build_messages(retrieval_quality, context, query)
    response = call_ollama(system_msg, user_msg, timeout=60)

    return {
        "query": query,
        "direction": direction,
        "response": response,
        "retrieval_quality": retrieval_quality,
        "citations": pairs
    }


if __name__ == "__main__":
    print(f"Model    : {LLM_MODEL}")
    print(f"FAISS    : {FAISS_INDEX_PATH}")
    print(f"Corrections: {len(_corrections)}")
    print("=" * 60)

    test_queries = [
        ("Ụtụtụ ọma", "igbo_to_en"),
        ("I love you", "en_to_igbo"),
        ("Thank you very much", "en_to_igbo"),
        ("Please sit down", "en_to_igbo"),
        ("Biko nọdụ ala", "igbo_to_en"),
        ("I miss you", "en_to_igbo"),
        ("Come and eat", "en_to_igbo"),
    ]

    for query, direction in test_queries:
        print(f"\n{'='*60}")
        start = time.time()
        result = translate(query, direction=direction)
        elapsed = time.time() - start
        print(f"Query     : {result['query']} [{direction}]")
        print(f"Retrieval : {result['retrieval_quality'].upper()}")
        print(f"Total time: {elapsed:.1f}s")
        print(f"Response  : {result['response'].split(chr(10))[0]}")
