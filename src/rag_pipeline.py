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

# --- Initialise FAISS index + metadata once at module load ---
print(f"Loading FAISS index from {FAISS_INDEX_PATH}...")
_faiss_index = faiss.read_index(FAISS_INDEX_PATH)
with open(FAISS_META_PATH, "r", encoding="utf-8") as f:
    _metadata = json.load(f)
print(f"FAISS index ready: {_faiss_index.ntotal:,} vectors")


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
    distance_threshold: float = 0.70
):
    """
    Retrieve the most relevant translation pairs using FAISS.
    Sub-millisecond search over 1M curated pairs.

    Args:
        query: Text to find similar translations for
        direction: 'igbo_to_en', 'en_to_igbo', or None for both
        n_results: Number of pairs to return
        distance_threshold: Minimum cosine similarity to accept (higher = stricter)
    """
    fetch_k = n_results * 8 if direction else n_results * 4
    fetch_k = min(fetch_k, _faiss_index.ntotal)

    vec = embed_query(query)
    distances, indices = _faiss_index.search(vec, fetch_k)

    pairs = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx < 0 or idx >= len(_metadata):
            continue
        m = _metadata[idx]
        if direction and m.get("direction") != direction:
            continue
        if dist < distance_threshold:
            continue
        pairs.append({
            "input": m["input"],
            "output": m["output"],
            "direction": m["direction"],
            "distance": float(1.0 - dist)
        })
        if len(pairs) >= n_results:
            break

    # Fallback: if nothing passed threshold, return top n_results unfiltered
    if not pairs:
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(_metadata):
                continue
            m = _metadata[idx]
            if direction and m.get("direction") != direction:
                continue
            pairs.append({
                "input": m["input"],
                "output": m["output"],
                "direction": m["direction"],
                "distance": float(1.0 - dist)
            })
            if len(pairs) >= min(3, n_results):
                break

    return pairs


def format_context(pairs: list) -> str:
    """Format retrieved pairs as numbered grounding context for the LLM."""
    lines = []
    for i, p in enumerate(pairs, 1):
        lines.append(
            f"{i}. [{p['direction']}] \"{p['input']}\" → \"{p['output']}\""
        )
    return "\n".join(lines)


def assess_retrieval_quality(pairs: list) -> str:
    """
    Assess quality based on best distance score (lower = better match).

    Thresholds:
        < 0.08  -> high   (strong match)
        < 0.10  -> medium (reasonable match)
        >= 0.10 -> low    (weak match, rely on model knowledge)
        empty   -> no_matches
    """
    if not pairs:
        return "no_matches"
    best_distance = pairs[0]["distance"]
    if best_distance < 0.08:
        return "high"
    elif best_distance < 0.10:
        return "medium"
    else:
        return "low"


def build_messages(retrieval_quality: str, context: str, query: str):
    """Return (system_msg, user_msg) tuple based on retrieval quality."""
    if retrieval_quality in ("low", "no_matches"):
        system_msg = """You are an expert Igbo-English translator with deep knowledge of
formal Igbo as spoken in southeastern Nigeria (Owerri, Onitsha, Enugu dialects).

IMPORTANT: The corpus examples are LOW QUALITY or not relevant.
DO NOT use them. Rely on your own linguistic knowledge of formal Igbo.

Rules:
- Use standard formal Igbo only
- Never use transliterated English as Igbo
- Keep response concise: translation, confidence, one-line note

CRITICAL CORRECTIONS — always use these exact translations, no exceptions:
    "Please sit down"       = "Biko nọdụ ala"  (nọdụ = sit, ala = down/ground)
    "Biko nọdụ ala"         = "Please sit down" (NOT calm down, NOT arrived on Earth)
    "I miss you"            = "A chefuo m gị"   (NOT Agụụ which means hunger/appetite)
    "My name is [X]"        = "Aha m bụ [X]"    (m = my/I, NOT ya which means his/her)
    "Come and eat"          = "Bịa rie nri"

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
        system_msg = """You are an expert Igbo-English translator.
Use the corpus examples to ground your translation.
If examples are noisy, use your own knowledge instead.
Keep response concise: translation, confidence, one-line note.

CRITICAL CORRECTIONS — always use these exact translations, no exceptions:
    "Biko nọdụ ala"  = "Please sit down" (NOT arrived on Earth, NOT calm down)
    "My name is [X]" = "Aha m bụ [X]"    (m = my, NOT ya which means his/her)"""

    user_msg = f"""Corpus examples:
{context}

Translate: {query}

Reply with:
1. Translation
2. Confidence (high/medium/low)
3. Note (one sentence)"""

    return system_msg, user_msg


def call_ollama(system_msg: str, user_msg: str, timeout: int = 60) -> str:
    """Call Ollama directly via HTTP."""
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
    Retrieves from FAISS locally (~86ms), generates via Ollama.
    """
    pairs = retrieve_translation_pairs(
        query, direction, n_results=5, distance_threshold=0.70
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
    print("=" * 60)

    test_queries = [
        ("Ụtụtụ ọma", "igbo_to_en"),
        ("I love you", "en_to_igbo"),
        ("Thank you very much", "en_to_igbo"),
        ("A hụrụ m gị n'anya", "igbo_to_en"),
        ("My name is Kere", "en_to_igbo"),
        ("Good morning, how are you?", "en_to_igbo"),
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
        print(f"\nResponse:\n{result['response']}")
        print(f"\nCitations ({len(result['citations'])} pairs):")
        for c in result["citations"]:
            print(f"  [{c['distance']:.4f}] \"{c['input']}\" --> \"{c['output']}\"")