import os
import json
import requests as http_requests
from dotenv import load_dotenv
import chromadb
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "/Users/kere/igbo_vector_db")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "Llama3.1:8b")

# Initialise ChromaDB once at module load — reused across all requests
_chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
_chroma_collection = _chroma_client.get_collection("igbo_translations")


def get_chroma_collection():
    return _chroma_collection


def call_ollama(system_msg: str, user_msg: str, timeout: int = 30) -> str:
    """
    Call Ollama directly via HTTP — bypasses LangChain for reliability.
    Uses /api/chat with stream=false and hard timeout.
    """
    payload = {
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
    }
    response = http_requests.post(
        f"{OLLAMA_BASE_URL}/api/chat",
        json=payload,
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()["message"]["content"]


def retrieve_translation_pairs(
    query: str,
    direction: str = None,
    n_results: int = 5,
    distance_threshold: float = 0.70
):
    """
    Retrieve the most relevant translation pairs for a given query.
    Fetches 4x n_results then filters by distance_threshold to remove noisy pairs.
    Falls back to top 3 unfiltered if nothing passes the threshold.
    """
    col = get_chroma_collection()
    where_filter = {"direction": direction} if direction else None

    raw_results = col.query(
        query_texts=[query],
        n_results=n_results * 4,
        where=where_filter,
        include=["documents", "metadatas", "distances"]
    )

    pairs = []
    for i in range(len(raw_results["ids"][0])):
        distance = raw_results["distances"][0][i]
        if distance <= distance_threshold:
            pairs.append({
                "input": raw_results["metadatas"][0][i]["input"],
                "output": raw_results["metadatas"][0][i]["output"],
                "direction": raw_results["metadatas"][0][i]["direction"],
                "distance": distance
            })

    pairs = sorted(pairs, key=lambda x: x["distance"])[:n_results]

    if not pairs:
        for i in range(min(3, len(raw_results["ids"][0]))):
            pairs.append({
                "input": raw_results["metadatas"][0][i]["input"],
                "output": raw_results["metadatas"][0][i]["output"],
                "direction": raw_results["metadatas"][0][i]["direction"],
                "distance": raw_results["distances"][0][i]
            })

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
    Thresholds:
        < 0.55  -> high
        < 0.70  -> medium
        >= 0.70 -> low
        empty   -> no_matches
    """
    if not pairs:
        return "no_matches"
    best_distance = pairs[0]["distance"]
    if best_distance < 0.55:
        return "high"
    elif best_distance < 0.70:
        return "medium"
    else:
        return "low"


def build_messages(retrieval_quality: str, context: str, query: str):
    """Return (system_msg, user_msg) tuple based on retrieval quality."""
    if retrieval_quality in ("low", "no_matches"):
        system_msg = """You are an expert Igbo-English translator with deep knowledge of
formal Igbo as spoken in southeastern Nigeria (Owerri, Onitsha, Enugu dialects).

IMPORTANT: The corpus examples are LOW QUALITY -- noisy or not genuine Igbo.
DO NOT use them. Rely on your own linguistic knowledge of formal Igbo.

Rules:
- Use standard formal Igbo only
- Never use transliterated English as Igbo
- Keep response concise: translation, confidence, one-line note
- Common phrases:
    Daalụ / Imeela     = Thank you
    A hụrụ m gị n'anya = I love you
    Aha m bụ [name]    = My name is [name]
    Biko               = Please
    Ụtụtụ ọma          = Good morning
    Ehihie ọma         = Good afternoon
    Anyasị ọma         = Good evening"""
    else:
        system_msg = """You are an expert Igbo-English translator.
Use the corpus examples to ground your translation.
If examples are noisy, use your own knowledge instead.
Keep response concise: translation, confidence, one-line note."""

    user_msg = f"""Corpus examples:
{context}

Translate: {query}

Reply with:
1. Translation
2. Confidence (high/medium/low)
3. Note (one sentence)"""

    return system_msg, user_msg


def translate(query: str, direction: str = None) -> dict:
    """
    Main translation function. Calls Ollama directly via HTTP.
    """
    pairs = retrieve_translation_pairs(
        query, direction, n_results=5, distance_threshold=0.70
    )

    retrieval_quality = assess_retrieval_quality(pairs)
    context = format_context(pairs) if pairs else "No close matches found."

    system_msg, user_msg = build_messages(retrieval_quality, context, query)

    response = call_ollama(system_msg, user_msg, timeout=30)

    return {
        "query": query,
        "direction": direction,
        "response": response,
        "retrieval_quality": retrieval_quality,
        "citations": pairs
    }


if __name__ == "__main__":
    print("Igbo-English RAG Translation Pipeline")
    print(f"Model    : {LLM_MODEL}")
    print(f"Chroma DB: {CHROMA_DB_PATH}")
    print("=" * 60)

    test_queries = [
        ("Ụtụtụ ọma", "igbo_to_en"),
        ("I love you", "en_to_igbo"),
        ("Thank you very much", "en_to_igbo"),
        ("A hụrụ m gị n'anya", "igbo_to_en"),
    ]

    for query, direction in test_queries:
        print(f"\n{'='*60}")
        result = translate(query, direction=direction)
        print(f"Query     : {result['query']} [{direction}]")
        print(f"Retrieval : {result['retrieval_quality'].upper()}")
        print(f"\nResponse:\n{result['response']}")
        print(f"\nCitations ({len(result['citations'])} pairs):")
        for c in result["citations"]:
            print(f"  [{c['distance']:.4f}] \"{c['input']}\" --> \"{c['output']}\"")