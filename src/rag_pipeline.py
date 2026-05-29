import os
from dotenv import load_dotenv
import chromadb
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "/Users/kere/igbo_vector_db")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-r1:14b")


def get_chroma_client():
    return chromadb.PersistentClient(path=CHROMA_DB_PATH)


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

    Args:
        query: The text to find similar translations for
        direction: 'igbo_to_en', 'en_to_igbo', or None for both
        n_results: Number of pairs to return after filtering
        distance_threshold: Maximum distance score to accept (lower = stricter)
    """
    client = get_chroma_client()
    col = client.get_collection("igbo_translations")

    where_filter = {"direction": direction} if direction else None

    # Fetch more than needed so we can filter by quality
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

    # Sort by distance and keep top n_results
    pairs = sorted(pairs, key=lambda x: x["distance"])[:n_results]

    # Fallback: if nothing passed the threshold, return top 3 unfiltered
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
    Assess the quality of retrieved pairs based on best distance score.

    Thresholds:
        < 0.55  -> high   (strong semantic match, corpus is reliable)
        < 0.70  -> medium (reasonable match, use with some caution)
        >= 0.70 -> low    (weak match, rely on model knowledge instead)
        empty   -> no_matches

    Returns: 'high', 'medium', 'low', or 'no_matches'
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


def build_prompt(retrieval_quality: str) -> ChatPromptTemplate:
    """
    Return a quality-aware prompt template.
    LOW quality uses a strict fallback prompt that instructs the model
    to ignore noisy corpus examples and rely on its own Igbo knowledge.
    HIGH/MEDIUM quality uses a grounding prompt that stays close to corpus.
    """
    if retrieval_quality in ("low", "no_matches"):
        system_msg = """You are an expert Igbo-English translator with deep knowledge of
formal Igbo as spoken in southeastern Nigeria (Owerri, Onitsha, Enugu dialects).

IMPORTANT: The corpus examples provided are LOW QUALITY -- they are noisy,
contain transliterated English, or are not genuine Igbo translations.
DO NOT use these examples as a basis for your translation.
Instead, rely entirely on your own linguistic knowledge of formal Igbo.

Rules:
- Use standard formal Igbo only
- Never use transliterated English as a substitute for real Igbo words
- If you are genuinely uncertain, say confidence is low and explain why
- Common reference phrases:
    Daalụ / Imeela        = Thank you
    A hụrụ m gị n'anya    = I love you
    Aha m bụ [name]       = My name is [name]
    Biko                  = Please
    Kedu                  = How / Where are you
    Ụtụtụ ọma             = Good morning
    Ehihie ọma            = Good afternoon
    Anyasị ọma            = Good evening"""
    else:
        system_msg = """You are an expert Igbo-English translator with deep knowledge of
formal and everyday Igbo as spoken in southeastern Nigeria.

You will be given translation examples retrieved from a large verified corpus.
Some examples may be noisy (internet slang, mixed languages) -- if so, note this
and rely on your own linguistic knowledge instead.

Rules:
- Always provide a translation
- Never invent Igbo words -- if unsure, say so explicitly
- Prefer formal Igbo over transliterated English
- If corpus provides strong semantic matches, stay close to them"""

    return ChatPromptTemplate.from_messages([
        ("system", system_msg),
        ("human", """Corpus examples (retrieved by semantic similarity):

{context}

Task: Translate the following
Text: {query}

Respond with:
1. Translation
2. Confidence level (high / medium / low)
3. Usage notes (corpus quality, dialect notes, or alternatives if relevant)""")
    ])


def translate(query: str, direction: str = None) -> dict:
    """
    Main translation function.

    Retrieves relevant corpus pairs from ChromaDB, assesses retrieval quality,
    selects an appropriate prompt path (grounded vs fallback), and invokes
    DeepSeek via Ollama to produce a grounded translation with citations.

    Args:
        query: Text to translate
        direction: 'igbo_to_en', 'en_to_igbo', or None

    Returns:
        dict with keys:
            query             - original query text
            direction         - translation direction used
            response          - LLM translation response (string)
            retrieval_quality - 'high', 'medium', 'low', or 'no_matches'
            citations         - list of retrieved corpus pairs with distances
    """
    pairs = retrieve_translation_pairs(
        query,
        direction,
        n_results=5,
        distance_threshold=0.70
    )

    retrieval_quality = assess_retrieval_quality(pairs)
    context = format_context(pairs) if pairs else "No close matches found in corpus."

    llm = ChatOllama(
        model=LLM_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0.1
    )

    prompt = build_prompt(retrieval_quality)
    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({"context": context, "query": query})

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
        # HIGH retrieval expected -- strong corpus matches
        ("Ụtụtụ ọma", "igbo_to_en"),
        ("Kedu ka i mere?", "igbo_to_en"),
        ("Please sit down", "en_to_igbo"),
        # MEDIUM retrieval expected
        ("Good morning, how are you?", "en_to_igbo"),
        ("Where is the market?", "en_to_igbo"),
        # LOW retrieval -- model should fall back on own Igbo knowledge
        ("I love you", "en_to_igbo"),
        ("Thank you very much", "en_to_igbo"),
        ("My name is Kere", "en_to_igbo"),
        # Bonus -- verify reverse direction works
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
            print(
                f"  [{c['distance']:.4f}] \"{c['input']}\" --> \"{c['output']}\""
            )