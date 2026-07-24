"""
weaviate_store.py

Weaviate-backed vector store for the igbo-rag benchmark.

This module mirrors the responsibilities of the FAISS + metadata JSON combo in
rag_pipeline.py, but uses a Weaviate collection instead.  It is designed to be
swappable: the rest of the pipeline (prompting, LLM generation, corrections)
remains unchanged.

Environment variables:
    WEAVIATE_HOST       default: localhost
    WEAVIATE_HTTP_PORT  default: 8080
    WEAVIATE_GRPC_PORT  default: 50051
    WEAVIATE_COLLECTION default: TranslationPair
"""

import atexit
import os
import time
from typing import Optional

import numpy as np
import weaviate
from dotenv import load_dotenv
from weaviate.classes.query import Filter, MetadataQuery
from weaviate.config import AdditionalConfig, Timeout

load_dotenv()

from embeddings import embed_query

WEAVIATE_HOST = os.getenv("WEAVIATE_HOST", "localhost")
WEAVIATE_HTTP_PORT = int(os.getenv("WEAVIATE_HTTP_PORT", "8081"))
WEAVIATE_GRPC_PORT = int(os.getenv("WEAVIATE_GRPC_PORT", "50052"))
COLLECTION_NAME = os.getenv("WEAVIATE_COLLECTION", "TranslationPair")

_client: Optional[weaviate.WeaviateClient] = None


def _close_client() -> None:
    global _client
    if _client is not None and _client.is_connected():
        _client.close()
        _client = None


atexit.register(_close_client)


def get_client() -> weaviate.WeaviateClient:
    """Lazy singleton Weaviate client with generous insert timeout."""
    global _client
    if _client is None or not _client.is_connected():
        # Large insert timeout is important for big batch imports over gRPC.
        _client = weaviate.connect_to_custom(
            http_host=WEAVIATE_HOST,
            http_port=WEAVIATE_HTTP_PORT,
            http_secure=False,
            grpc_host=WEAVIATE_HOST,
            grpc_port=WEAVIATE_GRPC_PORT,
            grpc_secure=False,
            additional_config=AdditionalConfig(
                timeout=Timeout(init=10, query=60, insert=300)
            ),
        )
    return _client


def count_pairs() -> int:
    """Return the total number of vectors in the collection."""
    collection = get_client().collections.get(COLLECTION_NAME)
    return collection.aggregate.over_all().total_count


def retrieve_translation_pairs(
    query: str,
    direction: Optional[str] = None,
    n_results: int = 5,
    similarity_threshold: float = 0.70,
    use_hybrid: bool = False,
    hybrid_alpha: float = 0.5,
):
    """
    Retrieve the most relevant translation pairs from Weaviate.

    Args:
        query: source phrase to translate.
        direction: optional "en_to_igbo" or "igbo_to_en" filter.
        n_results: number of pairs to return.
        similarity_threshold: minimum cosine similarity (0-1).  Weaviate returns
            distances; we convert to cosine similarity as 1 - distance.
        use_hybrid: if True, use Weaviate hybrid search (vector + BM25).
        hybrid_alpha: balance between vector (1.0) and keyword (0.0).

    Returns:
        List of dicts: input, output, direction, similarity.
    """
    collection = get_client().collections.get(COLLECTION_NAME)
    vector = embed_query(query)[0].tolist()

    filters = None
    if direction:
        filters = Filter.by_property("direction").equal(direction)

    if use_hybrid:
        response = collection.query.hybrid(
            query=query,
            vector=vector,
            alpha=hybrid_alpha,
            limit=n_results * 4 if direction else n_results * 2,
            filters=filters,
            return_metadata=MetadataQuery(distance=True),
        )
    else:
        response = collection.query.near_vector(
            near_vector=vector,
            limit=n_results * 4 if direction else n_results * 2,
            filters=filters,
            return_metadata=MetadataQuery(distance=True),
        )

    pairs = []
    for obj in response.objects:
        distance = obj.metadata.distance or 0.0
        similarity = 1.0 - distance
        if similarity < similarity_threshold:
            continue
        pairs.append({
            "input": obj.properties["input"],
            "output": obj.properties["output"],
            "direction": obj.properties["direction"],
            "similarity": float(similarity),
        })
        if len(pairs) >= n_results:
            break

    # Fallback: if nothing passes the threshold, return the best available
    # (mirrors the FAISS backend behaviour so comparisons are fair).
    if not pairs:
        for obj in response.objects:
            distance = obj.metadata.distance or 0.0
            similarity = 1.0 - distance
            pairs.append({
                "input": obj.properties["input"],
                "output": obj.properties["output"],
                "direction": obj.properties["direction"],
                "similarity": float(similarity),
            })
            if len(pairs) >= min(3, n_results):
                break

    return pairs


def measure_raw_search_latency(k: int = 5) -> float:
    """Time a single Weaviate near_vector search with a random probe."""
    collection = get_client().collections.get(COLLECTION_NAME)
    dim = 768  # nomic-embed-text
    vec = np.random.rand(dim).astype(np.float32)
    # Normalise so it is a valid cosine probe (Weaviate stores normalised vectors).
    vec = vec / np.linalg.norm(vec)

    start = time.perf_counter()
    collection.query.near_vector(
        near_vector=vec.tolist(),
        limit=k,
        return_metadata=MetadataQuery(distance=True),
    )
    return round((time.perf_counter() - start) * 1000, 2)


def ingest_batch(
    pairs: list[dict],
    embeddings: list[list[float]],
    batch_size: int = 100,
    concurrent_requests: int = 1,
) -> tuple[int, list]:
    """
    Insert a batch of translation pairs with pre-computed vectors.

    Uses fixed-size batches and serialized concurrent_requests to avoid
    overwhelming Weaviate (especially inside a Colima container).  Failed
    objects are returned so the caller can retry.

    Args:
        pairs: list of dicts with keys input, output, direction, source,
               and optionally reingested_at.
        embeddings: parallel list of vectors.
        batch_size: number of objects per Weaviate batch.
        concurrent_requests: number of parallel batch requests.

    Returns:
        (number_of_objects_inserted, list_of_failed_objects)
    """
    collection = get_client().collections.get(COLLECTION_NAME)
    inserted = 0
    failed: list = []
    with collection.batch.fixed_size(
        batch_size=batch_size, concurrent_requests=concurrent_requests
    ) as batch:
        for pair, vector in zip(pairs, embeddings):
            batch.add_object(properties=pair, vector=vector)
            inserted += 1

    # Failed objects are accumulated on the collection-level batch helper.
    failed = list(collection.batch.failed_objects)
    return inserted, failed


def upsert_correction(query: str, correct_translation: str, direction: str) -> None:
    """
    Upsert a single correction pair.

    Weaviate does not have a native upsert by business key, so we delete any
    existing objects with the same (input, direction) and re-insert.
    """
    client = get_client()
    collection = client.collections.get(COLLECTION_NAME)

    # Delete existing matches
    where = (
        Filter.by_property("input").equal(query)
        & Filter.by_property("direction").equal(direction)
    )
    collection.data.delete_many(where=where)

    # Re-embed and insert
    vector = embed_query(query)[0].tolist()
    collection.data.insert(
        properties={
            "input": query,
            "output": correct_translation,
            "direction": direction,
            "source": "correction",
        },
        vector=vector,
    )
