"""
weaviate_setup.py

Create (or recreate) the Weaviate collection used by the igbo-rag benchmark.

Run after starting Weaviate locally:

    docker run -p 8080:8080 -p 50051:50051 \
        semitechnologies/weaviate:latest

Then:

    python scripts/weaviate_setup.py

This script is idempotent: if the collection already exists it is deleted and
re-created so the schema is guaranteed to match what the benchmark code expects.
"""

import os
import sys

_src = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _src not in sys.path:
    sys.path.insert(0, _src)

from dotenv import load_dotenv

load_dotenv()

import weaviate
from weaviate.classes.config import Configure, DataType, Property, VectorDistances

WEAVIATE_HOST = os.getenv("WEAVIATE_HOST", "localhost")
WEAVIATE_HTTP_PORT = int(os.getenv("WEAVIATE_HTTP_PORT", "8081"))
WEAVIATE_GRPC_PORT = int(os.getenv("WEAVIATE_GRPC_PORT", "50052"))
COLLECTION_NAME = os.getenv("WEAVIATE_COLLECTION", "TranslationPair")


def main() -> None:
    client = weaviate.connect_to_custom(
        http_host=WEAVIATE_HOST,
        http_port=WEAVIATE_HTTP_PORT,
        http_secure=False,
        grpc_host=WEAVIATE_HOST,
        grpc_port=WEAVIATE_GRPC_PORT,
        grpc_secure=False,
    )

    try:
        if client.collections.exists(COLLECTION_NAME):
            print(f"Deleting existing collection: {COLLECTION_NAME}")
            client.collections.delete(COLLECTION_NAME)

        print(f"Creating collection: {COLLECTION_NAME}")
        collection = client.collections.create(
            name=COLLECTION_NAME,
            vectorizer_config=Configure.Vectorizer.none(),
            properties=[
                Property(name="input", data_type=DataType.TEXT),
                Property(name="output", data_type=DataType.TEXT),
                Property(name="direction", data_type=DataType.TEXT),
                Property(name="source", data_type=DataType.TEXT),
                Property(name="reingested_at", data_type=DataType.TEXT),
            ],
            vector_index_config=Configure.VectorIndex.hnsw(
                distance_metric=VectorDistances.COSINE
            ),
        )

        print(f"Collection created: {collection.name}")
        print("Schema ready for ingestion.")
    finally:
        client.close()


if __name__ == "__main__":
    main()
