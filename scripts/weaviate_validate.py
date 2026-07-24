"""
weaviate_validate.py

End-to-end validation of the Weaviate benchmark stack using embedded Weaviate.
No Docker required. This script:

1. Starts an embedded Weaviate instance
2. Creates the TranslationPair collection
3. Ingests a small sample of pairs from the existing FAISS index
4. Runs the FAISS vs Weaviate retrieval benchmark
5. Shuts down Weaviate

Run:

    python scripts/weaviate_validate.py --limit 1000

This is intended as a smoke test before running the full benchmark against a
real Weaviate container with the full 1M-pair corpus.
"""

import argparse
import os
import sys

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_src = os.path.join(_project_root, "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

# Point the Weaviate client at the embedded instance ports before any module
# imports its own defaults.
os.environ.setdefault("WEAVIATE_HOST", "localhost")
os.environ.setdefault("WEAVIATE_HTTP_PORT", "8079")
os.environ.setdefault("WEAVIATE_GRPC_PORT", "50050")

import weaviate
from weaviate.classes.config import Configure, DataType, Property, VectorDistances


def create_collection(client):
    """Create the TranslationPair collection on the given client."""
    collection_name = os.getenv("WEAVIATE_COLLECTION", "TranslationPair")
    if client.collections.exists(collection_name):
        print(f"Deleting existing collection: {collection_name}")
        client.collections.delete(collection_name)

    print(f"Creating collection: {collection_name}")
    client.collections.create(
        name=collection_name,
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=1000, help="Pairs to ingest")
    parser.add_argument(
        "--skip-ingest",
        action="store_true",
        help="Skip ingestion and only run benchmark (collection must exist).",
    )
    args = parser.parse_args()

    print("Starting embedded Weaviate...")
    client = weaviate.connect_to_embedded()
    print(f"Embedded Weaviate ready at {client._connection.url}")

    try:
        if not args.skip_ingest:
            print("\n[1/3] Setting up schema...")
            create_collection(client)

            print("\n[2/3] Ingesting sample pairs...")
            import weaviate_ingest
            # Patch argv so the ingest script sees our flags
            sys.argv = [
                "weaviate_ingest.py",
                "--from-faiss",
                "--limit",
                str(args.limit),
            ]
            weaviate_ingest.main()
        else:
            print("\n[1/3] Skipping setup/ingest (--skip-ingest)")

        print("\n[3/3] Running benchmark...")
        import weaviate_benchmark
        sys.argv = ["weaviate_benchmark.py"]
        weaviate_benchmark.main()

    finally:
        print("\nShutting down embedded Weaviate...")
        client.close()
        print("Done.")


if __name__ == "__main__":
    main()
