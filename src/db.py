"""
db.py — MongoDB backing store for igbo-rag.

Collections:
  feedback   — HITL corrections submitted via POST /feedback
  evals      — RAGAS evaluation run snapshots

The in-memory _corrections dict in rag_pipeline.py remains the hot path
for every translation lookup.  This module is the persistence layer that
backs it — corrections survive restarts and accumulate over time.
"""

import os
from datetime import datetime, timezone

from dotenv import load_dotenv
from pymongo import DESCENDING, MongoClient
from pymongo.collection import Collection

load_dotenv()

_client: MongoClient | None = None
_db = None


def get_db():
    """Return a handle to the igbo_rag database (lazy singleton)."""
    global _client, _db
    if _db is None:
        uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
        db_name = os.getenv("MONGODB_DB", "igbo_rag")
        _client = MongoClient(uri, serverSelectionTimeoutMS=5000)
        _db = _client[db_name]
        _ensure_indexes(_db)
    return _db


def _ensure_indexes(db) -> None:
    """Create indexes on first connection — idempotent."""
    db.feedback.create_index(
        [("query", DESCENDING), ("direction", DESCENDING)],
        name="query_direction",
    )
    db.feedback.create_index("created_at", name="created_at")
    db.feedback.create_index("reingested", name="reingested")
    db.evals.create_index("run_at", name="run_at")


# ---------------------------------------------------------------------------
# feedback collection
# ---------------------------------------------------------------------------

def save_correction(correction: dict) -> None:
    """
    Persist a single correction.

    Expected fields (from api.py FeedbackRequest):
        query, direction, correct_translation,
        wrong_translation (optional), note (optional), timestamp (ISO str)

    Adds:
        created_at       — UTC datetime object for range queries
        reingested       — False until the nightly re-ingestion job flips it
        reingested_at    — set by the re-ingestion job
    """
    db = get_db()
    doc = {
        **correction,
        "created_at": datetime.now(timezone.utc),
        "reingested": False,
        "reingested_at": None,
    }
    db.feedback.insert_one(doc)


def load_all_corrections() -> list[dict]:
    """
    Return all corrections as plain dicts (no _id).
    Used to rebuild the in-memory _corrections dict on startup or reload.
    """
    db = get_db()
    return list(db.feedback.find({}, {"_id": 0}))


def list_corrections(limit: int = 500, skip: int = 0) -> list[dict]:
    """
    Paginated correction listing for GET /feedback/list.
    Returns newest-first.
    """
    db = get_db()
    cursor = (
        db.feedback.find({}, {"_id": 0})
        .sort("created_at", DESCENDING)
        .skip(skip)
        .limit(limit)
    )
    return list(cursor)


def count_corrections() -> int:
    """Total number of corrections in the store."""
    db = get_db()
    return db.feedback.count_documents({})


def pending_reingestion() -> list[dict]:
    """
    Return corrections not yet re-ingested into the vector store.
    Used by the nightly re-ingestion job (future work).
    """
    db = get_db()
    return list(db.feedback.find({"reingested": False}, {"_id": 0}))


def mark_reingested(queries: list[str]) -> None:
    """
    Mark a batch of corrections as reingested.
    Called by the re-ingestion job after successful corpus update.
    """
    db = get_db()
    db.feedback.update_many(
        {"query": {"$in": queries}},
        {"$set": {"reingested": True, "reingested_at": datetime.now(timezone.utc)}},
    )


# ---------------------------------------------------------------------------
# evals collection
# ---------------------------------------------------------------------------

def save_eval_run(results: dict) -> None:
    """
    Persist a RAGAS evaluation run snapshot.
    Called at the end of eval.py main().
    """
    db = get_db()
    db.evals.insert_one(
        {
            **results,
            "run_at": datetime.now(timezone.utc),
        }
    )


def list_eval_runs(limit: int = 20) -> list[dict]:
    """Return recent eval runs newest-first (for a future /evals endpoint)."""
    db = get_db()
    return list(
        db.evals.find({}, {"_id": 0})
        .sort("run_at", DESCENDING)
        .limit(limit)
    )
