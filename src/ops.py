"""
Live /ops dashboard for igbo-rag.

Matches the actual module layout in api.py: flat imports (no `src.` prefix),
rag_pipeline._faiss_index / _corrections, db.count_corrections() /
db.list_corrections().

Wire in:

    from ops import router as ops_router
    app.include_router(ops_router)

Then visit http://localhost:8000/ops.

One thing I couldn't confirm without seeing db.py: whether there's an
`evals` collection / a function to fetch the latest RAGAS snapshot. This
file calls `db.get_latest_eval()` if it exists and falls back to "no eval
data" if it doesn't — add that function to db.py (reading your evals
collection, sorted by created_at desc, limit 1) to light up that panel.
"""

import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import faiss
from fastapi import APIRouter
from fastapi.responses import HTMLResponse

import rag_pipeline
from rag_pipeline import _faiss_index, FAISS_INDEX_PATH, translate

import db as db_module

router = APIRouter()

# Real query to time the live end-to-end path. Change to something you know
# is well-represented in the corpus if you want a more representative number.
PROBE_QUERY = "Ụtụtụ ọma"


def get_correction_stats():
    """Real numbers from db.py — same functions api.py already uses."""
    total = db_module.count_corrections()
    loaded = len(rag_pipeline._corrections)
    pending = len(db_module.pending_reingestion())
    latest_list = db_module.list_corrections(limit=1, skip=0)
    latest_at = None
    if latest_list:
        ts = latest_list[0].get("created_at")
        latest_at = ts.isoformat() if hasattr(ts, "isoformat") else ts
    return {
        "total_in_mongo": total,
        "loaded_in_memory": loaded,
        "pending_reingestion": pending,
        "latest_correction_at": latest_at,
    }


def get_latest_eval():
    """
    Real function is list_eval_runs(limit=1) — there's no get_latest_eval()
    in db.py. Scores live nested under doc["summary"] (see eval.py
    merge_results()), not as top-level fields.

    Note: as of this codebase, eval.py never actually calls save_eval_run()
    — it only writes eval_results.json to disk. This panel will show
    "no snapshot" against your real system until you add that call to
    eval.py's main(), e.g.:
        from db import save_eval_run
        save_eval_run(results)
    """
    runs = db_module.list_eval_runs(limit=1)
    if not runs:
        return None
    doc = runs[0]
    summary = doc.get("summary", {})
    run_at = doc.get("run_at")
    return {
        "answer_relevancy": summary.get("answer_relevancy"),
        "faithfulness": summary.get("faithfulness"),
        "context_precision": summary.get("context_precision"),
        "created_at": run_at.isoformat() if hasattr(run_at, "isoformat") else run_at,
    }


def get_faiss_stats():
    """Live off the actual loaded index object, same as /debug/faiss."""
    ntotal = _faiss_index.ntotal
    path = Path(FAISS_INDEX_PATH)
    last_modified = None
    if path.exists():
        last_modified = datetime.fromtimestamp(
            path.stat().st_mtime, tz=timezone.utc
        ).isoformat()
    return {"vector_count": ntotal, "index_last_modified": last_modified}


def measure_raw_search_latency():
    """Same pattern as your existing /debug/faiss — raw FAISS search only."""
    start = time.time()
    vec = np.random.rand(1, 768).astype(np.float32)
    faiss.normalize_L2(vec)
    _distances, _indices = _faiss_index.search(vec, 1)
    return round((time.time() - start) * 1000, 2)


def measure_end_to_end_latency():
    """
    Real call through translate() — embed + retrieval + generation — using
    the actual probe query, not synthetic random vectors. This is the number
    that matches what a user actually experiences.
    """
    start = time.perf_counter()
    result = translate(PROBE_QUERY, direction=None)
    elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
    return {
        "total_ms": elapsed_ms,
        "retrieval_quality": result.get("retrieval_quality"),
        "probe_query": PROBE_QUERY,
    }


def render_dashboard(corrections, eval_snapshot, faiss_stats, raw_search_ms, e2e) -> str:
    if eval_snapshot:
        eval_block = f"""
        <div class="card">
          <p class="label">answer relevancy</p>
          <p class="value">{eval_snapshot['answer_relevancy']:.3f}</p>
        </div>
        <div class="card">
          <p class="label">faithfulness</p>
          <p class="value warn">{eval_snapshot['faithfulness']:.3f}</p>
        </div>
        <div class="card">
          <p class="label">context precision</p>
          <p class="value">{eval_snapshot['context_precision']:.3f}</p>
        </div>
        """
        eval_meta = f"<p class='meta'>last eval run: {eval_snapshot['created_at']}</p>"
    else:
        eval_block = "<div class='card'><p class='label'>eval</p><p class='value' style='font-size:13px;'>no snapshot in MongoDB</p></div>"
        eval_meta = "<p class='meta'>eval.py doesn't call save_eval_run() yet — see get_latest_eval() docstring</p>"

    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>igbo-rag / ops</title>
<style>
  body {{ background: #0e0e0d; color: #e8e6df; font-family: -apple-system, sans-serif; margin: 0; padding: 2rem; }}
  .wrap {{ max-width: 780px; margin: 0 auto; }}
  h1 {{ font-size: 16px; font-weight: 500; font-family: monospace; margin: 0 0 4px; }}
  .status {{ font-size: 13px; color: #9a988f; margin: 0 0 1.5rem; }}
  .dot {{ display: inline-block; width: 7px; height: 7px; border-radius: 50%; background: #5dcaa5; margin-right: 6px; }}
  .grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin-bottom: 1.5rem; }}
  .card {{ background: #1a1a18; border-radius: 8px; padding: 1rem; }}
  .label {{ font-size: 12px; color: #9a988f; margin: 0 0 6px; }}
  .value {{ font-size: 22px; font-weight: 500; margin: 0; font-family: monospace; }}
  .value.warn {{ color: #ef9f27; }}
  .value.ok {{ color: #5dcaa5; }}
  table {{ width: 100%; font-size: 13px; border-collapse: collapse; }}
  td {{ padding: 6px 0; border-top: 1px solid #2a2a27; }}
  td.k {{ color: #9a988f; }}
  td.v {{ text-align: right; font-family: monospace; }}
  .meta {{ font-size: 12px; color: #6a6960; margin-top: 4px; }}
  .section {{ background: #1a1a18; border-radius: 12px; padding: 1rem 1.25rem; margin-bottom: 1rem; }}
  .section p.title {{ font-size: 14px; font-weight: 500; margin: 0 0 12px; }}
</style></head>
<body><div class="wrap">
  <h1>igbo-rag / ops</h1>
  <p class="status"><span class="dot"></span>live — generated {datetime.now(timezone.utc).isoformat()}</p>

  <div class="grid">
    <div class="card">
      <p class="label">end-to-end latency (live call)</p>
      <p class="value ok">{e2e['total_ms']}ms</p>
      <p class="meta">translate("{e2e['probe_query']}") · quality: {e2e['retrieval_quality']}</p>
    </div>
    <div class="card">
      <p class="label">raw FAISS search</p>
      <p class="value">{raw_search_ms}ms</p>
      <p class="meta">random probe vector, k=1</p>
    </div>
    <div class="card">
      <p class="label">index size</p>
      <p class="value">{faiss_stats['vector_count']:,}</p>
      <p class="meta">updated {faiss_stats['index_last_modified'] or 'unknown'}</p>
    </div>
  </div>

  <div class="grid">
    {eval_block}
  </div>
  {eval_meta}

  <div class="section">
    <p class="title">corrections store</p>
    <table>
      <tr><td class="k">total in MongoDB</td><td class="v">{corrections['total_in_mongo']}</td></tr>
      <tr><td class="k">pending reingestion</td><td class="v">{corrections['pending_reingestion']}</td></tr>
      <tr><td class="k">loaded in running process</td><td class="v">{corrections['loaded_in_memory']}</td></tr>
      <tr><td class="k">most recent correction</td><td class="v">{corrections['latest_correction_at'] or 'none'}</td></tr>
    </table>
  </div>
</div></body></html>"""
    return html


@router.get("/ops", response_class=HTMLResponse)
def ops_dashboard():
    corrections = get_correction_stats()
    eval_snapshot = get_latest_eval()
    faiss_stats = get_faiss_stats()
    raw_search_ms = measure_raw_search_latency()
    e2e = measure_end_to_end_latency()
    return render_dashboard(corrections, eval_snapshot, faiss_stats, raw_search_ms, e2e)


@router.get("/ops.json")
def ops_dashboard_json():
    return {
        "corrections": get_correction_stats(),
        "eval": get_latest_eval(),
        "faiss": get_faiss_stats(),
        "raw_search_ms": measure_raw_search_latency(),
        "end_to_end": measure_end_to_end_latency(),
    }