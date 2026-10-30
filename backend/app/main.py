"""
Gurbani Bhavarth Search -- backend API (FastAPI).

Run from the project root:

    python -m uvicorn backend.app.main:app --port 8000

then open http://localhost:8000 (the backend also serves the frontend/ folder).

Endpoints
  GET /api/health                         what is loaded (counts, models, explanations on/off)
  GET /api/search?q=...&k=5&d=5           k main results + a pool of "different words" candidates
                                          (up to 2d+2; the frontend shows at most d, verified)
  GET /api/summaries?query=ID,ID&padas=ID,ID[&q=text]   reason + verdict per pada (one LLM call)
  GET /api/ask?q=question                 a question (English/Punjabi) -> 1-2 padas, explained
  GET /api/explain?query=ID,ID&match=ID   detailed: why two padas match (free LLM; needs LLM_API_KEY)
"""

import os
import sys

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from . import ask as asking, config, summaries
from .engine import Engine, QueryError

# The explanation prompt and schema live in common/ (shared with the pipeline's
# command line), so the command line and the website use exactly the same ones.
sys.path.insert(0, str(config.COMMON_DIR))
import explain_match  # noqa: E402

# Padas send the three paraphrasing Steeks only: the word glossary roughly
# doubles the prompt, and Groq's free tier caps a request at 8K tokens a minute.
# They are labelled Steek 1/2/3, so explanations never name the commentators --
# the website identifies them only in its footer.
EXPLAIN_COLUMNS = [("steek_ss", "Steek 1"), ("steek_ms", "Steek 2"), ("steek_ft", "Steek 3")]

engine = Engine()

app = FastAPI(title="Gurbani Bhavarth Search", version="2.0",
              description="Pada-level semantic search over Sri Guru Granth Sahib")
app.add_middleware(CORSMiddleware, allow_origins=config.CORS_ORIGINS,
                   allow_methods=["GET"], allow_headers=["*"])


@app.middleware("http")
async def revalidate_frontend(request, call_next):
    """Without a Cache-Control header the browser guesses how long to keep the page
    and its JS/CSS, and kept showing an old frontend after it changed. no-cache makes
    it ask every time; unchanged files come back as a tiny 304, so it stays fast."""
    response = await call_next(request)
    if not request.url.path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-cache"
    return response


@app.get("/api/health")
def health():
    ref, _ = engine.model_info()
    return {"status": "ok", "unit": "pada",
            "padas": len(engine.padas), "lines": len(engine.docs),
            "shabads": int(len(set(engine.pada_shabad.tolist()))),
            "pada_model": ref, "ranking": "keyword (char-BM25) + meaning model, reciprocal-rank fusion",
            "explain": bool(os.getenv(explain_match.API_KEY_VAR)),
            "explain_model": explain_match.MODEL, "ask_model": asking.writer_model()}


@app.get("/api/search")
def search(q: str = Query(..., min_length=1, max_length=4000),
           k: int = Query(config.TOP_K, ge=1, le=config.MAX_K),
           d: int = Query(config.DIFF_K, ge=0, le=config.MAX_K)):
    try:
        return engine.search(q.strip(), k, d)
    except QueryError as e:
        raise HTTPException(e.status, str(e))


def _ids(text: str, name: str) -> list[int]:
    try:
        ids = [int(x) for x in text.split(",") if x.strip()]
    except ValueError:
        raise HTTPException(400, f"{name} must be pada ids, e.g. 12,13")
    if not all(0 <= p < len(engine.padas) for p in ids):
        raise HTTPException(404, "Unknown pada id.")
    return ids


@app.get("/api/summaries")
def card_reasons(padas: str, query: str = "", q: str = ""):
    """Per result pada: a one-line reason and a verdict (same / related / different),
    all from a single LLM request. The frontend filters "different words" by verdict."""
    if not os.getenv(explain_match.API_KEY_VAR):
        return {"items": {}, "error": "off"}
    pada_ids = _ids(padas, "padas")[:config.MAX_K + config.DIFF_POOL_MAX]
    result = summaries.reasons(engine, explain_match.make_client(), _ids(query, "query"), q, pada_ids)
    return {"items": {str(k): v for k, v in result["items"].items()}, "error": result["error"]}


@app.get("/api/ask")
def ask(q: str = Query(..., min_length=1, max_length=asking.QUESTION_MAX)):
    """A question in English or Punjabi -> one or two padas that answer it, explained
    in both languages (see backend/app/ask.py)."""
    try:
        return asking.ask(engine, q)
    except asking.AskError as e:
        raise HTTPException(e.status, str(e))


@app.get("/api/explain")
def explain(query: str, match: int):
    """query: comma-separated pada ids of the query; match: one result pada id."""
    if not os.getenv(explain_match.API_KEY_VAR):
        raise HTTPException(503, "Explanations are off: add LLM_API_KEY to .env and restart the server.")
    try:
        q_ids = [int(x) for x in query.split(",") if x.strip()]
    except ValueError:
        raise HTTPException(400, "query must be pada ids, e.g. 12,13")
    n = len(engine.padas)
    if not q_ids or not all(0 <= p < n for p in q_ids + [match]):
        raise HTTPException(404, "Unknown pada id.")
    try:
        result = explain_match.explain(explain_match.make_client(),
                                       engine.pada_doc(q_ids, explain_match.STEEKS),
                                       engine.pada_doc([match], explain_match.STEEKS),
                                       engine.raag, explain_match.load_cache(), EXPLAIN_COLUMNS)
    except SystemExit as e:                  # explain_match reports API errors by exiting
        raise HTTPException(502, str(e))
    return {"query_padas": q_ids, "match_pada": match, **result}


# --- frontend -----------------------------------------------------------------
if not (config.FRONTEND_DIR / "index.html").exists():
    sys.exit(f"Missing {config.FRONTEND_DIR / 'index.html'}.")
app.mount("/assets", StaticFiles(directory=config.FRONTEND_DIR / "assets"), name="assets")


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(config.FRONTEND_DIR / "index.html")
