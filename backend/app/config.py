"""
Backend settings. Everything comes from the project's .env (see .env.example);
no path is hard-coded, matching the rest of the project.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# backend/app/config.py -> project root is two levels above backend/
PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

PROCESSED = PROJECT_ROOT / os.getenv("PROCESSED_DATA_DIR", "data/processed")
EMBED_DIR = PROJECT_ROOT / os.getenv("EMBEDDING_DIR", "data/embeddings")
RAAG_META = PROJECT_ROOT / os.getenv("RAAG_METADATA", "data/metadata/raag_metadata.csv")
CORPUS_JSONL = PROCESSED / "corpus_pankatis.jsonl"
PADAS_JSONL = PROCESSED / "padas.jsonl"
COMMON_DIR = PROJECT_ROOT / "common"                # code shared by the server and the pipeline
FRONTEND_DIR = PROJECT_ROOT / "frontend"

PADA_TAG = os.getenv("PADA_TAG", "pada")            # pada vectors: data/embeddings/pada_<PADA_TAG>/
LINE_TAG = os.getenv("SEARCH_TAG", "verse_hn")      # line vectors, used to highlight the best line
TOP_K = int(os.getenv("TOP_K", "10"))
DIFF_K = int(os.getenv("DIFF_K", "5"))
MAX_K = 30
DIFF_POOL_MAX = 12      # most "different words" candidates sent for LLM checking per search
CANDIDATES = 400        # taken from each ranking before fusion
RRF_K = 60              # reciprocal-rank-fusion constant, as in the evaluation; not tuned

# Browsers allowed to call the API when the frontend is served separately
# (e.g. a static-file server during development). Same-origin needs nothing.
CORS_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "http://localhost:5500,"
                                              "http://127.0.0.1:5500").split(",") if o.strip()]
