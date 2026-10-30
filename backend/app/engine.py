"""
The search engine: pada retrieval with hybrid ranking.

The unit is the PADA -- the lines of a Shabad up to a ॥n॥ marker or a Rahao
line. A single line is often half a thought; its pada carries the context.

  * Query  : a line from the Granth is mapped to its pada; several lines or a
             whole Shabad average their padas; anything else is free text,
             encoded by the pada model (loaded on first use, slow on CPU).
  * Main results : char-BM25 and the pada model (models/bge-m3-pada) merged by
             reciprocal-rank fusion -- best on the pada gold set (H@10 51.8 vs
             48.8 keyword and 42.6 model alone). One pada per Shabad.
  * Different words : the pada model's best matches sharing NO word (by
             consonant skeleton) with the query. Fusion buries these, and they
             are what keyword search can never find.
  * Each result highlights the line closest to the query's line.

Loaded once at import; the server is ready in a few seconds.
"""

import json
import re
import sys
import threading
import time
import unicodedata

import numpy as np
import pandas as pd

from . import config

_STRIP = re.compile(r"[॥।0-9੦-੯]")
_NOISE = re.compile(r"[॥।0-9੦-੯]+")
CHAR_N = (3, 4, 5)
# Consonant skeleton marks, as strip_marks in the mining scripts (keep in step).
_MARKS = "ੁੂਿੀੇੈੋੌਾਂੰ੍ਃ"
# Steeks shown on the website, in display order: Steek 1 = Prof. Sahib Singh,
# Steek 2 = Manmohan Singh, Steek 3 = Faridkot Teeka. The Padd-arth word glossary
# (pss) is deliberately never sent to the website.
STEEK_COLS = ("ss", "ms", "ft")
# A query is split into lines after each ॥ -- but never before ਰਹਾਉ, which
# belongs to the line it closes ("...॥੧॥ ਰਹਾਉ ॥"), not a line of its own.
_LINE_SPLIT = re.compile(r"(?<=॥)\s+(?!ਰਹਾਉ)")


# --- text helpers (tokenisers identical to evaluate_padas.py; keep in step) -------
def tokenize(text: str):
    return [w for w in _STRIP.sub(" ", text).split() if len(w) > 1]


def char_tokenize(text: str):
    cleaned = "_".join(tokenize(text))
    return [cleaned[i:i + n] for n in CHAR_N for i in range(len(cleaned) - n + 1)]


def skeleton_words(text: str) -> set:
    """Words reduced to consonants: ਧਨ / ਧਨੁ / ਧਨਿ count as one word."""
    out = set()
    for w in tokenize(text):
        s = "".join(c for c in unicodedata.normalize("NFD", w)
                    if not unicodedata.combining(c) and c not in _MARKS)
        if len(s) > 1:
            out.add(s)
    return out


_STEEK_TOKEN = re.compile(r"[^\s॥।,—()!?'\"‘’:;.\-0-9੦-੯]+")


def steek_tokenize(text: str):
    """Words of Steek prose (and Gurbani) for question search. A final ੁ / ਿ is a
    grammatical ending in Gurbani (ਨਾਮੁ, ਹਰਿ), so it is dropped to meet the Steek's ਨਾਮ."""
    return [t for t in (w.rstrip("ੁਿ") for w in _STEEK_TOKEN.findall(text or "")) if len(t) > 1]


def lookup_key(text: str) -> str:
    """Loose key for lookup: nobody types the danda and verse numbers."""
    return " ".join(_NOISE.sub(" ", text).split())


def rrf(first: list, second: list) -> list:
    score = {}
    for ranking in (first, second):
        for r, c in enumerate(ranking):
            score[c] = score.get(c, 0.0) + 1.0 / (config.RRF_K + r + 1)
    return sorted(score, key=lambda c: -score[c])


class QueryError(Exception):
    """A query the engine cannot answer, with a message for the reader."""

    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


# --- the engine ------------------------------------------------------------------
class Engine:
    def __init__(self):
        started = time.time()
        pada_dir = config.EMBED_DIR / f"pada_{config.PADA_TAG}"
        line_dir = config.EMBED_DIR / config.LINE_TAG
        for path, hint in ((config.PADAS_JSONL, "Run backend/pipeline/preprocessing/build_padas.py."),
                           (config.CORPUS_JSONL, "Run backend/pipeline/training/create_training_pairs.py."),
                           (pada_dir / "embeddings.npy",
                            f"Encode the padas: build_pada_index.py --tag {config.PADA_TAG}."),
                           (line_dir / "embeddings.npy",
                            f"Encode the lines: build_index.py --tag {config.LINE_TAG}.")):
            if not path.exists():
                sys.exit(f"Missing {path}. {hint}")

        self.padas = [json.loads(l) for l in config.PADAS_JSONL.open(encoding="utf-8")]
        self.docs = [json.loads(l) for l in config.CORPUS_JSONL.open(encoding="utf-8")]
        self.pada_emb = np.load(pada_dir / "embeddings.npy")
        self.line_emb = np.load(line_dir / "embeddings.npy")
        if len(self.pada_emb) != len(self.padas) or len(self.line_emb) != len(self.docs):
            sys.exit("Pada or line embeddings do not match their text files -- rebuild them.")
        assert all(p["pada_id"] == i for i, p in enumerate(self.padas)), "pada ids must be row numbers"

        self.raag = {}
        if config.RAAG_META.exists():
            df = pd.read_csv(config.RAAG_META, encoding="utf-8")
            self.raag = {int(r.raag_id): str(r.raag) for r in df.itertuples() if not pd.isna(r.raag_id)}

        self.occ = {o["id"]: o for d in self.docs for o in d["occurrences"]}
        self.pada_words = [skeleton_words(p["text"]) for p in self.padas]
        self.pada_shabad = np.array([p["shabad_id"] for p in self.padas])
        self.padas_of_line = {}
        for p in self.padas:
            for i in p["line_index"]:
                self.padas_of_line.setdefault(i, []).append(p["pada_id"])
        self.by_text = {d["pankati"]: i for i, d in enumerate(self.docs)}
        self.by_key = {}
        for i, d in enumerate(self.docs):
            self.by_key.setdefault(lookup_key(d["pankati"]), i)

        from rank_bm25 import BM25Okapi
        print("building the keyword (char-BM25) index over the padas ...", flush=True)
        self.bm25 = BM25Okapi([char_tokenize(p["text"]) for p in self.padas])
        # Questions ("how do I remove ego?") are matched against the Steeks: they are
        # modern Punjabi prose, the language a question's search terms come in.
        # Steek 1 + 2 only; Steek 3 is in Braj.
        print("building the question index over the Steeks ...", flush=True)
        self.steek_bm25 = BM25Okapi([steek_tokenize(p.get("steek_ss")) + steek_tokenize(p.get("steek_ms"))
                                     + steek_tokenize(p["text"]) for p in self.padas])
        self.shabad_padas = {}
        for p in self.padas:
            self.shabad_padas.setdefault(p["shabad_id"], []).append(p["pada_id"])

        self._model = None
        self._model_lock = threading.Lock()
        print(f"engine ready in {time.time() - started:.0f} s: {len(self.padas):,} padas, "
              f"{len(self.docs):,} lines", flush=True)

    # -- model --------------------------------------------------------------------
    def model_info(self):
        info = config.EMBED_DIR / f"pada_{config.PADA_TAG}" / "model.json"
        if not info.exists():
            return None, 256
        meta = json.loads(info.read_text(encoding="utf-8"))
        local = config.PROJECT_ROOT / meta["model"]
        return (str(local) if local.exists() else meta["model"]), meta.get("max_seq", 256)

    def encode(self, text: str):
        """Free text -> pada-space vector, with the model that built the pada vectors."""
        ref, max_seq = self.model_info()
        if ref is None:
            raise QueryError("No model is recorded for the pada vectors, so only text from the "
                             "Granth can be searched.", 503)
        with self._model_lock:
            if self._model is None:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(ref)
                self._model.max_seq_length = max_seq
        return self._model.encode(text, normalize_embeddings=True)

    # -- query --------------------------------------------------------------------
    def find(self, text: str):
        """Exact -> punctuation-insensitive -> partial. Returns (line index, note)."""
        if text in self.by_text:
            return self.by_text[text], None
        key = lookup_key(text)
        if key in self.by_key:
            return self.by_key[key], None
        if len(key) >= 8:
            partial = [i for k, i in self.by_key.items() if key in k]
            if len(partial) == 1:
                return partial[0], None
            if partial:
                best = min(partial, key=lambda i: len(self.docs[i]["pankati"]))
                return best, f"{len(partial)} lines contain that text; the shortest was used"
        return None, None

    def resolve(self, q: str) -> dict:
        parts = [p.strip() for p in _LINE_SPLIT.split(q) if p.strip()] or [q]
        found = [(p, *self.find(p)) for p in parts]
        lines = [i for _, i, _ in found if i is not None and i in self.padas_of_line]
        if lines:
            pada_ids = list(dict.fromkeys(self.padas_of_line[i][0] for i in lines))
            vec = self.pada_emb[pada_ids].mean(axis=0)
            excl = {self.padas[p]["shabad_id"] for p in pada_ids}
            for i in lines:
                excl |= {o["shabad_id"] for o in self.docs[i]["occurrences"]}
            return {"mode": "granth", "vec": vec / np.linalg.norm(vec),
                    "line_vec": self.line_emb[lines].mean(axis=0),
                    "text": " ".join(self.padas[p]["text"] for p in pada_ids),
                    "excl": excl, "pada_ids": pada_ids,
                    "lines": [self.docs[i]["pankati"] for i in lines],
                    "missed": [p for p, i, _ in found if i is None],
                    "note": next((n for _, _, n in found if n), None),
                    "ang": self.padas[pada_ids[0]]["ang"]}
        return {"mode": "free_text", "vec": self.encode(q), "line_vec": None, "text": q,
                "excl": set(), "pada_ids": [], "lines": [], "missed": [], "note": None, "ang": None}

    def _ranked(self, scores, excl):
        order = np.argpartition(-scores, config.CANDIDATES)[:config.CANDIDATES]
        order = order[np.argsort(-scores[order])]
        return [int(c) for c in order if scores[c] > 0 and self.pada_shabad[c] not in excl]

    def _card(self, pid, query, dense_sim, found_by):
        p = self.padas[pid]
        if query["line_vec"] is not None:          # line closest to the query's line(s)
            best = int(np.argmax(self.line_emb[p["line_index"]] @ query["line_vec"]))
        else:                                      # free text: most shared words
            qwords = skeleton_words(query["text"])
            best = int(np.argmax([len(qwords & skeleton_words(t)) for t in p["lines"]]))
        lines = [{"text": text, "best": n == best,
                  "steeks": [self.occ.get(oid, {}).get(f"steek_{k}") or None for k in STEEK_COLS]}
                 for n, (text, oid) in enumerate(zip(p["lines"], p["line_ids"]))]
        return {"pada_id": pid, "lines": lines, "ang": p["ang"],
                "ang_gap": abs(p["ang"] - query["ang"]) if query["ang"] else None,
                "raag": self.raag.get(p["raag_id"]) if p["raag_id"] else None,
                "shabad_id": p["shabad_id"], "similarity": round(float(dense_sim), 4),
                "found_by": found_by,
                "shared_words": len(self.pada_words[pid] & skeleton_words(query["text"]))}

    def search(self, q: str, k: int, d: int = config.DIFF_K) -> dict:
        """k results in the main list, d in "same meaning, different words"."""
        started = time.time()
        query = self.resolve(q)
        dense_scores = self.pada_emb @ query["vec"]
        dense = self._ranked(dense_scores, query["excl"])
        lexical = self._ranked(self.bm25.get_scores(char_tokenize(query["text"])), query["excl"])
        dense_top, lexical_top = set(dense[:50]), set(lexical[:50])

        def found_by(pid):
            return [n for n, top in (("meaning", dense_top), ("words", lexical_top)) if pid in top]

        main, used = [], set()
        for pid in rrf(lexical, dense):                         # one pada per Shabad
            if self.pada_shabad[pid] in used:
                continue
            used.add(self.pada_shabad[pid])
            main.append(self._card(pid, query, dense_scores[pid], found_by(pid)))
            if len(main) >= k:
                break

        # "Same meaning, different words": the model PROPOSES a pool larger than d;
        # the frontend shows only those the LLM check (/api/summaries) judges same
        # or related -- up to d, and nothing at all if none pass. The model alone
        # filled every slot, including weak matches the LLM then called unrelated.
        qwords = skeleton_words(query["text"])
        pool_size = min(2 * d + 2, config.DIFF_POOL_MAX) if d else 0
        different = []
        for pid in dense:
            if len(different) >= pool_size:
                break
            if self.pada_shabad[pid] in used or self.pada_words[pid] & qwords:
                continue
            used.add(self.pada_shabad[pid])
            different.append(self._card(pid, query, dense_scores[pid], ["meaning"]))

        read_as = {key: query[key] for key in ("mode", "lines", "missed", "note")}
        read_as["padas"] = [{"pada_id": p, "lines": self.padas[p]["lines"], "ang": self.padas[p]["ang"],
                             "raag": self.raag.get(self.padas[p]["raag_id"]) if self.padas[p]["raag_id"] else None}
                            for p in query["pada_ids"]]
        return {"query": q, "read_as": read_as, "results": main,
                "different_candidates": different, "different_limit": d,
                "seconds": round(time.time() - started, 2)}

    # -- questions ----------------------------------------------------------------
    def question_candidates(self, terms: str, n: int) -> list[int]:
        """Punjabi search terms -> the n best padas by their Steeks, one per Shabad."""
        scores = self.steek_bm25.get_scores(steek_tokenize(terms))
        out, used = [], set()
        for pid in self._ranked(scores, set()):
            if self.pada_shabad[pid] not in used:
                used.add(self.pada_shabad[pid])
                out.append(pid)
                if len(out) >= n:
                    break
        return out

    def rahao_pada(self, shabad_id):
        """The Shabad's Rahao pada (its central idea), if it has one."""
        return next((p for p in self.shabad_padas.get(shabad_id, [])
                     if "ਰਹਾਉ" in self.padas[p]["text"]), None)

    def pada_view(self, pid, key_line=None) -> dict:
        """A pada for display with every line's Steeks, plus its whole Shabad."""
        p = self.padas[pid]
        key = lookup_key(key_line) if key_line else None
        keys = [lookup_key(t) for t in p["lines"]]
        if key and key not in keys:             # the LLM copied it loosely: nearest by words
            words = skeleton_words(key)
            overlap = [len(words & skeleton_words(t)) for t in p["lines"]]
            key = keys[overlap.index(max(overlap))] if max(overlap) >= 2 else None
        lines = [{"text": text, "best": key is not None and lookup_key(text) == key,
                  "steeks": [self.occ.get(oid, {}).get(f"steek_{k}") or None for k in STEEK_COLS]}
                 for text, oid in zip(p["lines"], p["line_ids"])]
        shabad = [{"pada_id": q, "lines": self.padas[q]["lines"], "this": q == pid}
                  for q in self.shabad_padas.get(p["shabad_id"], [])]
        return {"pada_id": pid, "lines": lines, "ang": p["ang"], "shabad_id": p["shabad_id"],
                "raag": self.raag.get(p["raag_id"]) if p["raag_id"] else None, "shabad": shabad}

    # -- explanation support ------------------------------------------------------
    def pada_doc(self, pada_ids, steek_columns):
        """Padas -> the document shape explain_match expects, Steeks joined per column."""
        first = self.padas[pada_ids[0]]
        occ = {"ang": first["ang"], "raag_id": first["raag_id"], "shabad_id": first["shabad_id"]}
        for col, _ in steek_columns:
            occ[col] = " ".join((self.occ.get(oid, {}).get(col) or "").strip()
                                for p in pada_ids for oid in self.padas[p]["line_ids"]).strip()
        return {"pankati": " ".join(self.padas[p]["text"] for p in pada_ids), "occurrences": [occ]}
