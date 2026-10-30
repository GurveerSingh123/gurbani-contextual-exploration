"""
Interactive Pankati -> Pankati semantic search.

    python backend/pipeline/retrieval/semantic_search.py --tag baseline
    python backend/pipeline/retrieval/semantic_search.py --tag finetuned --compare baseline

Given a Pankati, returns other Pankatis carrying the same Bhavarth, together
with their Steeks, Ang, Raag and Shabad.

Results are drawn from DIFFERENT Shabads, and capped at one verse per Shabad,
because a neighbouring line is something the reader can already see. That is a
presentation choice, not a correctness one -- same-Shabad verses are usually
relevant, just not useful to surface.

A verse already in the corpus is looked up by its cached vector, so the demo
answers instantly without loading the 2.3 GB model. Free text (--text) needs
the model and takes a few seconds on CPU.
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(PROJECT_ROOT / ".env")

PROCESSED = PROJECT_ROOT / os.getenv("PROCESSED_DATA_DIR", "data/processed")
EMBED_DIR = PROJECT_ROOT / os.getenv("EMBEDDING_DIR", "data/embeddings")
RAAG_META = PROJECT_ROOT / os.getenv("RAAG_METADATA", "data/metadata/raag_metadata.csv")
CORPUS_JSONL = PROCESSED / "corpus_pankatis.jsonl"
TOP_K = int(os.getenv("TOP_K", "10"))


_NOISE = re.compile(r"[॥।0-9੦-੯]+")


def lookup_key(text: str) -> str:
    """Loose key for corpus lookup: nobody types the danda and verse numbers."""
    return " ".join(_NOISE.sub(" ", text).split())


def load_raag_names() -> dict:
    if not RAAG_META.exists():
        return {}
    df = pd.read_csv(RAAG_META, encoding="utf-8")
    return {int(r.raag_id): str(r.raag)
            for r in df.itertuples() if not pd.isna(r.raag_id)}


def load_corpus(tag: str):
    """Documents in the exact order build_index.py encoded them."""
    meta_path = EMBED_DIR / tag / "corpus_meta.jsonl"
    emb_path = EMBED_DIR / tag / "embeddings.npy"
    for p in (meta_path, emb_path, CORPUS_JSONL):
        if not p.exists():
            sys.exit(f"Missing {p}. Run build_index.py --tag {tag} first.")

    meta = [json.loads(l) for l in meta_path.open(encoding="utf-8")]
    docs = [json.loads(l) for l in CORPUS_JSONL.open(encoding="utf-8")]
    emb = np.load(emb_path)

    if len(meta) != len(emb) or len(docs) != len(meta):
        sys.exit("corpus/embedding size mismatch -- rebuild the index.")
    # Order is assumed, so verify it rather than trusting it.
    for i in (0, len(docs) // 2, len(docs) - 1):
        if docs[i]["pankati"] != meta[i]["pankati"]:
            sys.exit("corpus order mismatch -- rebuild the index.")
    return meta, docs, emb


def model_for_tag(tag: str) -> str:
    """Model that encoded data/embeddings/<tag>, as recorded by build_index.py.

    A free-text query has to be encoded by that same model; a vector from any
    other model lives in a different space and its scores are meaningless.
    """
    info = EMBED_DIR / tag / "model.json"
    if not info.exists():
        sys.exit(f"No model recorded for tag '{tag}' ({info} is missing).\n"
                 f"Verses already in the corpus still work; free text needs the model "
                 f"that built this tag. Create {info} containing "
                 '{"model": "<path or Hugging Face id>"}.')
    ref = json.loads(info.read_text(encoding="utf-8"))["model"]
    local = PROJECT_ROOT / ref
    if local.exists():
        return str(local)
    if Path(ref).is_absolute() or ref.startswith(("models/", "models\\")):
        sys.exit(f"Tag '{tag}' was built by {ref}, which is not on this machine.")
    return ref                                   # Hugging Face id, e.g. BAAI/bge-m3


def search(emb, meta, query_vec, exclude_shabads, k, per_shabad=1):
    scores = emb @ query_vec
    order = np.argsort(-scores)
    results, used = [], {}
    for i in order:
        shabads = set(meta[i]["shabad_ids"])
        if shabads & exclude_shabads:
            continue
        key = min(shabads)
        if used.get(key, 0) >= per_shabad:
            continue
        used[key] = used.get(key, 0) + 1
        results.append((int(i), float(scores[i])))
        if len(results) >= k:
            break
    return results


def show(results, meta, docs, raag_names, query_ang, label):
    print(f"\n{'=' * 74}\n{label}\n{'=' * 74}")
    for rank, (i, score) in enumerate(results, 1):
        occ = docs[i]["occurrences"][0]
        raag = raag_names.get(occ["raag_id"], "-") if occ["raag_id"] else "-"
        gap = f" ({abs(occ['ang'] - query_ang):+d})" if query_ang else ""
        print(f"\n{rank:>2}. [{score:.3f}]  {docs[i]['pankati']}")
        print(f"    Ang {occ['ang']}{gap}   Raag: {raag}   Shabad {occ['shabad_id']}")
        if occ.get("steek_ss"):
            print(f"    ਅਰਥ (SS): {occ['steek_ss'][:150]}")
        elif occ.get("steek_ms"):
            print(f"    ਅਰਥ (MS): {occ['steek_ms'][:150]}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default=os.getenv("SEARCH_TAG", "baseline"))
    ap.add_argument("--compare", default=None,
                    help="second embedding tag, to show both side by side")
    ap.add_argument("--pankati", default=None, help="exact verse text from the corpus")
    ap.add_argument("--text", default=None, help="free text (loads the model)")
    ap.add_argument("--top-k", type=int, default=TOP_K)
    ap.add_argument("--per-shabad", type=int, default=1)
    ap.add_argument("--allow-same-shabad", action="store_true")
    args = ap.parse_args()

    raag_names = load_raag_names()
    tags = [args.tag] + ([args.compare] if args.compare else [])
    loaded = {t: load_corpus(t) for t in tags}
    meta, docs, _ = loaded[args.tag]
    print(f"corpus: {len(docs):,} Pankatis")

    by_text = {d["pankati"]: i for i, d in enumerate(docs)}
    by_key = {}
    for i, d in enumerate(docs):
        by_key.setdefault(lookup_key(d["pankati"]), i)

    def find_in_corpus(query_text):
        """Exact -> punctuation-insensitive -> partial. Returns index or None."""
        if query_text in by_text:
            return by_text[query_text]
        key = lookup_key(query_text)
        if key in by_key:
            return by_key[key]
        if len(key) >= 8:                      # too short to disambiguate
            partial = [i for k, i in by_key.items() if key in k]
            if len(partial) == 1:
                return partial[0]
            if len(partial) > 1:
                print(f"({len(partial)} verses contain that text; using the shortest)")
                return min(partial, key=lambda i: len(docs[i]["pankati"]))
        return None

    models = {}

    def encoder(tag):
        """The model that built `tag`, loaded once per session."""
        if tag not in models:
            from sentence_transformers import SentenceTransformer
            models[tag] = SentenceTransformer(model_for_tag(tag))
            models[tag].max_seq_length = 128     # same as build_index.py
        return models[tag]

    def split_verses(text):
        """Several Pankatis pasted together -> the individual verses."""
        parts = [p.strip() for p in re.split(r"(?<=॥)\s+", text) if p.strip()]
        return parts if len(parts) > 1 else []

    def resolve_multi(query_text):
        """Average the vectors of several verses.

        Averaging sharpens a query when the verses share a theme, and blurs it
        when they do not -- the mean of two different meanings belongs to
        neither. Averaging cached vectors also beats encoding the concatenated
        string, and needs no model.
        """
        parts = split_verses(query_text)
        if len(parts) < 2:
            return None
        found = [(p, find_in_corpus(p)) for p in parts]
        hits = [(p, i) for p, i in found if i is not None]
        if len(hits) < 2:
            return None

        for p, i in found:
            print(f"  {'OK  ' if i is not None else 'MISS'} {p}")
        if len(hits) < len(found):
            print(f"  (using {len(hits)} of {len(found)} verses)")

        excl = set()
        if not args.allow_same_shabad:
            for _, i in hits:
                excl |= set(meta[i]["shabad_ids"])
        vecs = {}
        for t in tags:
            v = np.mean([loaded[t][2][i] for _, i in hits], axis=0)
            vecs[t] = v / np.linalg.norm(v)
        return vecs, excl, docs[hits[0][1]]["occurrences"][0]["ang"]

    def resolve(query_text):
        """-> (query vectors per tag, shabads to exclude, ang)"""
        multi = resolve_multi(query_text)
        if multi is not None:
            return multi

        qi = find_in_corpus(query_text)
        if qi is not None:
            if docs[qi]["pankati"] != query_text:
                print(f"(matched: {docs[qi]['pankati']})")
            occ = docs[qi]["occurrences"][0]
            excl = set() if args.allow_same_shabad else set(meta[qi]["shabad_ids"])
            return ({t: loaded[t][2][qi] for t in tags}, excl, occ["ang"])

        print("(not in corpus -- encoding with the model, this is slow on CPU)")
        vecs = {t: encoder(t).encode(query_text, normalize_embeddings=True) for t in tags}
        return vecs, set(), None

    queries = [args.pankati or args.text] if (args.pankati or args.text) else None

    while True:
        if queries is None:
            try:
                q = input("\nPankati (or 'q' to quit) > ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return
            if q.lower() in {"q", "quit", "exit"} or not q:
                return
        else:
            q = queries[0]

        vecs, excl, ang = resolve(q)
        print(f"\nQUERY: {q}")
        for t in tags:
            res = search(loaded[t][2], loaded[t][0], vecs[t], excl,
                         args.top_k, args.per_shabad)
            show(res, loaded[t][0], loaded[t][1], raag_names, ang, f"model: {t}")

        if queries is not None:
            return


if __name__ == "__main__":
    main()
