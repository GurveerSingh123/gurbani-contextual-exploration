"""
Pada step 4: score retrievers on the pada -> pada gold set.

Systems (any combination):
  --bm25 / --bm25-char   keyword search on the pada text (the control)
  --linemean TAG         pada vector = mean of its lines' vectors from
                         data/embeddings/TAG (no new encoding needed)
  --dense TAG            pada vector = the whole pada encoded as one text,
                         from data/embeddings/pada_TAG (build_pada_index.py)
  --fuse TAG             HYBRID: char-BM25 and pada_TAG merged by reciprocal-rank
                         fusion (1/(60 + rank) from each list, summed; k = 60 is
                         the standard constant, deliberately not tuned on this set)

Same metrics as evaluate.py, with padas in place of lines: other-Shabad results
only; H@k, MRR, nDCG@10; H@10 on queries whose gold includes a pada sharing no
word with the query; and "no-word", the share of top-10 results sharing no word
with the query (higher = less keyword-driven).

  python backend/pipeline/evaluation/evaluate_padas.py --bm25-char --linemean verse_hn --dense verse_hn
  python backend/pipeline/evaluation/evaluate_padas.py --bm25-char --dense pada --fuse pada

Reads  : data/processed/padas.jsonl, pada_eval_pairs.jsonl
         data/embeddings/<TAG>/embeddings.npy, data/embeddings/pada_<TAG>/embeddings.npy
Writes : data/processed/pada_results.txt
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(PROJECT_ROOT / ".env")

PROCESSED = PROJECT_ROOT / os.getenv("PROCESSED_DATA_DIR", "data/processed")
EMBED_DIR = PROJECT_ROOT / os.getenv("EMBEDDING_DIR", "data/embeddings")
PADAS_JSONL = PROCESSED / "padas.jsonl"
EVAL_JSONL = PROCESSED / "pada_eval_pairs.jsonl"
RESULTS_TXT = PROCESSED / "pada_results.txt"

K_VALUES = [1, 5, 10, 50]
NDCG_K = 10
TOP = 400
RRF_K = 60            # standard reciprocal-rank-fusion constant; deliberately not tuned


def rrf(first: list, second: list) -> list:
    """Reciprocal-rank fusion of two rankings (a missing item contributes 0)."""
    score = {}
    for ranking in (first, second):
        for r, c in enumerate(ranking):
            score[c] = score.get(c, 0.0) + 1.0 / (RRF_K + r + 1)
    return sorted(score, key=lambda c: -score[c])

# --- tokenisers identical to evaluate.py; keep in step -----------------------------
_STRIP = re.compile(r"[॥।0-9੦-੯]")
CHAR_N = (3, 4, 5)


def tokenize(text: str):
    return [w for w in _STRIP.sub(" ", text).split() if len(w) > 1]


def char_tokenize(text: str):
    cleaned = "_".join(tokenize(text))
    return [cleaned[i:i + n] for n in CHAR_N for i in range(len(cleaned) - n + 1)]
# ----------------------------------------------------------------------------------


def evaluate(rankings, queries, label):
    """rankings[qi] -> pada ids, best first, same-Shabad already removed."""
    hits = {k: 0 for k in K_VALUES}
    rr_sum = ndcg_sum = 0.0
    zero_hits = zero_total = 0
    discounts = 1.0 / np.log2(np.arange(2, NDCG_K + 2))
    for qi, q in enumerate(queries):
        gold = {g["pada_id"] for g in q["gold"]}
        gold_zero = {g["pada_id"] for g in q["gold"] if g["word_overlap"] == 0.0}
        ranked = rankings[qi][:max(K_VALUES)]
        found = [c in gold for c in ranked]
        for k in K_VALUES:
            hits[k] += any(found[:k])
        rr_sum += next((1.0 / (i + 1) for i, f in enumerate(found) if f), 0.0)
        gains = np.pad(np.array(found[:NDCG_K], dtype=float), (0, NDCG_K - len(found[:NDCG_K])))
        ideal = np.zeros(NDCG_K)
        ideal[:min(len(gold), NDCG_K)] = 1.0
        ndcg_sum += (gains * discounts).sum() / (ideal * discounts).sum()
        if gold_zero:
            zero_total += 1
            zero_hits += any(c in gold_zero for c in ranked[:10])
    n = len(queries)
    return {"label": label, **{f"hit@{k}": 100 * hits[k] / n for k in K_VALUES},
            "mrr": 100 * rr_sum / n, "ndcg": 100 * ndcg_sum / n,
            "zero": 100 * zero_hits / max(zero_total, 1), "zero_q": zero_total}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bm25", action="store_true", help="word-level BM25 on pada text")
    ap.add_argument("--bm25-char", action="store_true", help="char 3-5gram BM25 on pada text")
    ap.add_argument("--linemean", action="append", default=[], help="line-vector tag to average (repeatable)")
    ap.add_argument("--dense", action="append", default=[], help="pada-encoded tag (repeatable)")
    ap.add_argument("--fuse", action="append", default=[],
                    help="pada-encoded tag to merge with char-BM25 by reciprocal-rank fusion (repeatable)")
    ap.add_argument("--sample", type=int, default=0,
                    help="score only the first N queries (quick test; numbers not comparable)")
    args = ap.parse_args()
    if args.fuse:                       # a hybrid needs both of its halves scored
        args.bm25_char = True
        args.dense += [t for t in args.fuse if t not in args.dense]
    if not (args.bm25 or args.bm25_char or args.linemean or args.dense):
        ap.error("pick at least one of --bm25 / --bm25-char / --linemean / --dense")

    for p in (PADAS_JSONL, EVAL_JSONL):
        if not p.exists():
            sys.exit(f"Missing {p}. Run build_padas.py and build_pada_eval.py first.")
    padas = [json.loads(l) for l in PADAS_JSONL.open(encoding="utf-8")]
    queries = [json.loads(l) for l in EVAL_JSONL.open(encoding="utf-8")]
    if args.sample:
        queries = queries[:args.sample]
    assert all(p["pada_id"] == i for i, p in enumerate(padas)), "pada ids must be row numbers"
    shabad = np.array([p["shabad_id"] for p in padas])
    words = [set(tokenize(p["text"])) for p in padas]
    q_ids = [q["query_pada_id"] for q in queries]
    print(f"padas: {len(padas):,}   queries: {len(queries):,}")

    def no_word_share(rankings):
        share = [not (words[pid] & words[c]) for pid, ranked in zip(q_ids, rankings) for c in ranked[:10]]
        return 100 * np.mean(share)

    def rank_from(scores_fn):
        rankings = []
        for pid in q_ids:
            scores = scores_fn(pid)
            order = np.argpartition(-scores, TOP)[:TOP]
            order = order[np.argsort(-scores[order])]
            sid = shabad[pid]
            rankings.append([int(c) for c in order if shabad[c] != sid and scores[c] > 0])
        return rankings

    results, nowords, kept = [], {}, {}      # kept: label -> rankings, for fusion

    def record(label, rankings):
        results.append(evaluate(rankings, queries, label))
        nowords[label] = no_word_share(rankings)
        kept[label] = rankings

    def run(label, scores_fn):
        print(f"scoring {label} ...")
        record(label, rank_from(scores_fn))

    if args.bm25 or args.bm25_char:
        from rank_bm25 import BM25Okapi
        for flag, tok, label in ((args.bm25, tokenize, "BM25 (word)"),
                                 (args.bm25_char, char_tokenize, "BM25 (char 3-5gram)")):
            if flag:
                print(f"building {label} index ...")
                index = BM25Okapi([tok(p["text"]) for p in padas])
                run(label, lambda pid, index=index, tok=tok: index.get_scores(tok(padas[pid]["text"])))

    for tag in args.linemean:
        path = EMBED_DIR / tag / "embeddings.npy"
        if not path.exists():
            sys.exit(f"Missing {path}.")
        lines = np.load(path)
        vec = np.stack([lines[p["line_index"]].mean(axis=0) for p in padas])
        vec /= np.linalg.norm(vec, axis=1, keepdims=True)
        run(f"line-mean: {tag}", lambda pid, vec=vec: vec @ vec[pid])

    for tag in args.dense:
        path = EMBED_DIR / f"pada_{tag}" / "embeddings.npy"
        if not path.exists():
            sys.exit(f"Missing {path}. Run build_pada_index.py --tag {tag} first.")
        vec = np.load(path)
        if len(vec) != len(padas):
            sys.exit(f"{path} has {len(vec):,} rows but there are {len(padas):,} padas "
                     "(was it built with --limit?). Rebuild it.")
        run(f"pada text: {tag}", lambda pid, vec=vec: vec @ vec[pid])

    for tag in args.fuse:
        lexical, semantic = kept["BM25 (char 3-5gram)"], kept[f"pada text: {tag}"]
        print(f"scoring fused char-BM25 + '{tag}' ...")
        record(f"fused: bm25c + {tag}", [rrf(a, b) for a, b in zip(lexical, semantic)])

    lines = ["=" * 92, "PADA -> PADA RETRIEVAL  (same-Shabad hits excluded)", "=" * 92,
             f"queries: {len(queries):,}    padas searched: {len(padas):,}", ""]
    head = f"{'system':<28}" + "".join(f"{'H@' + str(k):>7}" for k in K_VALUES) + \
           f"{'MRR':>7}{'nDCG10':>8}{'zero@10':>9}{'no-word':>9}"
    lines += [head, "-" * len(head)]
    for r in results:
        lines.append(f"{r['label']:<28}" + "".join(f"{r['hit@' + str(k)]:>7.2f}" for k in K_VALUES)
                     + f"{r['mrr']:>7.2f}{r['ndcg']:>8.2f}{r['zero']:>8.2f}%{nowords[r['label']]:>8.1f}%")
    lines += ["", f"zero@10 : H@10 on the {results[0]['zero_q']} queries whose gold includes a pada "
                  "sharing no word with the query",
              "no-word : share of top-10 results sharing NO word with the query "
              "(higher = less keyword-driven)", "=" * 92]
    report = "\n".join(lines)
    print("\n" + report)
    RESULTS_TXT.write_text(report, encoding="utf-8")
    print(f"\nWrote {RESULTS_TXT}")


if __name__ == "__main__":
    main()
