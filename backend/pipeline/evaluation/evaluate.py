"""
Step 3b (part 2): score retrievers on the Pankati -> Pankati gold set.

Compares BM25 (keyword) against dense embeddings. BM25 is the control: the
project's premise is that keyword search cannot find verses that share no
words, and this measures that instead of asserting it.

  python backend/pipeline/evaluation/evaluate.py --bm25
  python backend/pipeline/evaluation/evaluate.py --bm25 --dense baseline
  python backend/pipeline/evaluation/evaluate.py --dense baseline --dense finetuned
  python backend/pipeline/evaluation/evaluate.py --bm25-char --dense verse_hn --fuse verse_hn

--fuse TAG also scores a HYBRID: char-BM25 and dense TAG merged by reciprocal-
rank fusion (each candidate scores 1/(60 + rank) in each list, summed). The two
systems return mostly different verses, so the merge can beat both. k = 60 is
the standard constant and is NOT tuned here -- tuning it on the test set would
flatter the result.

Same-Shabad hits are excluded from results before scoring, because the gold set
is cross-Shabad and the product deliberately surfaces verses from elsewhere in
the Granth.

Reads  : data/processed/eval_semantic_pairs.jsonl
         data/embeddings/<tag>/{embeddings.npy,corpus_meta.jsonl}
Writes : data/processed/baseline_results.txt   (fusion_results.txt with --fuse)
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
EVAL_JSONL = PROCESSED / "eval_semantic_pairs.jsonl"
RESULTS_TXT = PROCESSED / "baseline_results.txt"

K_VALUES = [1, 5, 10, 50]
NDCG_K = 10
RRF_K = 60            # standard reciprocal-rank-fusion constant; deliberately not tuned
_STRIP = re.compile(r"[॥।0-9੦-੯]")


def tokenize(text: str):
    return [w for w in _STRIP.sub(" ", text).split() if len(w) > 1]


CHAR_N = (3, 4, 5)


def char_tokenize(text: str):
    """Character n-grams, so Gurmukhi spelling variants still match.

    Word-level BM25 misses ਭਵਜਲੁ / ਭਉਜਲ and ਉਤਾਰਣਹਾਰਾ / ਤਾਰਣਹਾਰੁ, which are the
    same words spelled differently. Scoring only word-BM25 would understate the
    lexical baseline and flatter the dense model.
    """
    cleaned = "_".join(tokenize(text))
    return [cleaned[i:i + n]
            for n in CHAR_N
            for i in range(len(cleaned) - n + 1)]


def rrf(first: list, second: list) -> list:
    """Reciprocal-rank fusion of two rankings (a missing item contributes 0)."""
    score = {}
    for ranking in (first, second):
        for r, c in enumerate(ranking):
            score[c] = score.get(c, 0.0) + 1.0 / (RRF_K + r + 1)
    return sorted(score, key=lambda c: -score[c])


def memo(rank_fn):
    """Cache a ranking per query, so fusion reuses it instead of recomputing BM25."""
    cache = {}

    def wrapped(qi):
        if qi not in cache:
            cache[qi] = rank_fn(qi)
        return cache[qi]
    return wrapped


def load_corpus(tag: str):
    meta_path = EMBED_DIR / tag / "corpus_meta.jsonl"
    if not meta_path.exists():
        sys.exit(f"Missing {meta_path}. Run build_index.py --tag {tag} first.")
    return [json.loads(line) for line in meta_path.open(encoding="utf-8")]


def evaluate(rank_fn, queries, corpus, label: str):
    """rank_fn(query_index) -> corpus indices, best first, same-Shabad removed."""
    max_k = max(K_VALUES)
    hits = {k: 0 for k in K_VALUES}
    recall_sum = {k: 0.0 for k in K_VALUES}
    rr_sum = 0.0
    ndcg_sum = 0.0
    zero_overlap_hits = 0
    zero_overlap_total = 0

    discounts = 1.0 / np.log2(np.arange(2, NDCG_K + 2))

    for qi, q in enumerate(queries):
        gold_ids = {g["id"] for g in q["gold"]}
        gold_zero = {g["id"] for g in q["gold"] if g["word_overlap"] == 0.0}
        ranked = rank_fn(qi)[:max_k]
        # A corpus doc can carry several ids (repeated scripture) -- any match counts.
        found = [bool(set(corpus[c]["ids"]) & gold_ids) for c in ranked]

        for k in K_VALUES:
            top = found[:k]
            if any(top):
                hits[k] += 1
            n_found = sum(
                1 for c in ranked[:k] if set(corpus[c]["ids"]) & gold_ids
            )
            recall_sum[k] += n_found / len(gold_ids)

        rr_sum += next((1.0 / (i + 1) for i, f in enumerate(found) if f), 0.0)

        gains = np.array([1.0 if f else 0.0 for f in found[:NDCG_K]])
        if gains.size < NDCG_K:
            gains = np.pad(gains, (0, NDCG_K - gains.size))
        ideal = np.zeros(NDCG_K)
        ideal[:min(len(gold_ids), NDCG_K)] = 1.0
        denom = (ideal * discounts).sum()
        ndcg_sum += (gains * discounts).sum() / denom if denom else 0.0

        if gold_zero:
            zero_overlap_total += 1
            if any(set(corpus[c]["ids"]) & gold_zero for c in ranked[:10]):
                zero_overlap_hits += 1

    n = len(queries)
    return {
        "label": label,
        **{f"hit@{k}": 100 * hits[k] / n for k in K_VALUES},
        **{f"recall@{k}": 100 * recall_sum[k] / n for k in K_VALUES},
        "mrr@50": 100 * rr_sum / n,
        f"ndcg@{NDCG_K}": 100 * ndcg_sum / n,
        "zero_overlap_hit@10": 100 * zero_overlap_hits / max(zero_overlap_total, 1),
        "zero_overlap_queries": zero_overlap_total,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bm25", action="store_true", help="word-level BM25")
    ap.add_argument("--bm25-char", action="store_true",
                    help="character 3-5gram BM25 (robust to spelling variants)")
    ap.add_argument("--dense", action="append", default=[],
                    help="embedding tag under data/embeddings/ (repeatable)")
    ap.add_argument("--meta-tag", default=None,
                    help="which tag's corpus_meta to use for BM25 (default: first dense, else 'baseline')")
    ap.add_argument("--fuse", action="append", default=[],
                    help="dense tag to merge with char-BM25 by reciprocal-rank fusion (repeatable)")
    ap.add_argument("--sample", type=int, default=0,
                    help="score only the first N queries (quick test; numbers not comparable)")
    ap.add_argument("--out", default=None,
                    help="report file (default baseline_results.txt; fusion_results.txt with --fuse, "
                         "so a fusion run never overwrites the full results table)")
    args = ap.parse_args()
    out_path = Path(args.out) if args.out else (
        PROCESSED / "fusion_results.txt" if args.fuse else RESULTS_TXT)
    if args.fuse:                       # a hybrid needs both of its halves scored
        args.bm25_char = True
        args.dense += [t for t in args.fuse if t not in args.dense]
    if not (args.bm25 or args.bm25_char or args.dense):
        ap.error("pick at least one of --bm25 / --bm25-char / --dense")

    if not EVAL_JSONL.exists():
        sys.exit(f"Missing {EVAL_JSONL}. Run build_eval_sets.py first.")
    queries = [json.loads(line) for line in EVAL_JSONL.open(encoding="utf-8")]
    if args.sample:
        queries = queries[:args.sample]
    print(f"eval queries: {len(queries):,}")

    meta_tag = args.meta_tag or (args.dense[0] if args.dense else "baseline")
    corpus = load_corpus(meta_tag)
    print(f"corpus: {len(corpus):,} unique Pankatis (meta from '{meta_tag}')")

    # Query row -> corpus row, and the Shabads to suppress for that query.
    by_text = {c["pankati"]: i for i, c in enumerate(corpus)}
    q_corpus_idx = [by_text.get(q["query_pankati"], -1) for q in queries]
    missing = sum(1 for i in q_corpus_idx if i < 0)
    if missing:
        print(f"WARNING: {missing} queries not found in corpus")

    def blocked(qi):
        """Corpus rows to drop: the query itself and anything in its Shabad."""
        shabad = queries[qi]["query_shabad_id"]
        out = {q_corpus_idx[qi]} if q_corpus_idx[qi] >= 0 else set()
        return out, shabad

    results = []
    rank_fns = {}                       # label -> cached ranking function, for fusion

    shabad_of = [set(c["shabad_ids"]) for c in corpus]

    if args.bm25 or args.bm25_char:
        from rank_bm25 import BM25Okapi

    def make_bm25(tok_fn, label):
        print(f"building {label} ...")
        index = BM25Okapi([tok_fn(c["pankati"]) for c in corpus])

        def rank(qi):
            scores = index.get_scores(tok_fn(queries[qi]["query_pankati"]))
            drop, shabad = blocked(qi)
            order = np.argsort(-scores)[:400]
            return [c for c in order
                    if scores[c] > 0 and c not in drop and shabad not in shabad_of[c]]

        rank = memo(rank)
        rank_fns[label] = rank
        print(f"scoring {label} ...")
        return evaluate(rank, queries, corpus, label)

    if args.bm25:
        results.append(make_bm25(tokenize, "BM25 (word)"))
    if args.bm25_char:
        results.append(make_bm25(char_tokenize, "BM25 (char 3-5gram)"))

    for tag in args.dense:
        emb_path = EMBED_DIR / tag / "embeddings.npy"
        if not emb_path.exists():
            sys.exit(f"Missing {emb_path}. Run build_index.py --tag {tag} first.")
        emb = np.load(emb_path)
        print(f"dense '{tag}': {emb.shape}")

        def dense_rank(qi, emb=emb):
            ci = q_corpus_idx[qi]
            if ci < 0:
                return []
            scores = emb @ emb[ci]
            drop, shabad = blocked(qi)
            order = np.argpartition(-scores, 400)[:400]
            order = order[np.argsort(-scores[order])]
            return [c for c in order if c not in drop and shabad not in shabad_of[c]]

        dense_rank = memo(dense_rank)
        rank_fns[f"dense: {tag}"] = dense_rank
        print(f"scoring dense '{tag}' ...")
        results.append(evaluate(dense_rank, queries, corpus, f"dense: {tag}"))

    for tag in args.fuse:
        lexical, semantic = rank_fns["BM25 (char 3-5gram)"], rank_fns[f"dense: {tag}"]
        print(f"scoring fused char-BM25 + '{tag}' ...")
        results.append(evaluate(lambda qi, a=lexical, b=semantic: rrf(a(qi), b(qi)),
                                queries, corpus, f"fused: bm25c+{tag}"))

    # --- report -----------------------------------------------------------
    lines = []
    add = lines.append
    add("=" * 78)
    add("PANKATI -> PANKATI RETRIEVAL  (same-Shabad hits excluded)")
    add("=" * 78)
    add(f"queries: {len(queries):,}    corpus: {len(corpus):,}")
    add("")
    header = f"{'model':<22}" + "".join(f"{'H@' + str(k):>8}" for k in K_VALUES) + \
             f"{'MRR':>8}{'nDCG@10':>9}"
    add(header)
    add("-" * len(header))
    for r in results:
        row = f"{r['label']:<22}" + "".join(f"{r['hit@' + str(k)]:>8.2f}" for k in K_VALUES)
        row += f"{r['mrr@50']:>8.2f}{r['ndcg@10']:>9.2f}"
        add(row)
    add("")
    add("H@k = % of queries with at least one gold verse in the top k")
    add("")
    add("-- the subset keyword search cannot reach ---------------------------------")
    add(f"queries whose gold includes a ZERO-word-overlap verse: "
        f"{results[0]['zero_overlap_queries']:,}")
    for r in results:
        add(f"  {r['label']:<22} hit@10 on those: {r['zero_overlap_hit@10']:>6.2f}%")
    add("=" * 78)

    report = "\n".join(lines)
    print("\n" + report)
    out_path.write_text(report, encoding="utf-8")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
