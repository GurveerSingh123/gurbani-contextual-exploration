"""
Step 3c: does a cross-encoder reranker improve the dense ranking?

The dense model scores each verse on its own vector; a cross-encoder reads the
query and a candidate TOGETHER, which is slower but sees how the two relate.
It is only affordable on a short list, so it reorders the top --candidates
results of a dense tag and leaves the rest of the ranking untouched (so H@50 is
unchanged by construction when --candidates is 50).

What the reranker reads (--mode, repeatable):

  pankati  verse vs verse.  The fair test, but the reranker has never seen Gurbani.
  ft       Faridkot Teeka vs Faridkot Teeka.  Fair: the gold set never used FT.
  ss       Sahib Singh vs Sahib Singh.  OPTIMISTIC: the gold set was mined from
           SS (and MS) commentary similarity, so this partly re-measures the
           benchmark's own criterion. Report it only with that caveat.

Each mode is also scored fused with the dense ranking (reciprocal-rank fusion),
which costs nothing extra and protects against a reranker that is confidently
wrong about Gurmukhi.

  python backend/pipeline/evaluation/evaluate_rerank.py --sample 200 --mode pankati --mode ft --mode ss
  python backend/pipeline/evaluation/evaluate_rerank.py --mode ft          # all 827 queries

Reads  : data/processed/eval_semantic_pairs.jsonl, corpus_pankatis.jsonl
         data/embeddings/<tag>/{embeddings.npy,corpus_meta.jsonl}
Writes : data/processed/rerank_results.txt
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(PROJECT_ROOT / ".env")

PROCESSED = PROJECT_ROOT / os.getenv("PROCESSED_DATA_DIR", "data/processed")
EMBED_DIR = PROJECT_ROOT / os.getenv("EMBEDDING_DIR", "data/embeddings")
EVAL_JSONL = PROCESSED / "eval_semantic_pairs.jsonl"
CORPUS_JSONL = PROCESSED / "corpus_pankatis.jsonl"
RESULTS_TXT = PROCESSED / "rerank_results.txt"
RERANKER = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "50"))
SEED = int(os.getenv("SEED", "42"))

K_VALUES = [1, 5, 10, 50]
NDCG_K = 10
RRF_K = 60                      # standard reciprocal-rank-fusion constant
MAX_LEN = 256                   # Steeks run longer than verses; this covers ~all

MODES = {"pankati": None, "ft": "steek_ft", "ss": "steek_ss"}


def evaluate(rankings, queries, corpus, label: str):
    """Same metrics as evaluate.py. rankings[qi] -> corpus indices, best first."""
    max_k = max(K_VALUES)
    hits = {k: 0 for k in K_VALUES}
    rr_sum = 0.0
    ndcg_sum = 0.0
    zero_overlap_hits = 0
    zero_overlap_total = 0

    discounts = 1.0 / np.log2(np.arange(2, NDCG_K + 2))

    for qi, q in enumerate(queries):
        gold_ids = {g["id"] for g in q["gold"]}
        gold_zero = {g["id"] for g in q["gold"] if g["word_overlap"] == 0.0}
        ranked = rankings[qi][:max_k]
        # A corpus doc can carry several ids (repeated scripture) -- any match counts.
        found = [bool(set(corpus[c]["ids"]) & gold_ids) for c in ranked]

        for k in K_VALUES:
            if any(found[:k]):
                hits[k] += 1

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
        "mrr@50": 100 * rr_sum / n,
        f"ndcg@{NDCG_K}": 100 * ndcg_sum / n,
        "zero_overlap_hit@10": 100 * zero_overlap_hits / max(zero_overlap_total, 1),
        "zero_overlap_queries": zero_overlap_total,
    }


def rrf(first: list, second: list) -> list:
    """Reciprocal-rank fusion of two orderings of the same candidates."""
    pos1 = {c: r for r, c in enumerate(first)}
    pos2 = {c: r for r, c in enumerate(second)}
    return sorted(first, key=lambda c: -(1 / (RRF_K + pos1[c]) + 1 / (RRF_K + pos2[c])))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default=os.getenv("SEARCH_TAG", "verse_hn"),
                    help="dense tag whose ranking is reranked")
    ap.add_argument("--mode", action="append", choices=list(MODES), default=[],
                    help="what the reranker reads (repeatable)")
    ap.add_argument("--candidates", type=int, default=RETRIEVAL_K,
                    help="how many dense results to rerank")
    ap.add_argument("--sample", type=int, default=0,
                    help="score a fixed random subset of queries (0 = all)")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--out", default=str(RESULTS_TXT), help="report file")
    args = ap.parse_args()
    modes = args.mode or ["pankati"]

    for p in (EVAL_JSONL, CORPUS_JSONL):
        if not p.exists():
            sys.exit(f"Missing {p}.")
    queries = [json.loads(l) for l in EVAL_JSONL.open(encoding="utf-8")]
    if args.sample and args.sample < len(queries):
        keep = sorted(random.Random(SEED).sample(range(len(queries)), args.sample))
        queries = [queries[i] for i in keep]
    print(f"eval queries: {len(queries):,}" + (" (fixed sample)" if args.sample else ""))

    meta_path = EMBED_DIR / args.tag / "corpus_meta.jsonl"
    emb_path = EMBED_DIR / args.tag / "embeddings.npy"
    for p in (meta_path, emb_path):
        if not p.exists():
            sys.exit(f"Missing {p}. Run build_index.py --tag {args.tag} first.")
    corpus = [json.loads(l) for l in meta_path.open(encoding="utf-8")]
    docs = [json.loads(l) for l in CORPUS_JSONL.open(encoding="utf-8")]
    if len(docs) != len(corpus) or docs[0]["pankati"] != corpus[0]["pankati"]:
        sys.exit("corpus order mismatch -- rebuild the index.")
    emb = np.load(emb_path)

    by_text = {c["pankati"]: i for i, c in enumerate(corpus)}
    q_idx = [by_text.get(q["query_pankati"], -1) for q in queries]
    shabad_of = [set(c["shabad_ids"]) for c in corpus]

    def dense_rank(qi):
        ci = q_idx[qi]
        if ci < 0:
            return []
        scores = emb @ emb[ci]
        order = np.argpartition(-scores, 400)[:400]
        order = order[np.argsort(-scores[order])]
        shabad = queries[qi]["query_shabad_id"]
        return [int(c) for c in order if c != ci and shabad not in shabad_of[c]]

    base = [dense_rank(qi) for qi in range(len(queries))]
    results = [evaluate(base, queries, corpus, f"dense: {args.tag}")]

    def write_report():
        """Rewritten after every mode, so a crash keeps the modes already scored."""
        lines = []
        add = lines.append
        add("=" * 78)
        add(f"RERANKING THE TOP {args.candidates} OF dense:{args.tag}  (same-Shabad hits excluded)")
        add("=" * 78)
        add(f"queries: {len(queries):,}{' (fixed sample, seed ' + str(SEED) + ')' if args.sample else ''}"
            f"    reranker: {RERANKER}")
        add("")
        header = f"{'system':<22}" + "".join(f"{'H@' + str(k):>8}" for k in K_VALUES) + \
                 f"{'MRR':>8}{'nDCG@10':>9}"
        add(header)
        add("-" * len(header))
        for r in results:
            row = f"{r['label']:<22}" + "".join(f"{r['hit@' + str(k)]:>8.2f}" for k in K_VALUES)
            row += f"{r['mrr@50']:>8.2f}{r['ndcg@10']:>9.2f}"
            add(row)
        add("")
        add("pankati / ft : fair on this gold set.")
        add("ss           : OPTIMISTIC -- the gold set was mined from SS and MS commentary similarity.")
        add("")
        add("-- zero-word-overlap subset -----------------------------------------------")
        add(f"queries whose gold includes a ZERO-word-overlap verse: {results[0]['zero_overlap_queries']:,}")
        for r in results:
            add(f"  {r['label']:<22} hit@10 on those: {r['zero_overlap_hit@10']:>6.2f}%")
        add("=" * 78)

        report = "\n".join(lines)
        Path(args.out).write_text(report, encoding="utf-8")
        return report

    from sentence_transformers import CrossEncoder
    print(f"loading {RERANKER} ...")
    model = CrossEncoder(RERANKER, max_length=MAX_LEN, device="cpu")

    for mode in modes:
        col = MODES[mode]
        fallbacks = 0

        def text(i):
            nonlocal fallbacks
            if col is None:
                return docs[i]["pankati"]
            t = docs[i]["occurrences"][0].get(col) or ""
            if not t.strip():
                fallbacks += 1
                return docs[i]["pankati"]       # no commentary: compare verses
            return t

        reranked, fused = [], []
        started = time.time()
        for qi in range(len(queries)):
            ranked = base[qi]
            top, rest = ranked[:args.candidates], ranked[args.candidates:]
            if not top:
                reranked.append(ranked)
                fused.append(ranked)
                continue
            q = text(q_idx[qi])
            scores = model.predict([(q, text(c)) for c in top],
                                   batch_size=args.batch_size, show_progress_bar=False)
            by_ce = [top[j] for j in np.argsort(-np.asarray(scores))]
            reranked.append(by_ce + rest)
            fused.append(rrf(top, by_ce) + rest)

            done = qi + 1
            if done % 20 == 0 or done == len(queries):
                el = time.time() - started
                print(f"  [{mode}] {done}/{len(queries)}  "
                      f"elapsed {el / 60:.1f}m  eta {el / done * (len(queries) - done) / 60:.1f}m",
                      flush=True)

        if fallbacks:
            print(f"  [{mode}] {fallbacks} candidates had no {col}; used the verse instead")
        results.append(evaluate(reranked, queries, corpus, f"rerank: {mode}"))
        results.append(evaluate(fused, queries, corpus, f"rerank+fuse: {mode}"))
        write_report()
        print(f"  [{mode}] scored; report updated", flush=True)

    print("\n" + write_report())
    print(f"\nWrote {args.out}")

if __name__ == "__main__":
    main()
