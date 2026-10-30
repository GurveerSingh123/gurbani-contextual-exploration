"""
Does context help? Score Pankati vectors blended with their pada / Shabad.

A single Pankati is often half a thought; the complete unit is the pada (the
lines up to a ॥੧॥-style marker, usually 2-4) and the Shabad has one central
idea. Read alone, a line leans on its words. This tests whether adding context
finds meaning better -- and in particular whether it reduces the keyword bias.

No retraining: a pada or Shabad vector is the mean of its lines' cached vectors,
so every variant is scored in minutes on CPU.

    v' = normalise( v_line + a * mean(pada) + b * mean(Shabad) )

Queries and corpus get the same transform. Same metrics and same-Shabad
exclusion as evaluate.py, plus the share of top-10 results that share no word
with the query (the keyword-bias measure).

  python backend/pipeline/evaluation/evaluate_context.py
  python backend/pipeline/evaluation/evaluate_context.py --tag baseline

Reads  : data/processed/eval_semantic_pairs.jsonl, corpus_pankatis.jsonl
         data/embeddings/<tag>/{embeddings.npy,corpus_meta.jsonl}
Writes : data/processed/context_results.txt
"""

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(PROJECT_ROOT / ".env")

PROCESSED = PROJECT_ROOT / os.getenv("PROCESSED_DATA_DIR", "data/processed")
EMBED_DIR = PROJECT_ROOT / os.getenv("EMBEDDING_DIR", "data/embeddings")
EVAL_JSONL = PROCESSED / "eval_semantic_pairs.jsonl"
CORPUS_JSONL = PROCESSED / "corpus_pankatis.jsonl"
RESULTS_TXT = PROCESSED / "context_results.txt"

K_VALUES = [1, 5, 10, 50]
NDCG_K = 10
PADA_END = re.compile(r"॥\s*[੦-੯0-9]+\s*॥|ਰਹਾਉ")   # ॥੧॥ / ॥੨॥ / Rahao close a pada
_STRIP = re.compile(r"[॥।0-9੦-੯]")

# (label, weight on pada mean, weight on Shabad mean, keep the line itself?)
VARIANTS = [
    ("line only (current)", 0.0, 0.0, True),
    ("line + 0.5 pada", 0.5, 0.0, True),
    ("line + 1.0 pada", 1.0, 0.0, True),
    ("pada only", 1.0, 0.0, False),
    ("line + 0.5 Shabad", 0.0, 0.5, True),
    ("line + 0.5 pada + 0.25 Shabad", 0.5, 0.25, True),
    ("Shabad only", 0.0, 1.0, False),
]


def words(text: str):
    return {w for w in _STRIP.sub(" ", text).split() if len(w) > 1}


def evaluate(rankings, queries, corpus, label):
    """Same metrics as evaluate.py. rankings[qi] -> corpus indices, best first."""
    hits = {k: 0 for k in K_VALUES}
    rr_sum = ndcg_sum = 0.0
    zero_hits = zero_total = 0
    discounts = 1.0 / np.log2(np.arange(2, NDCG_K + 2))
    for qi, q in enumerate(queries):
        gold = {g["id"] for g in q["gold"]}
        gold_zero = {g["id"] for g in q["gold"] if g["word_overlap"] == 0.0}
        ranked = rankings[qi][:max(K_VALUES)]
        found = [bool(set(corpus[c]["ids"]) & gold) for c in ranked]
        for k in K_VALUES:
            hits[k] += any(found[:k])
        rr_sum += next((1.0 / (i + 1) for i, f in enumerate(found) if f), 0.0)
        gains = np.array([1.0 if f else 0.0 for f in found[:NDCG_K]])
        gains = np.pad(gains, (0, NDCG_K - gains.size))
        ideal = np.zeros(NDCG_K)
        ideal[:min(len(gold), NDCG_K)] = 1.0
        denom = (ideal * discounts).sum()
        ndcg_sum += (gains * discounts).sum() / denom if denom else 0.0
        if gold_zero:
            zero_total += 1
            zero_hits += any(set(corpus[c]["ids"]) & gold_zero for c in ranked[:10])
    n = len(queries)
    return {"label": label, **{f"hit@{k}": 100 * hits[k] / n for k in K_VALUES},
            "mrr": 100 * rr_sum / n, "ndcg@10": 100 * ndcg_sum / n,
            "zero@10": 100 * zero_hits / max(zero_total, 1), "zero_q": zero_total}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default=os.getenv("SEARCH_TAG", "verse_hn"))
    args = ap.parse_args()

    for p in (EVAL_JSONL, CORPUS_JSONL, EMBED_DIR / args.tag / "embeddings.npy"):
        if not p.exists():
            sys.exit(f"Missing {p}.")
    queries = [json.loads(l) for l in EVAL_JSONL.open(encoding="utf-8")]
    docs = [json.loads(l) for l in CORPUS_JSONL.open(encoding="utf-8")]
    corpus = [json.loads(l) for l in (EMBED_DIR / args.tag / "corpus_meta.jsonl").open(encoding="utf-8")]
    if len(docs) != len(corpus) or docs[0]["pankati"] != corpus[0]["pankati"]:
        sys.exit("corpus order mismatch -- rebuild the index.")
    emb = np.load(EMBED_DIR / args.tag / "embeddings.npy")
    print(f"tag '{args.tag}': {emb.shape}, {len(queries)} queries")

    # --- context groups, from each verse's first occurrence -------------------
    lines_of = defaultdict(list)                  # shabad -> [(verse_id, corpus idx)]
    for i, d in enumerate(docs):
        for o in d["occurrences"]:
            lines_of[o["shabad_id"]].append((o["verse_id"], i))
    pada_of, shabad_of_line = {}, {}              # (shabad, idx) -> member list
    for sid, lines in lines_of.items():
        ordered = [i for _, i in sorted(lines)]
        group = []
        for i in ordered:
            group.append(i)
            if PADA_END.search(docs[i]["pankati"]):
                for m in group:
                    pada_of[(sid, m)] = list(group)
                group = []
        for m in group:                           # trailing lines without a marker
            pada_of[(sid, m)] = list(group)
        for m in ordered:
            shabad_of_line[(sid, m)] = ordered

    first_sid = [d["occurrences"][0]["shabad_id"] for d in docs]
    pada_mean = np.zeros_like(emb)
    shabad_mean = np.zeros_like(emb)
    cache = {}
    for i, sid in enumerate(first_sid):
        p = tuple(pada_of[(sid, i)])
        if p not in cache:
            cache[p] = emb[list(p)].mean(axis=0)
        pada_mean[i] = cache[p]
        s = ("S", sid)
        if s not in cache:
            cache[s] = emb[shabad_of_line[(sid, i)]].mean(axis=0)
        shabad_mean[i] = cache[s]
    sizes = [len(pada_of[(first_sid[i], i)]) for i in range(len(docs))]
    print(f"pada size: mean {np.mean(sizes):.1f} lines, median {int(np.median(sizes))}")

    # --- score each variant -----------------------------------------------------
    by_text = {c["pankati"]: i for i, c in enumerate(corpus)}
    q_idx = [by_text.get(q["query_pankati"], -1) for q in queries]
    shabads = [set(c["shabad_ids"]) for c in corpus]
    word_sets = [words(d["pankati"]) for d in docs]

    results, nooverlap = [], {}
    for label, a, b, keep_line in VARIANTS:
        vec = (emb if keep_line else 0) + a * pada_mean + b * shabad_mean
        vec = vec / np.linalg.norm(vec, axis=1, keepdims=True)
        rankings, zero_share = [], []
        for qi, q in enumerate(queries):
            ci = q_idx[qi]
            if ci < 0:
                rankings.append([])
                continue
            scores = vec @ vec[ci]
            order = np.argpartition(-scores, 400)[:400]
            order = order[np.argsort(-scores[order])]
            sid = q["query_shabad_id"]
            ranked = [int(c) for c in order if c != ci and sid not in shabads[c]]
            rankings.append(ranked)
            zero_share.extend(not (word_sets[ci] & word_sets[c]) for c in ranked[:10])
        results.append(evaluate(rankings, queries, corpus, label))
        nooverlap[label] = 100 * np.mean(zero_share)
        print(f"  scored: {label}")

    # --- report -----------------------------------------------------------------
    lines = ["=" * 92, f"CONTEXT EXPERIMENT on dense:{args.tag}  (same-Shabad hits excluded)", "=" * 92,
             f"queries: {len(queries):,}    v' = line + a*pada_mean + b*Shabad_mean, re-normalised", ""]
    head = f"{'variant':<32}" + "".join(f"{'H@' + str(k):>7}" for k in K_VALUES) + \
           f"{'MRR':>7}{'nDCG10':>8}{'zero@10':>9}{'no-word':>9}"
    lines += [head, "-" * len(head)]
    for r in results:
        lines.append(f"{r['label']:<32}" + "".join(f"{r['hit@' + str(k)]:>7.2f}" for k in K_VALUES)
                     + f"{r['mrr']:>7.2f}{r['ndcg@10']:>8.2f}{r['zero@10']:>8.2f}%"
                     f"{nooverlap[r['label']]:>8.1f}%")
    lines += ["", f"zero@10 : H@10 on the {results[0]['zero_q']} queries whose gold includes a "
                  "zero-word-overlap verse",
              "no-word : share of the top-10 results that share NO word with the query "
              "(higher = less keyword-driven)", "=" * 92]
    report = "\n".join(lines)
    print("\n" + report)
    RESULTS_TXT.write_text(report, encoding="utf-8")
    print(f"\nWrote {RESULTS_TXT}")


if __name__ == "__main__":
    main()
