"""
Pada step 2: build a pada -> pada gold set.

The line-level gold set asks "did we find the known matching LINE?", which
cannot judge a system that matches whole padas. This applies the same rule one
level up: two padas are a gold pair when BOTH independent commentators (Sahib
Singh and Manmohan Singh) explain them alike -- comparing each pada's combined
Steeks instead of a single line's.

Same safeguards as build_eval_sets.py: Gurmukhi-safe tokenisation, speaker
formulas ("ਕਬੀਰ ਆਖਦਾ ਹੈ") removed, cross-Shabad only, a ceiling to drop
verbatim-reused commentary, and identical pada text never counted as a match.
Queries come from the TEST split only.

The line-level threshold (0.30) was tuned for single lines; a pada's combined
commentary is longer, so the threshold must be chosen again. Run --sweep first:
it reports how many pairs each threshold yields and writes sample pairs per
threshold band to a text file to read. Then build with the chosen value.

  python backend/pipeline/evaluation/build_pada_eval.py --sweep
  python backend/pipeline/evaluation/build_pada_eval.py --threshold 0.30

Reads  : data/processed/padas.jsonl
Writes : data/processed/pada_eval_pairs.jsonl, pada_eval_report.txt
         data/processed/pada_threshold_samples.txt      (--sweep)
"""

import argparse
import json
import os
import random
import re
import sys
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

PROJECT_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(PROJECT_ROOT / ".env")

PROCESSED = PROJECT_ROOT / os.getenv("PROCESSED_DATA_DIR", "data/processed")
PADAS_JSONL = PROCESSED / "padas.jsonl"
EVAL_JSONL = PROCESSED / "pada_eval_pairs.jsonl"
REPORT_TXT = PROCESSED / "pada_eval_report.txt"
SAMPLES_TXT = PROCESSED / "pada_threshold_samples.txt"

THRESHOLD = float(os.getenv("PADA_GOLD_THRESHOLD", "0.30"))
CEILING = float(os.getenv("GOLD_CEILING", "0.97"))
MAX_GOLD_PER_QUERY = int(os.getenv("MAX_GOLD_PER_QUERY", "10"))
EVAL_SPLIT = os.getenv("EVAL_SPLIT", "test")
SEED = int(os.getenv("SEED", "42"))
CHUNK = 400
# A pada's combined commentary is 2-4x longer than a line's, so tf-idf cosines run
# lower than at line level; the sweep therefore starts well below the line's 0.30.
SWEEP = [0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30]

# --- identical to build_eval_sets.py; keep in step --------------------------------
_STRIP = re.compile(r"[॥।0-9੦-੯]")


def tokens(text: str) -> set:
    return {w for w in _STRIP.sub(" ", text).split() if len(w) > 1}


def jaccard(a: str, b: str) -> float:
    ta, tb = tokens(a), tokens(b)
    return len(ta & tb) / len(ta | tb) if ta and tb else 0.0


_SPEAKER_FORMULAS = set("""
ਕਬੀਰ ਕਬੀਰੁ ਨਾਨਕ ਨਾਨਕੁ ਫਰੀਦ ਰਵਿਦਾਸ ਨਾਮਦੇਵ ਤ੍ਰਿਲੋਚਨ ਧੰਨਾ ਬੇਣੀ ਸੈਣ ਪੀਪਾ ਜੈਦੇਵ
ਭੀਖਨ ਸੂਰਦਾਸ ਆਖਦਾ ਆਖਦੇ ਆਖਦੀ ਕਹਿੰਦਾ ਕਹਿੰਦੇ ਆਖ ਕਹਿ ਭਾਈ ਹੇ ਜੀ ਜੀਉ ਸੰਤ ਜਨੋ
ਭਰਾਵੋ ਮੈਂ ਹਾਂ ਹੈ ਹਨ
""".split())
_EDGE_PUNCT = "।॥,—-!?()"


def drop_speaker_formulas(text: str) -> str:
    return " ".join(w for w in text.split() if w.strip(_EDGE_PUNCT) not in _SPEAKER_FORMULAS)


# Never the scikit-learn default \b\w\w+\b: it destroys Gurmukhi (see CLAUDE.md).
TOKEN_PATTERN = r"[^\s॥।,—()!?]+"


def tfidf(texts):
    cleaned = [drop_speaker_formulas(t) for t in texts]
    vec = TfidfVectorizer(analyzer="word", token_pattern=TOKEN_PATTERN,
                          ngram_range=(1, 2), min_df=3, max_df=0.30, sublinear_tf=True)
    return normalize(vec.fit_transform(cleaned)).astype(np.float32)
# ----------------------------------------------------------------------------------


def candidate_pairs(pool, Xs, Xm, query_idx, floor):
    """Yield (query, candidate, ss_sim, ms_sim, combined) for combined >= floor."""
    shabad = np.array([p["shabad_id"] for p in pool])
    text = [p["text"] for p in pool]
    for start in range(0, len(query_idx), CHUNK):
        block = query_idx[start:start + CHUNK]
        sim_ss = (Xs[block] @ Xs.T).toarray()
        sim_ms = (Xm[block] @ Xm.T).toarray()
        comb = np.minimum(sim_ss, sim_ms)               # both commentators must agree
        for r, qi in enumerate(block):
            scores = comb[r]
            scores[shabad == shabad[qi]] = -1.0         # cross-Shabad only
            for j in np.flatnonzero((scores >= floor) & (scores <= CEILING)):
                if text[j] == text[qi]:                 # same pada text elsewhere: trivial
                    continue
                yield qi, int(j), float(sim_ss[r, j]), float(sim_ms[r, j]), float(scores[j])
        print(f"  {min(start + CHUNK, len(query_idx)):,}/{len(query_idx):,} query padas", end="\r")
    print()


def sweep(pool, Xs, Xm, query_idx):
    pairs = list(candidate_pairs(pool, Xs, Xm, query_idx, min(SWEEP)))
    rng = random.Random(SEED)
    out = ["PADA GOLD THRESHOLD SWEEP", "=" * 70,
           "A pair counts when BOTH commentators' combined Steeks reach the threshold.",
           "Read the samples in each band and judge: do these padas carry the same teaching?",
           ""]
    print(f"\n{'threshold':>10}{'pairs':>9}{'queries':>9}{'zero-word pairs':>17}")
    for t in SWEEP:
        sel = [p for p in pairs if p[4] >= t]
        q = len({p[0] for p in sel})
        zero = sum(1 for p in sel if jaccard(pool[p[0]]["text"], pool[p[1]]["text"]) == 0)
        line = f"{t:>10.2f}{len(sel):>9,}{q:>9,}{zero:>11,} ({100 * zero / max(len(sel), 1):.0f}%)"
        print(line)
        out.append(line)
    out.append("")
    bands = list(zip(SWEEP, SWEEP[1:] + [1.01]))
    for lo, hi in bands:
        band = [p for p in pairs if lo <= p[4] < hi]
        out += ["", "#" * 70, f"BAND {lo:.2f} - {hi:.2f}   ({len(band):,} pairs; 12 random samples)", "#" * 70]
        for qi, j, s, m, c in rng.sample(band, min(12, len(band))):
            a, b = pool[qi], pool[j]
            out += ["", f"score {c:.3f}  (Sahib Singh {s:.3f}, Manmohan Singh {m:.3f})",
                    f"  PADA A  Ang {a['ang']}: {a['text']}",
                    f"    SS: {a['steek_ss'][:300]}",
                    f"  PADA B  Ang {b['ang']}: {b['text']}",
                    f"    SS: {b['steek_ss'][:300]}"]
    SAMPLES_TXT.write_text("\n".join(out), encoding="utf-8")
    print(f"\nWrote {SAMPLES_TXT}")
    print("Read it, choose a threshold, then run:  python backend/pipeline/evaluation/build_pada_eval.py --threshold <value>")


def build(pool, Xs, Xm, query_idx, threshold):
    by_query = {}
    for qi, j, s, m, c in candidate_pairs(pool, Xs, Xm, query_idx, threshold):
        by_query.setdefault(qi, []).append((c, j, s, m))

    records, n_gold = [], 0
    for qi in sorted(by_query):
        q = pool[qi]
        best = sorted(by_query[qi], reverse=True)[:MAX_GOLD_PER_QUERY]
        gold = [{"pada_id": pool[j]["pada_id"], "text": pool[j]["text"], "ang": pool[j]["ang"],
                 "shabad_id": pool[j]["shabad_id"], "raag_id": pool[j]["raag_id"],
                 "ss_sim": round(s, 4), "ms_sim": round(m, 4),
                 "word_overlap": round(jaccard(q["text"], pool[j]["text"]), 4)}
                for c, j, s, m in best]
        n_gold += len(gold)
        records.append({"query_pada_id": q["pada_id"], "query_text": q["text"],
                        "query_ang": q["ang"], "query_shabad_id": q["shabad_id"],
                        "query_raag_id": q["raag_id"], "gold": gold})

    with EVAL_JSONL.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    overlaps = np.array([g["word_overlap"] for r in records for g in r["gold"]] or [0.0])
    per_q = np.array([len(r["gold"]) for r in records] or [0])
    gaps = np.array([abs(g["ang"] - r["query_ang"]) for r in records for g in r["gold"]] or [0])
    diff_raag = sum(1 for r in records for g in r["gold"] if g["raag_id"] != r["query_raag_id"])
    lines = ["=" * 62, "PADA EVALUATION SET  (pada -> pada)", "=" * 62,
             f"Gold rule   : min(ss_sim, ms_sim) of combined pada Steeks in [{threshold}, {CEILING}]",
             "Constraint  : cross-Shabad only; identical pada text excluded",
             f"Query split : {EVAL_SPLIT}", "",
             f"Query padas offered : {len(query_idx):,}",
             f"Queries with gold   : {len(records):,} ({100 * len(records) / max(len(query_idx), 1):.1f}%)",
             f"Total gold pairs    : {n_gold:,}",
             f"Gold per query      : mean {per_q.mean():.2f}  median {int(np.median(per_q))}  max {int(per_q.max())}",
             "",
             f"gold pairs with ZERO shared words : {int((overlaps == 0).sum()):,} ({100 * (overlaps == 0).mean():.1f}%)",
             f"mean word overlap (Jaccard)       : {overlaps.mean():.3f}",
             f"Ang distance   : mean {gaps.mean():.0f}  median {int(np.median(gaps))}",
             f"different Raag : {diff_raag:,} of {n_gold:,} ({100 * diff_raag / max(n_gold, 1):.1f}%)",
             "=" * 62]
    report = "\n".join(lines)
    print(report)
    REPORT_TXT.write_text(report, encoding="utf-8")
    print(f"\nWrote {EVAL_JSONL}  ({len(records):,} queries)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", action="store_true", help="compare thresholds and write samples")
    ap.add_argument("--threshold", type=float, default=THRESHOLD)
    args = ap.parse_args()

    if not PADAS_JSONL.exists():
        sys.exit(f"Missing {PADAS_JSONL}. Run backend/pipeline/preprocessing/build_padas.py first.")
    padas = [json.loads(l) for l in PADAS_JSONL.open(encoding="utf-8")]
    pool = [p for p in padas if p["steek_ss"] and p["steek_ms"]]   # both commentators needed
    print(f"padas: {len(padas):,}; with both Sahib Singh and Manmohan Singh Steeks: {len(pool):,}")

    print("building tf-idf on the combined Steeks ...")
    Xs = tfidf([p["steek_ss"] for p in pool])
    Xm = tfidf([p["steek_ms"] for p in pool])
    query_idx = np.array([i for i, p in enumerate(pool) if p["split"] == EVAL_SPLIT])
    print(f"query padas from the '{EVAL_SPLIT}' split: {len(query_idx):,}")

    if args.sweep:
        sweep(pool, Xs, Xm, query_idx)
    else:
        build(pool, Xs, Xm, query_idx, args.threshold)


if __name__ == "__main__":
    main()
