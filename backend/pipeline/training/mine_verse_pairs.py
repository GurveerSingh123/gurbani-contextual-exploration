"""
Mine Pankati <-> Pankati training pairs.

Training on Steek->Pankati raised that task from 38% to 62% H@10 but left
Pankati->Pankati unchanged at ~19%. Aligning a verse with its own commentary
does not place semantically similar VERSES together, so the target task has to
be trained directly.

Two verses become a positive pair when two independent commentators (Sahib
Singh and Manmohan Singh) both explain them alike -- the same rule that built
the evaluation gold set, run on the TRAIN split only, so nothing leaks.

Pairs that merely look alike are dropped: a positive whose two verses share
most of their words teaches keyword matching, which is the behaviour we are
trying to remove. Low-overlap pairs are the ones that carry meaning.

Reads  : data/processed/cleaned_gurbani.csv, row_splits.csv
Writes : data/processed/verse_pairs.jsonl
         data/processed/verse_pairs_report.txt
"""

import json
import os
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

PROJECT_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(PROJECT_ROOT / ".env")

PROCESSED = PROJECT_ROOT / os.getenv("PROCESSED_DATA_DIR", "data/processed")
CLEANED_CSV = PROCESSED / "cleaned_gurbani.csv"
ROW_SPLITS_CSV = PROCESSED / "row_splits.csv"
OUT_JSONL = PROCESSED / "verse_pairs.jsonl"
REPORT_TXT = PROCESSED / "verse_pairs_report.txt"

# Matches the gold-set rule. Lower yields more pairs but visibly worse ones
# (0.30 paired "He cannot be created" with "Lanka's fort was gold"); higher
# collapses the count -- 0.40 leaves only 2,991 pairs, 0.45 only 1,424.
THRESHOLD = float(os.getenv("VERSE_PAIR_THRESHOLD", "0.35"))
CEILING = float(os.getenv("VERSE_PAIR_CEILING", "0.97"))
MAX_PER_VERSE = int(os.getenv("VERSE_PAIRS_PER_VERSE", "3"))
MAX_OVERLAP = float(os.getenv("VERSE_PAIR_MAX_OVERLAP", "0.60"))
TRAIN_SPLIT = "train"
CHUNK = 400

# Duplicated from build_eval_sets.py on purpose: these scripts are run
# standalone, and a shared import would need packaging. Keep them in step.
_MARKS = "ੁੂਿੀੇੈੋੌਾਂੰ੍ਃ"
_SPEAKER_FORMULAS = set("""
ਕਬੀਰ ਕਬੀਰੁ ਨਾਨਕ ਨਾਨਕੁ ਫਰੀਦ ਰਵਿਦਾਸ ਨਾਮਦੇਵ ਤ੍ਰਿਲੋਚਨ ਧੰਨਾ ਬੇਣੀ ਸੈਣ ਪੀਪਾ ਜੈਦੇਵ
ਭੀਖਨ ਸੂਰਦਾਸ ਆਖਦਾ ਆਖਦੇ ਆਖਦੀ ਕਹਿੰਦਾ ਕਹਿੰਦੇ ਆਖ ਕਹਿ ਭਾਈ ਹੇ ਜੀ ਜੀਉ ਸੰਤ ਜਨੋ
ਭਰਾਵੋ ਮੈਂ ਹਾਂ ਹੈ ਹਨ
""".split())
_EDGE = "।॥,—-!?()"
_NOISE = re.compile(r"[॥।0-9੦-੯]")


def strip_marks(token: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", token)
                   if not unicodedata.combining(c) and c not in _MARKS)


def drop_speaker_formulas(text: str) -> str:
    return " ".join(w for w in text.split()
                    if w.strip(_EDGE) not in _SPEAKER_FORMULAS)


def tokens(text: str) -> set:
    return {w for w in _NOISE.sub(" ", text).split() if len(w) > 1}


def jaccard(a: str, b: str) -> float:
    ta, tb = tokens(a), tokens(b)
    return len(ta & tb) / len(ta | tb) if ta and tb else 0.0


# See build_eval_sets.py: sklearn's default \b\w\w+\b keeps only consonant runs
# in Gurmukhi and makes unrelated verses look identical.
TOKEN_PATTERN = r"[^\s॥।,—()!?]+"


def tfidf(texts):
    cleaned = [drop_speaker_formulas(t) for t in texts]
    vec = TfidfVectorizer(analyzer="word", token_pattern=TOKEN_PATTERN,
                          ngram_range=(1, 2), min_df=3, max_df=0.30,
                          sublinear_tf=True)
    return normalize(vec.fit_transform(cleaned)).astype(np.float32)


def main() -> None:
    for p in (CLEANED_CSV, ROW_SPLITS_CSV):
        if not p.exists():
            sys.exit(f"Missing {p}. Run Steps 1 and 2 first.")

    df = pd.read_csv(CLEANED_CSV, encoding="utf-8-sig", low_memory=False)
    splits = pd.read_csv(ROW_SPLITS_CSV, encoding="utf-8-sig")
    df = df.merge(splits[["id", "split"]], on="id", how="left")

    pool = df[(df["split"] == TRAIN_SPLIT) & (df["is_trainable"] == 1)
              & df["steek_ss"].notna() & (df["steek_ss"] != "")
              & df["steek_ms"].notna() & (df["steek_ms"] != "")].reset_index(drop=True)
    for col in ["steek_ss", "steek_ms", "pankati"]:
        pool[col] = pool[col].astype(str)
    print(f"train pool (has ss + ms): {len(pool):,}")

    Xs, Xm = tfidf(pool["steek_ss"]), tfidf(pool["steek_ms"])
    shabad = pool["shabad_id"].values
    pankati = pool["pankati"].tolist()

    seen_pairs = set()
    records = []
    dropped_overlap = 0

    for start in range(0, len(pool), CHUNK):
        block = np.arange(start, min(start + CHUNK, len(pool)))
        sim = np.minimum((Xs[block] @ Xs.T).toarray(),
                         (Xm[block] @ Xm.T).toarray())
        for r, qi in enumerate(block):
            scores = sim[r]
            scores[shabad == shabad[qi]] = -1.0       # cross-Shabad only
            hits = np.flatnonzero((scores >= THRESHOLD) & (scores <= CEILING))
            if hits.size == 0:
                continue
            hits = hits[np.argsort(-scores[hits])]
            kept = 0
            for j in hits:
                if kept >= MAX_PER_VERSE:
                    break
                key = (min(qi, j), max(qi, j))
                if key in seen_pairs:
                    continue
                overlap = jaccard(pankati[qi], pankati[j])
                # A positive pair that already shares most of its words teaches
                # the shortcut we are trying to unlearn.
                if overlap > MAX_OVERLAP:
                    dropped_overlap += 1
                    continue
                seen_pairs.add(key)
                kept += 1
                records.append({
                    "query": pankati[qi],
                    "pos": [pankati[j]],
                    "neg": [],
                    "pair_type": "verse_verse",
                    "agreement": round(float(scores[j]), 4),
                    "word_overlap": round(overlap, 4),
                    "shabad_id": int(shabad[qi]),
                })
        print(f"  {min(start + CHUNK, len(pool)):,}/{len(pool):,}", end="\r")

    print()
    with OUT_JSONL.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    overlaps = np.array([r["word_overlap"] for r in records]) if records else np.array([0.0])
    agree = np.array([r["agreement"] for r in records]) if records else np.array([0.0])
    anchors = len({r["query"] for r in records})

    lines = [
        "=" * 62,
        "PANKATI <-> PANKATI TRAINING PAIRS",
        "=" * 62,
        f"Source split   : {TRAIN_SPLIT} only (no leakage into val/test)",
        f"Rule           : min(ss_sim, ms_sim) in [{THRESHOLD}, {CEILING}]",
        f"Constraints    : cross-Shabad, max {MAX_PER_VERSE}/verse, "
        f"word overlap <= {MAX_OVERLAP}",
        "",
        f"pairs written          : {len(records):,}",
        f"distinct anchor verses : {anchors:,} of {len(pool):,} "
        f"({100 * anchors / max(len(pool), 1):.1f}%)",
        f"dropped, too similar   : {dropped_overlap:,}",
        "",
        f"commentator agreement  : mean {agree.mean():.3f}  median {np.median(agree):.3f}",
        f"word overlap           : mean {overlaps.mean():.3f}",
        f"  pairs with ZERO shared words : {int((overlaps == 0).sum()):,} "
        f"({100 * (overlaps == 0).mean():.1f}%)",
        "=" * 62,
    ]
    report = "\n".join(lines)
    print("\n" + report)
    REPORT_TXT.write_text(report, encoding="utf-8")
    print(f"\nWrote {OUT_JSONL}")
    print(f"Wrote {REPORT_TXT}")


if __name__ == "__main__":
    main()
