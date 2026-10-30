"""
Step 3a: build a Pankati -> Pankati semantic gold set.

The primary feature is verse-to-verse: given a Pankati, return other Pankatis
carrying the same Bhavarth. There are no such labels in the data, so we mine
them from the commentaries.

Method: two Pankatis are a gold pair when TWO INDEPENDENT commentators both
explain them similarly. Requiring agreement between Sahib Singh and Manmohan
Singh -- different authors, different eras, different styles -- rejects pairs
that merely share formulaic commentary phrasing ("...says", "O brother!").

Word overlap between the verses is NOT filtered. Same meaning is the criterion;
different wording is a bonus, and it is recorded per pair so we can report how
much of the gold set keyword search cannot reach.

Gold pairs are cross-Shabad by construction, matching the product preference
for surfacing verses from elsewhere in the Granth.

Reads  : data/processed/cleaned_gurbani.csv, row_splits.csv
Writes : data/processed/eval_semantic_pairs.jsonl
         data/processed/eval_report.txt
"""

import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

PROJECT_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(PROJECT_ROOT / ".env")

OUT_DIR = PROJECT_ROOT / os.getenv("PROCESSED_DATA_DIR", "data/processed")
CLEANED_CSV = OUT_DIR / "cleaned_gurbani.csv"
ROW_SPLITS_CSV = OUT_DIR / "row_splits.csv"
EVAL_JSONL = OUT_DIR / "eval_semantic_pairs.jsonl"
REPORT_TXT = OUT_DIR / "eval_report.txt"

# A pair is gold when min(ss_similarity, ms_similarity) clears this bar.
GOLD_THRESHOLD = float(os.getenv("GOLD_THRESHOLD", "0.35"))
# Near-1.0 on both usually means the commentator reused one sentence verbatim;
# such pairs are real but trivial, and they would flatter every model equally.
GOLD_CEILING = float(os.getenv("GOLD_CEILING", "0.97"))
MAX_GOLD_PER_QUERY = int(os.getenv("MAX_GOLD_PER_QUERY", "10"))
EVAL_SPLIT = os.getenv("EVAL_SPLIT", "test")
CHUNK = 400

_STRIP = re.compile(r"[॥।0-9੦-੯]")


def tokens(text: str) -> set:
    return {w for w in _STRIP.sub(" ", text).split() if len(w) > 1}


def jaccard(a: str, b: str) -> float:
    ta, tb = tokens(a), tokens(b)
    return len(ta & tb) / len(ta | tb) if ta and tb else 0.0


# Attribution and vocative formulas. Both commentators render "ਕਹੈ ਕਬੀਰੁ" as
# "ਕਬੀਰ ਆਖਦਾ ਹੈ", so they "agree" on the speaker tag rather than the teaching --
# which passed the two-commentator test while meaning nothing. Removing these
# rejects that pair outright and RAISES coverage, because the formula noise was
# crowding out real matches.
_SPEAKER_FORMULAS = set("""
ਕਬੀਰ ਕਬੀਰੁ ਨਾਨਕ ਨਾਨਕੁ ਫਰੀਦ ਰਵਿਦਾਸ ਨਾਮਦੇਵ ਤ੍ਰਿਲੋਚਨ ਧੰਨਾ ਬੇਣੀ ਸੈਣ ਪੀਪਾ ਜੈਦੇਵ
ਭੀਖਨ ਸੂਰਦਾਸ ਆਖਦਾ ਆਖਦੇ ਆਖਦੀ ਕਹਿੰਦਾ ਕਹਿੰਦੇ ਆਖ ਕਹਿ ਭਾਈ ਹੇ ਜੀ ਜੀਉ ਸੰਤ ਜਨੋ
ਭਰਾਵੋ ਮੈਂ ਹਾਂ ਹੈ ਹਨ
""".split())

_EDGE_PUNCT = "।॥,—-!?()"


def drop_speaker_formulas(text: str) -> str:
    return " ".join(w for w in text.split()
                    if w.strip(_EDGE_PUNCT) not in _SPEAKER_FORMULAS)


# scikit-learn's default token pattern is \b\w\w+\b, which DESTROYS Gurmukhi:
# vowel signs are combining marks and are not word characters, so only runs of
# two or more consecutive consonants survive. "ਗਾਵੈ ਕੋ ਵਿਦਿਆ ਵਿਖਮੁ ਵੀਚਾਰੁ"
# tokenised to ['ਖਮ'], as did the unrelated "ਕੋਊ ਬਿਖਮ ਗਾਰ ਤੋਰੈ" -- making two
# different verses look identical. Split on whitespace and punctuation instead.
TOKEN_PATTERN = r"[^\s॥।,—()!?]+"


def tfidf(texts) -> "np.ndarray":
    # max_df drops the commentary's own boilerplate; min_df drops typos.
    cleaned = [drop_speaker_formulas(t) for t in texts]
    vec = TfidfVectorizer(analyzer="word", token_pattern=TOKEN_PATTERN,
                          ngram_range=(1, 2), min_df=3, max_df=0.30,
                          sublinear_tf=True)
    return normalize(vec.fit_transform(cleaned)).astype(np.float32)


def main() -> None:
    for path in (CLEANED_CSV, ROW_SPLITS_CSV):
        if not path.exists():
            sys.exit(f"Missing {path}. Run Step 1 and Step 2 first.")

    df = pd.read_csv(CLEANED_CSV, encoding="utf-8-sig", low_memory=False)
    splits = pd.read_csv(ROW_SPLITS_CSV, encoding="utf-8-sig")
    df = df.merge(splits[["id", "split"]], on="id", how="left")

    # Gold mining needs both commentators present on both sides of a pair.
    pool = df[(df["is_trainable"] == 1)
              & df["steek_ss"].notna() & (df["steek_ss"] != "")
              & df["steek_ms"].notna() & (df["steek_ms"] != "")].copy()
    pool = pool.reset_index(drop=True)
    for col in ["steek_ss", "steek_ms", "pankati"]:
        pool[col] = pool[col].astype(str)
    print(f"candidate pool (has ss + ms): {len(pool):,}")

    Xs = tfidf(pool["steek_ss"])
    Xm = tfidf(pool["steek_ms"])
    print(f"tf-idf vocab: ss={Xs.shape[1]:,}  ms={Xm.shape[1]:,}")

    query_idx = np.flatnonzero((pool["split"] == EVAL_SPLIT).values)
    print(f"queries from '{EVAL_SPLIT}' split: {len(query_idx):,}")

    shabad = pool["shabad_id"].values
    ang = pool["ang"].values
    raag = pool["raag_id"].values
    pankati = pool["pankati"].tolist()

    records = []
    n_gold = 0
    for start in range(0, len(query_idx), CHUNK):
        block = query_idx[start:start + CHUNK]
        sim_ss = (Xs[block] @ Xs.T).toarray()
        sim_ms = (Xm[block] @ Xm.T).toarray()
        # Agreement between two independent commentators.
        comb = np.minimum(sim_ss, sim_ms)

        for r, qi in enumerate(block):
            scores = comb[r]
            scores[shabad == shabad[qi]] = -1.0      # cross-Shabad only
            hits = np.flatnonzero((scores >= GOLD_THRESHOLD) & (scores <= GOLD_CEILING))
            if hits.size == 0:
                continue
            hits = hits[np.argsort(-scores[hits])][:MAX_GOLD_PER_QUERY]

            gold = []
            for j in hits:
                gold.append({
                    "id": int(pool.at[j, "id"]),
                    "pankati": pankati[j],
                    "ang": int(ang[j]),
                    "shabad_id": int(shabad[j]),
                    "raag_id": None if pd.isna(raag[j]) else int(raag[j]),
                    "ss_sim": round(float(sim_ss[r, j]), 4),
                    "ms_sim": round(float(sim_ms[r, j]), 4),
                    "word_overlap": round(jaccard(pankati[qi], pankati[j]), 4),
                    "same_ang": bool(ang[j] == ang[qi]),
                    "same_raag": bool(raag[j] == raag[qi]),
                })
            n_gold += len(gold)
            records.append({
                "query_id": int(pool.at[qi, "id"]),
                "query_pankati": pankati[qi],
                "query_ang": int(ang[qi]),
                "query_shabad_id": int(shabad[qi]),
                "query_raag_id": None if pd.isna(raag[qi]) else int(raag[qi]),
                "gold": gold,
            })
        print(f"  {min(start + CHUNK, len(query_idx)):,}/{len(query_idx):,} queries", end="\r")

    print()
    with EVAL_JSONL.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # --- report -----------------------------------------------------------
    def as_array(values, fallback=0.0):
        arr = np.array(values, dtype=float)
        return arr if arr.size else np.array([fallback])

    per_query = as_array([len(r["gold"]) for r in records])
    overlaps = as_array([g["word_overlap"] for r in records for g in r["gold"]])
    ang_gap = as_array([abs(g["ang"] - r["query_ang"]) for r in records for g in r["gold"]])
    diff_raag = sum(1 for r in records for g in r["gold"] if not g["same_raag"])

    lines = []
    add = lines.append
    add("=" * 62)
    add("SEMANTIC EVALUATION SET  (Pankati -> Pankati)")
    add("=" * 62)
    add(f"Gold rule   : min(ss_sim, ms_sim) in [{GOLD_THRESHOLD}, {GOLD_CEILING}]")
    add(f"Constraint  : cross-Shabad only; word overlap NOT filtered")
    add(f"Query split : {EVAL_SPLIT}")
    add("")
    add(f"Queries offered      : {len(query_idx):,}")
    add(f"Queries with gold    : {len(records):,} "
        f"({100 * len(records) / max(len(query_idx), 1):.1f}%)")
    add(f"Total gold pairs     : {n_gold:,}")
    add(f"Gold per query       : mean {per_query.mean():.2f}  median {int(np.median(per_query))}"
        f"  max {int(per_query.max())}")
    add("")
    add("-- how much of this is beyond keyword search ------------------")
    add(f"gold pairs with ZERO shared words : {int((overlaps == 0).sum()):,} "
        f"({100 * (overlaps == 0).mean():.1f}%)")
    add(f"mean word overlap (Jaccard)       : {overlaps.mean():.3f}")
    add("")
    add("-- spread across the Granth -----------------------------------")
    if ang_gap.size:
        add(f"Ang distance : mean {ang_gap.mean():.0f}  median {int(np.median(ang_gap))}"
            f"  max {int(ang_gap.max())}")
    add(f"different Raag : {diff_raag:,} of {n_gold:,} "
        f"({100 * diff_raag / max(n_gold, 1):.1f}%)")
    add("=" * 62)

    report = "\n".join(lines)
    print("\n" + report)
    REPORT_TXT.write_text(report, encoding="utf-8")
    print(f"\nWrote {EVAL_JSONL}  ({len(records):,} queries)")
    print(f"Wrote {REPORT_TXT}")


if __name__ == "__main__":
    main()
