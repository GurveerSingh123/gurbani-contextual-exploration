"""
Mine hard negatives: verses that LOOK alike but MEAN different things.

The remaining failure mode is lexical dominance. Asked for
"ਜੰਮਣੁ ਮਰਣਾ ਠਾਕਿ ਰਹਾਏ" (he STOPS the cycle of birth and death), the model
returns everything containing ਜੰਮਣੁ ਮਰਣਾ -- including "ਜੰਮਣੁ ਮਰਣੁ ਨ ਚੁਕਈ"
(the cycle NEVER ends), the opposite teaching.

Nothing in training ever said those should be far apart: the negatives were
random verses, which share no words, so matching words was always a winning
strategy. This mines the pairs that punish it.

A candidate becomes a hard negative when it:
  * shares a lot of vocabulary with the anchor        (lexical >= LEX_MIN)
  * is explained differently by both commentators     (meaning <= MEANING_MAX)
  * is clearly less related than the anchor's own positive   (margin)
  * sits in a different Shabad, and is not itself a positive

The margin matters. Absolute thresholds alone select genuinely related verses
whose commentaries merely word things differently -- training on those would
teach the model that related verses are opposites, which is worse than the
problem being solved.

Reads  : data/processed/verse_pairs.jsonl, cleaned_gurbani.csv, row_splits.csv
Writes : data/processed/verse_pairs_hn.jsonl
         data/processed/hard_negatives_report.txt
"""

import json
import os
import re
import sys
import unicodedata
from collections import defaultdict
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
PAIRS_JSONL = PROCESSED / "verse_pairs.jsonl"
OUT_JSONL = PROCESSED / "verse_pairs_hn.jsonl"
REPORT_TXT = PROCESSED / "hard_negatives_report.txt"

LEX_MIN = float(os.getenv("HN_LEXICAL_MIN", "0.35"))
# Upper bound: above this the "negative" is the same verse repeated in another
# Shabad, or a word-order variant of it (ਆਪੇ ਕਰਤਾ ਕਰੇ ਕਰਾਏ / ਆਪੇ ਕਰੇ ਕਰਾਏ ਕਰਤਾ).
# Training a verse to be unlike itself is worse than no negative at all.
LEX_MAX = float(os.getenv("HN_LEXICAL_MAX", "0.90"))
MEANING_MAX = float(os.getenv("HN_MEANING_MAX", "0.10"))
MARGIN = float(os.getenv("HN_MARGIN", "0.25"))
PER_PAIR = int(os.getenv("HN_PER_PAIR", "1"))
CHUNK = 200

_MARKS = "ੁੂਿੀੇੈੋੌਾਂੰ੍ਃ"
_SPEAKER_FORMULAS = set("""
ਕਬੀਰ ਕਬੀਰੁ ਨਾਨਕ ਨਾਨਕੁ ਫਰੀਦ ਰਵਿਦਾਸ ਨਾਮਦੇਵ ਤ੍ਰਿਲੋਚਨ ਧੰਨਾ ਬੇਣੀ ਸੈਣ ਪੀਪਾ ਜੈਦੇਵ
ਭੀਖਨ ਸੂਰਦਾਸ ਆਖਦਾ ਆਖਦੇ ਆਖਦੀ ਕਹਿੰਦਾ ਕਹਿੰਦੇ ਆਖ ਕਹਿ ਭਾਈ ਹੇ ਜੀ ਜੀਉ ਸੰਤ ਜਨੋ
ਭਰਾਵੋ ਮੈਂ ਹਾਂ ਹੈ ਹਨ
""".split())
_EDGE = "।॥,—-!?()"
_NOISE = re.compile(r"[॥।0-9੦-੯]")


def drop_speaker_formulas(text: str) -> str:
    return " ".join(w for w in text.split() if w.strip(_EDGE) not in _SPEAKER_FORMULAS)


# See build_eval_sets.py: sklearn's default \b\w\w+\b keeps only consonant runs
# in Gurmukhi. It scored "ਗਾਵੈ ਕੋ ਵਿਦਿਆ ਵਿਖਮੁ ਵੀਚਾਰੁ" and "ਕੋਊ ਬਿਖਮ ਗਾਰ ਤੋਰੈ"
# as a perfect 1.000 match -- both collapsed to the single fragment ['ਖਮ'].
TOKEN_PATTERN = r"[^\s॥।,—()!?]+"


def meaning_tfidf(texts):
    cleaned = [drop_speaker_formulas(t) for t in texts]
    vec = TfidfVectorizer(analyzer="word", token_pattern=TOKEN_PATTERN,
                          ngram_range=(1, 2), min_df=3, max_df=0.30,
                          sublinear_tf=True)
    return normalize(vec.fit_transform(cleaned)).astype(np.float32)


def lexical_tfidf(texts):
    """Similarity of the VERSE surface -- what the model is over-relying on."""
    cleaned = [_NOISE.sub(" ", t) for t in texts]
    vec = TfidfVectorizer(analyzer="word", token_pattern=TOKEN_PATTERN,
                          ngram_range=(1, 1), min_df=2, sublinear_tf=True)
    return normalize(vec.fit_transform(cleaned)).astype(np.float32)


def main() -> None:
    for p in (CLEANED_CSV, ROW_SPLITS_CSV, PAIRS_JSONL):
        if not p.exists():
            sys.exit(f"Missing {p}. Run mine_verse_pairs.py first.")

    df = pd.read_csv(CLEANED_CSV, encoding="utf-8-sig", low_memory=False)
    splits = pd.read_csv(ROW_SPLITS_CSV, encoding="utf-8-sig")
    df = df.merge(splits[["id", "split"]], on="id", how="left")
    pool = df[(df["split"] == "train") & (df["is_trainable"] == 1)
              & df["steek_ss"].notna() & (df["steek_ss"] != "")
              & df["steek_ms"].notna() & (df["steek_ms"] != "")].reset_index(drop=True)
    for col in ["steek_ss", "steek_ms", "pankati"]:
        pool[col] = pool[col].astype(str)
    print(f"train pool: {len(pool):,}")

    pairs = [json.loads(l) for l in PAIRS_JSONL.open(encoding="utf-8")]
    print(f"positive pairs: {len(pairs):,}")

    index_of = {}
    for i, verse in enumerate(pool["pankati"]):
        index_of.setdefault(verse, i)

    Xl = lexical_tfidf(pool["pankati"])
    Xs = meaning_tfidf(pool["steek_ss"])
    Xm = meaning_tfidf(pool["steek_ms"])
    shabad = pool["shabad_id"].values
    verses = pool["pankati"].tolist()

    # Every positive of an anchor is off-limits as its negative.
    positives_of = defaultdict(set)
    for rec in pairs:
        positives_of[rec["query"]].add(rec["pos"][0])

    by_anchor = defaultdict(list)
    for rec in pairs:
        by_anchor[rec["query"]].append(rec)
    anchors = [a for a in by_anchor if a in index_of]
    print(f"anchors resolvable in pool: {len(anchors):,}")

    out_records = []
    no_negative = 0
    dropped_margin = 0

    for start in range(0, len(anchors), CHUNK):
        block = anchors[start:start + CHUNK]
        rows = np.array([index_of[a] for a in block])
        lex = (Xl[rows] @ Xl.T).toarray()
        mean_sim = np.minimum((Xs[rows] @ Xs.T).toarray(),
                              (Xm[rows] @ Xm.T).toarray())

        for r, anchor in enumerate(block):
            qi = rows[r]
            lex[r, qi] = -1.0
            lex[r, shabad == shabad[qi]] = -1.0
            banned = {index_of[p] for p in positives_of[anchor] if p in index_of}
            for b in banned:
                lex[r, b] = -1.0

            cands = np.flatnonzero((lex[r] >= LEX_MIN) & (lex[r] <= LEX_MAX)
                                   & (mean_sim[r] <= MEANING_MAX))
            if cands.size == 0:
                no_negative += len(by_anchor[anchor])
                continue
            # Most lexically confusable first -- the hardest cases.
            cands = cands[np.argsort(-lex[r][cands])]

            for rec in by_anchor[anchor]:
                # Relative test: the negative must be clearly less related than
                # this pair's own positive, not merely below a fixed number.
                cutoff = rec["agreement"] - MARGIN
                chosen = [c for c in cands if mean_sim[r, c] <= cutoff][:PER_PAIR]
                if not chosen:
                    dropped_margin += 1
                    no_negative += 1
                    continue
                new = dict(rec)
                new["neg"] = [verses[c] for c in chosen]
                new["neg_lexical"] = [round(float(lex[r, c]), 4) for c in chosen]
                new["neg_meaning"] = [round(float(mean_sim[r, c]), 4) for c in chosen]
                out_records.append(new)
        print(f"  {min(start + CHUNK, len(anchors)):,}/{len(anchors):,}", end="\r")

    print()
    with OUT_JSONL.open("w", encoding="utf-8") as f:
        for rec in out_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    lexs = np.array([r["neg_lexical"][0] for r in out_records]) if out_records else np.array([0.0])
    means = np.array([r["neg_meaning"][0] for r in out_records]) if out_records else np.array([0.0])

    lines = [
        "=" * 62,
        "HARD NEGATIVES  (look alike, mean different things)",
        "=" * 62,
        f"Rule    : lexical in [{LEX_MIN}, {LEX_MAX}], meaning <= {MEANING_MAX}, "
        f"and <= positive's agreement - {MARGIN}",
        f"Limit   : {PER_PAIR} negative(s) per pair, cross-Shabad, never a known positive",
        "",
        f"pairs with a hard negative : {len(out_records):,} of {len(pairs):,} "
        f"({100 * len(out_records) / max(len(pairs), 1):.1f}%)",
        f"pairs left without one     : {no_negative:,}",
        f"  ...rejected by margin    : {dropped_margin:,}",
        "",
        f"negative lexical similarity : mean {lexs.mean():.3f}  max {lexs.max():.3f}",
        f"negative meaning similarity : mean {means.mean():.3f}",
        "=" * 62,
    ]
    report = "\n".join(lines)
    print("\n" + report)
    REPORT_TXT.write_text(report, encoding="utf-8")
    print(f"\nWrote {OUT_JSONL}")
    print(f"Wrote {REPORT_TXT}")


if __name__ == "__main__":
    main()
