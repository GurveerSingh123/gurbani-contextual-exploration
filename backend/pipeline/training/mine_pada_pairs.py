"""
Mine pada <-> pada training pairs, with a hard negative for each.

The pada version of mine_verse_pairs.py + mine_hard_negatives.py, run in one go.
A model trained on single lines (verse_hn) scored 39.0 H@10 on the pada gold set
against char-BM25's 48.8; it has never seen a pada. This builds what it needs to
learn from:

  positive  : two padas from different Shabads that BOTH commentators explain
              alike (combined-Steek tf-idf, same rule as build_pada_eval.py)
  negative  : a pada that shares wording with the anchor but whose commentary
              says something clearly different -- so matching words alone stops
              paying off

Positives that already share most of their words are dropped: they teach keyword
matching, the behaviour being trained away.

Scale: a pada's combined commentary is longer, so tf-idf cosines run about half
the line-level values (0.15 here plays the role 0.30 plays for lines). The
hard-negative limits are halved to match; all are overridable in .env.

Runs on the TRAIN split for training and the VAL split for validation, each
mined only within itself, so nothing crosses into the test split.

  python backend/pipeline/training/mine_pada_pairs.py

Caution on the negatives: a HIGH commentary similarity reliably means the same
teaching, but at pada level a LOW one does not reliably mean a different one --
longer commentaries can say the same thing in other words. Reading the samples
showed some "negatives" that teach the same thing as their anchor. So train a
positives-only model (pada_pairs.jsonl) as the main run, and the hard-negative
version (pada_pairs_hn.jsonl) as a comparison; the pada gold set decides.

Reads  : data/processed/padas.jsonl
Writes : data/processed/pada_pairs.jsonl            (train positives, all)
         data/processed/pada_pairs_hn.jsonl         (train, with hard negative)
         data/processed/pada_val_pairs.jsonl        (val positives, all)
         data/processed/pada_val_pairs_hn.jsonl     (val, with hard negative)
         data/processed/pada_pairs_report.txt
         data/processed/pada_pairs_samples.txt      (examples to read)
"""

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
REPORT_TXT = PROCESSED / "pada_pairs_report.txt"
SAMPLES_TXT = PROCESSED / "pada_pairs_samples.txt"
SEED = int(os.getenv("SEED", "42"))

# Positives: same rule as the pada gold set.
THRESHOLD = float(os.getenv("PADA_PAIR_THRESHOLD", "0.15"))
CEILING = float(os.getenv("PADA_PAIR_CEILING", "0.97"))
MAX_PER_PADA = int(os.getenv("PADA_PAIRS_PER_PADA", "3"))
MAX_OVERLAP = float(os.getenv("PADA_PAIR_MAX_OVERLAP", "0.60"))
# Hard negatives: line values 0.35 / 0.90 / 0.10 / 0.25, scaled for padas.
LEX_MIN = float(os.getenv("PADA_HN_LEXICAL_MIN", "0.25"))
LEX_MAX = float(os.getenv("PADA_HN_LEXICAL_MAX", "0.90"))
MEANING_MAX = float(os.getenv("PADA_HN_MEANING_MAX", "0.05"))
MARGIN = float(os.getenv("PADA_HN_MARGIN", "0.10"))
CHUNK = 400

# --- duplicated from the line-level miners; keep in step ---------------------------
_SPEAKER_FORMULAS = set("""
ਕਬੀਰ ਕਬੀਰੁ ਨਾਨਕ ਨਾਨਕੁ ਫਰੀਦ ਰਵਿਦਾਸ ਨਾਮਦੇਵ ਤ੍ਰਿਲੋਚਨ ਧੰਨਾ ਬੇਣੀ ਸੈਣ ਪੀਪਾ ਜੈਦੇਵ
ਭੀਖਨ ਸੂਰਦਾਸ ਆਖਦਾ ਆਖਦੇ ਆਖਦੀ ਕਹਿੰਦਾ ਕਹਿੰਦੇ ਆਖ ਕਹਿ ਭਾਈ ਹੇ ਜੀ ਜੀਉ ਸੰਤ ਜਨੋ
ਭਰਾਵੋ ਮੈਂ ਹਾਂ ਹੈ ਹਨ
""".split())
_EDGE = "।॥,—-!?()"
_NOISE = re.compile(r"[॥।0-9੦-੯]")
# Never sklearn's default \b\w\w+\b: it reduces Gurmukhi to consonant fragments.
TOKEN_PATTERN = r"[^\s॥।,—()!?]+"


def drop_speaker_formulas(text: str) -> str:
    return " ".join(w for w in text.split() if w.strip(_EDGE) not in _SPEAKER_FORMULAS)


def tokens(text: str) -> set:
    return {w for w in _NOISE.sub(" ", text).split() if len(w) > 1}


def jaccard(a: str, b: str) -> float:
    ta, tb = tokens(a), tokens(b)
    return len(ta & tb) / len(ta | tb) if ta and tb else 0.0


def meaning_tfidf(texts):
    cleaned = [drop_speaker_formulas(t) for t in texts]
    vec = TfidfVectorizer(analyzer="word", token_pattern=TOKEN_PATTERN,
                          ngram_range=(1, 2), min_df=3, max_df=0.30, sublinear_tf=True)
    return normalize(vec.fit_transform(cleaned)).astype(np.float32)


def lexical_tfidf(texts):
    """Similarity of the VERSE surface -- what the model over-relies on."""
    cleaned = [_NOISE.sub(" ", t) for t in texts]
    vec = TfidfVectorizer(analyzer="word", token_pattern=TOKEN_PATTERN,
                          ngram_range=(1, 1), min_df=2, sublinear_tf=True)
    return normalize(vec.fit_transform(cleaned)).astype(np.float32)
# ----------------------------------------------------------------------------------


def mine(pool):
    """-> (records with a negative, all positive records, counts)."""
    texts = [p["text"] for p in pool]
    shabad = np.array([p["shabad_id"] for p in pool])
    Xs = meaning_tfidf([p["steek_ss"] for p in pool])
    Xm = meaning_tfidf([p["steek_ms"] for p in pool])
    Xl = lexical_tfidf(texts)

    positives, seen, too_similar = {}, set(), 0
    for start in range(0, len(pool), CHUNK):
        block = np.arange(start, min(start + CHUNK, len(pool)))
        sim = np.minimum((Xs[block] @ Xs.T).toarray(), (Xm[block] @ Xm.T).toarray())
        for r, qi in enumerate(block):
            scores = sim[r]
            scores[shabad == shabad[qi]] = -1.0                       # cross-Shabad only
            hits = np.flatnonzero((scores >= THRESHOLD) & (scores <= CEILING))
            kept = 0
            for j in hits[np.argsort(-scores[hits])]:
                if kept >= MAX_PER_PADA:
                    break
                key = (min(qi, j), max(qi, j))
                if key in seen or texts[j] == texts[qi]:
                    continue
                overlap = jaccard(texts[qi], texts[j])
                if overlap > MAX_OVERLAP:                             # teaches keyword matching
                    too_similar += 1
                    continue
                seen.add(key)
                kept += 1
                positives.setdefault(int(qi), []).append((int(j), float(scores[j]), overlap))
        print(f"  positives: {min(start + CHUNK, len(pool)):,}/{len(pool):,} padas", end="\r")
    print()

    anchors = sorted(positives)
    with_neg, all_pos, no_neg = [], [], 0
    pos_of = {qi: {j for j, _, _ in positives[qi]} for qi in anchors}
    for start in range(0, len(anchors), CHUNK):
        rows = np.array(anchors[start:start + CHUNK])
        lex = (Xl[rows] @ Xl.T).toarray()
        mean = np.minimum((Xs[rows] @ Xs.T).toarray(), (Xm[rows] @ Xm.T).toarray())
        for r, qi in enumerate(rows):
            lex[r, shabad == shabad[qi]] = -1.0
            for b in pos_of[qi]:
                lex[r, b] = -1.0
            cands = np.flatnonzero((lex[r] >= LEX_MIN) & (lex[r] <= LEX_MAX)
                                   & (mean[r] <= MEANING_MAX))
            cands = cands[np.argsort(-lex[r][cands])]                 # most confusable first
            for j, agree, overlap in positives[qi]:
                rec = {"query": texts[qi], "pos": [texts[j]], "neg": [], "pair_type": "pada_pada",
                       "agreement": round(agree, 4), "word_overlap": round(overlap, 4),
                       "shabad_id": int(shabad[qi])}
                all_pos.append(rec)
                chosen = [c for c in cands if mean[r, c] <= agree - MARGIN and texts[c] != texts[qi]][:1]
                if not chosen:
                    no_neg += 1
                    continue
                c = chosen[0]
                with_neg.append({**rec, "neg": [texts[c]],
                                 "neg_lexical": [round(float(lex[r, c]), 4)],
                                 "neg_meaning": [round(float(mean[r, c]), 4)]})
        print(f"  negatives: {min(start + CHUNK, len(anchors)):,}/{len(anchors):,} anchors", end="\r")
    print()
    return with_neg, all_pos, {"pool": len(pool), "anchors": len(anchors),
                               "too_similar": too_similar, "no_neg": no_neg}


def write(path, records):
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main() -> None:
    if not PADAS_JSONL.exists():
        sys.exit(f"Missing {PADAS_JSONL}. Run backend/pipeline/preprocessing/build_padas.py first.")
    padas = [json.loads(l) for l in PADAS_JSONL.open(encoding="utf-8")]

    lines = ["=" * 66, "PADA <-> PADA TRAINING PAIRS  (with hard negatives)", "=" * 66,
             f"Positive rule : min(ss, ms) combined-Steek tf-idf in [{THRESHOLD}, {CEILING}], "
             f"cross-Shabad, word overlap <= {MAX_OVERLAP}, max {MAX_PER_PADA} per pada",
             f"Negative rule : verse tf-idf in [{LEX_MIN}, {LEX_MAX}], meaning <= {MEANING_MAX} "
             f"and <= positive's agreement - {MARGIN}", ""]
    outputs = {}
    for split, hn_name in (("train", "pada_pairs_hn.jsonl"), ("val", "pada_val_pairs_hn.jsonl")):
        pool = [p for p in padas if p["split"] == split and p["steek_ss"] and p["steek_ms"]]
        print(f"[{split}] padas with both Steeks: {len(pool):,}")
        with_neg, all_pos, n = mine(pool)
        write(PROCESSED / hn_name, with_neg)
        write(PROCESSED / ("pada_pairs.jsonl" if split == "train" else "pada_val_pairs.jsonl"), all_pos)
        outputs[split] = (with_neg, all_pos)
        zero = sum(1 for r in all_pos if r["word_overlap"] == 0)
        lines += [f"[{split}]",
                  f"  padas in pool            : {n['pool']:,}",
                  f"  padas with a positive    : {n['anchors']:,} ({100 * n['anchors'] / max(n['pool'], 1):.1f}%)",
                  f"  positive pairs           : {len(all_pos):,}   (zero shared words: {zero:,}, "
                  f"{100 * zero / max(len(all_pos), 1):.1f}%)",
                  f"  dropped, too similar     : {n['too_similar']:,}",
                  f"  with a hard negative     : {len(with_neg):,} ({100 * len(with_neg) / max(len(all_pos), 1):.1f}%)"
                  f"  -> {hn_name}", ""]
    lines.append("=" * 66)
    report = "\n".join(lines)
    print("\n" + report)
    REPORT_TXT.write_text(report, encoding="utf-8")

    # Samples to read: does the positive share the teaching, and is the negative
    # really different despite the shared words?
    rng = random.Random(SEED)
    train = outputs["train"][0]
    out = ["PADA TRAINING TRIPLETS -- random samples to read", "=" * 70,
           "Check: POSITIVE should carry the same teaching as ANCHOR;",
           "NEGATIVE should share words with ANCHOR but teach something else.", ""]
    by_text = {p["text"]: p for p in padas}
    for rec in rng.sample(train, min(15, len(train))):
        out += ["-" * 70,
                f"agreement {rec['agreement']}  negative: lexical {rec['neg_lexical'][0]}, "
                f"meaning {rec['neg_meaning'][0]}"]
        for label, text in (("ANCHOR  ", rec["query"]), ("POSITIVE", rec["pos"][0]), ("NEGATIVE", rec["neg"][0])):
            ss = by_text.get(text, {}).get("steek_ss", "")
            out += [f"{label}: {text}", f"          SS: {ss[:220]}"]
    SAMPLES_TXT.write_text("\n".join(out), encoding="utf-8")
    print(f"\nWrote pada_pairs.jsonl, pada_pairs_hn.jsonl, pada_val_pairs.jsonl, pada_val_pairs_hn.jsonl")
    print(f"Wrote {REPORT_TXT.name} and {SAMPLES_TXT.name} (read the samples)")


if __name__ == "__main__":
    main()
