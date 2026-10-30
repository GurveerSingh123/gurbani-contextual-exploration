"""
Step 1: profile and clean the Gurbani dataset.

Reads the source workbook read-only and writes ONLY new files:
  data/processed/cleaned_gurbani.csv
  data/processed/profile_report.txt
  data/processed/flagged_structural_review.csv

Nothing is deleted. Structural/title lines are FLAGGED, not dropped, so the
regex can be reviewed before Step 2 excludes them from the corpus.
"""

import os
import re
import sys
import unicodedata
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(PROJECT_ROOT / ".env")

SOURCE = PROJECT_ROOT / os.getenv(
    "SOURCE_DATASET", "data/processed/final_gurbani_dataset_all_steeks.xlsx"
)
OUT_DIR = PROJECT_ROOT / os.getenv("PROCESSED_DATA_DIR", "data/processed")
RAAG_METADATA = PROJECT_ROOT / os.getenv("RAAG_METADATA", "data/metadata/raag_metadata.csv")

TRAIN_STEEKS = [c.strip() for c in os.getenv("TRAIN_STEEK_COLS", "").split(",") if c.strip()]
ALL_STEEKS = [c.strip() for c in os.getenv("ALL_STEEK_COLS", "").split(",") if c.strip()]

CLEANED_CSV = OUT_DIR / "cleaned_gurbani.csv"
REPORT_TXT = OUT_DIR / "profile_report.txt"
REVIEW_CSV = OUT_DIR / "flagged_structural_review.csv"

# --------------------------------------------------------------------------
# Text normalisation
# --------------------------------------------------------------------------

_TAG = re.compile(r"<[^>]{1,40}>")
_WS = re.compile(r"\s+")
# NBSP, zero-width space/joiner/non-joiner, BOM, soft hyphen
_INVISIBLE = dict.fromkeys(map(ord, "\u00a0\u200b\u200c\u200d\ufeff\u00ad"), " ")

_NULLISH = {"", "nan", "none", "null", "na", "n/a", "-"}


def normalize(value) -> str:
    """Collapse a cell to clean single-line text. Empty-ish -> ''."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value)
    text = text.replace("<br>", " ").replace("<br/>", " ").replace("<br />", " ")
    text = _TAG.sub(" ", text)
    text = text.translate(_INVISIBLE)
    text = _WS.sub(" ", text).strip()
    return "" if text.lower() in _NULLISH else text


# --------------------------------------------------------------------------
# Structural / title-line detection
#
# `is_structural` in the source flags only 720 rows, but the corpus contains
# thousands of unflagged title lines. These are not teachings and, because
# they repeat verbatim, they poison contrastive training as false negatives.
# --------------------------------------------------------------------------

# `\b` is unusable here: Gurmukhi vowel signs (ਾ ਿ ੁ) are combining marks, which
# Python's re does not treat as word characters, so a word boundary never fires
# after them. Require an explicit terminator instead.
_END = r"(?=[\s॥।]|$)"

_HEADER_PATTERNS = [
    rf"^ਰਾਗੁ?{_END}",                       # raag heading
    r"^(ਪਉੜੀ|ਸਲੋਕੁ?|ਸਲੋਕ|ਛੰਤੁ?|ਅਸਟਪਦੀ|ਸੋਹਿਲਾ|ਵਾਰ)\s*॥?\s*$",
    rf"^(ਛਕਾ|ਛਕੇ|ਦੁਪਦੇ|ਦੁਪਦਾ|ਚਉਪਦੇ|ਤਿਪਦੇ|ਪੰਚਪਦਾ){_END}",
    r"^ਰਾਗ\s*ਮਾਲਾ",
    # Must be followed by a number ("ਘਰੁ ੨ ॥"). A bare `^ਘਰੁ\s` also swallowed
    # real verses such as "ਘਰੁ ਦਰੁ ਰਾਖਹਿ ਜੇ ਰਸੁ ਚਾਖਹਿ ...".
    r"^ਘਰੁ\s*[੦-੯0-9]",
    r"^ਮਃ\s*[੦-੯0-9]",                      # "ਮਃ ੩ ॥"
    r"^ਮਹਲਾ\s*[੦-੯0-9]",
    r"^[੦-੯0-9\s॥।]+$",                     # digits / danda only
]
_HEADER_RE = re.compile("|".join(_HEADER_PATTERNS))

# "ਆਸਾ ਮਹਲਾ ੫ ॥", "ਸਾਰਗ ਮਹਲਾ ੫ ॥" — raag + author, only when short.
_AUTHOR_RE = re.compile(r"(ਮਹਲਾ|ਮਃ)\s*[੦-੯0-9]")
_SHORT_TITLE_MAX = 45


def is_header_line(pankati: str) -> bool:
    if not pankati:
        return True
    # A bare "ੴ ਸਤਿਗੁਰ ਪ੍ਰਸਾਦਿ ॥" is a section header (522 occurrences), but the
    # full Mool Mantar opening the Granth is scripture. Separate them by length.
    if pankati.startswith("ੴ"):
        return len(pankati) <= _SHORT_TITLE_MAX
    if _HEADER_RE.search(pankati):
        return True
    if len(pankati) <= _SHORT_TITLE_MAX and _AUTHOR_RE.search(pankati):
        return True
    return False


# --------------------------------------------------------------------------
# Lexicon title detection.
#
# The prefix rules above miss bare raag headings — "ਆਸਾ ॥" (39x), "ਗਉੜੀ ॥",
# "ਗੋਂਡ ॥", "ਪਵੜੀ ॥". A length rule cannot catch them without also eating real
# short verses like "ਆਦੇਸੁ ਤਿਸੈ ਆਦੇਸੁ ॥", so match on vocabulary instead: a
# short line ending in ॥ whose every token is a raag name or a composition
# term is a heading, not a teaching.
# --------------------------------------------------------------------------

# Vowel signs vary between the metadata and the corpus (ਬਿਲਾਵਲੁ / ਬਿਲਾਵਲ,
# ਭੈਰਉ / ਭੈਰੳ), so compare on the consonant skeleton.
_MARKS = "ੁੂਿੀੇੈੋੌਾਂੰ੍ਃ"


def strip_marks(token: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", token)
        if not unicodedata.combining(c) and c not in _MARKS
    )


_COMPOSITION_TERMS = """
ਰਾਗ ਰਾਗੁ ਮਹਲਾ ਮਃ ਘਰੁ ਘਰ ਕਬੀਰ ਨਾਮਦੇਵ ਰਵਿਦਾਸ ਫਰੀਦ ਬੇਣੀ ਧੰਨਾ ਸੈਣ ਪੀਪਾ ਸਧਨਾ
ਤ੍ਰਿਲੋਚਨ ਜੈਦੇਵ ਭੀਖਨ ਸੂਰਦਾਸ ਪਰਮਾਨੰਦ ਰਾਮਾਨੰਦ ਸਦਨਾ ਜੀ ਜੀਉ ਕੇ ਕੀ ਕਾ ਭਗਤ ਬਾਣੀ
ਸਲੋਕ ਸਲੋਕੁ ਪਉੜੀ ਪਵੜੀ ਛੰਤ ਛੰਤੁ ਵਾਰ ਵਾਰਾਂ ਅਸਟਪਦੀ ਅਸਟਪਦੀਆ ਡਖਣਾ ਸੋਹਿਲਾ ਦੁਪਦੇ
ਦੁਪਦਾ ਚਉਪਦੇ ਤਿਪਦੇ ਤਿਪਦਾ ਪੰਚਪਦਾ ਪੰਚਪਦੇ ਛਕਾ ਛਕੇ ਸੁਧੁ ਦੂਜਾ ਤੀਜਾ ਚਉਥਾ ਪਹਿਲਾ
ਪੰਜਵਾ ਨਾਵਾ ਦਸਵਾ ਕਾਫੀ ਦਖਣੀ ਗੁਆਰੇਰੀ ਚੇਤੀ ਬਿਭਾਸ ਪੜਤਾਲ ਅਸਟ ਸੁਖਮਨੀ ਥਿਤੀ ਥਿਤੰੀ
ਬਾਵਨ ਅਖਰੀ ਜਪੁ ਗਾਥਾ ਫੁਨਹੇ ਚਉਬੋਲੇ ਸਵਯੇ ਮੁੰਦਾਵਣੀ ਸਹਸਕ੍ਰਿਤੀ ਸੇਖ ਵਧੀਕ ਤੇ ਸ੍ਰੀ
ਮੁਖ ਬਾਕ ਮਾਲਾ ਏਕਤੁਕੇ ਏਕਤੁਕਾ ਦੁਤੁਕੇ ਦੁਤੁਕੀਆ
""".split()

_DIGITS_ONLY = re.compile(r"^[੦-੯0-9॥।]+$")
_MAX_TITLE_TOKENS = 6


def load_title_vocabulary() -> set:
    """Raag names from the metadata file + fixed composition terms."""
    vocab = {strip_marks(t) for t in _COMPOSITION_TERMS}
    if RAAG_METADATA.exists():
        raags = pd.read_csv(RAAG_METADATA, encoding="utf-8")
        for name in raags.get("raag_uni", pd.Series(dtype=str)).dropna():
            for token in str(name).replace("ਰਾਗੁ", "").replace("ਰਾਗ", "").split():
                if len(token) > 1:
                    vocab.add(strip_marks(token))
    else:
        print(f"  WARNING: {RAAG_METADATA} not found; bare raag headings "
              f"such as 'ਆਸਾ ॥' will not be detected.")
    return vocab


def make_lexicon_detector(vocab: set):
    def is_title_by_lexicon(pankati: str) -> bool:
        pankati = pankati.strip()
        if not pankati.endswith("॥"):
            return False
        tokens = [t for t in pankati.rstrip("॥ ").split() if t]
        if not tokens or len(tokens) > _MAX_TITLE_TOKENS:
            return False
        for token in tokens:
            token = token.strip("॥।")
            if not token or _DIGITS_ONLY.match(token):
                continue
            if strip_marks(token) not in vocab:
                return False
        return True

    return is_title_by_lexicon


# --------------------------------------------------------------------------

def main() -> None:
    if not SOURCE.exists():
        sys.exit(f"Source workbook not found: {SOURCE}")

    print(f"Reading {SOURCE.name} ...")
    df = pd.read_excel(SOURCE, engine="openpyxl")
    n_raw = len(df)
    print(f"  {n_raw:,} rows x {len(df.columns)} columns")

    df = df.copy()

    # --- normalise text columns (originals are left untouched on disk) ---
    text_cols = ["pankati", "source"] + [c for c in ALL_STEEKS if c in df.columns]
    for col in text_cols:
        df[col] = df[col].map(normalize)

    # --- steek counts (needed by the scripture guard below) ---
    train_cols = [c for c in TRAIN_STEEKS if c in df.columns]
    all_cols = [c for c in ALL_STEEKS if c in df.columns]
    df["n_steeks_train"] = (df[train_cols] != "").sum(axis=1)
    df["n_steeks_all"] = (df[all_cols] != "").sum(axis=1)

    # --- derived flags ---
    df["is_structural"] = pd.to_numeric(df["is_structural"], errors="coerce").fillna(0).astype(int)

    is_title_by_lexicon = make_lexicon_detector(load_title_vocabulary())
    prefix_hit = df["pankati"].map(is_header_line)
    lexicon_hit = df["pankati"].map(is_title_by_lexicon)

    # Scripture guard: a substantial line that all three commentators bothered to
    # translate is a teaching, whatever it starts with. This is what saves verses
    # like "ਰਾਗੁ ਨਾਦੁ ਸਭੁ ਸਚੁ ਹੈ ..." from the raag-heading rule.
    is_scripture = (df["pankati"].str.len() >= 30) & (df["n_steeks_train"] >= 3)

    df["is_header_line"] = ((prefix_hit | lexicon_hit) & ~is_scripture).astype(int)
    df["is_excluded"] = ((df["is_structural"] == 1) | (df["is_header_line"] == 1)).astype(int)

    dup_mask = df["pankati"].duplicated(keep=False) & (df["pankati"] != "")
    df["is_duplicate_pankati"] = dup_mask.astype(int)

    # Rows sharing identical Pankati text get a shared group id. These are
    # genuinely repeated scripture (Mool Mantar, rahao lines), NOT headers, so
    # they must not be excluded -- but Step 2 has to treat them as one unit or
    # in-batch negatives will punish the model for a correct match.
    df["pankati_group_id"] = df.groupby("pankati", sort=False).ngroup()

    # Rows usable for training pairs: real Pankati + at least one usable Steek.
    df["is_trainable"] = ((df["is_excluded"] == 0) & (df["n_steeks_train"] > 0)).astype(int)

    # --- ss/bdb duplication, measured not assumed ---
    both = (df["steek_ss"] != "") & (df["steek_bdb"] != "")
    identical = both & (df["steek_ss"] == df["steek_bdb"])

    # --- report ---
    lines = []
    add = lines.append
    add("=" * 62)
    add("GURBANI DATASET PROFILE")
    add("=" * 62)
    add(f"Source            : {SOURCE.name}")
    add(f"Total rows        : {n_raw:,}")
    add(f"Unique Pankatis   : {df['pankati'].nunique():,}")
    add("")
    add("-- exclusions ------------------------------------------------")
    add(f"is_structural = 1       : {int((df['is_structural'] == 1).sum()):,}")
    add(f"prefix-rule headings    : {int(prefix_hit.sum()):,}")
    add(f"lexicon-only headings   : {int((lexicon_hit & ~prefix_hit).sum()):,}")
    add(f"rescued by guard        : {int(((prefix_hit | lexicon_hit) & is_scripture).sum()):,}")
    add(f"excluded (total)        : {int(df['is_excluded'].sum()):,}")
    add(f"TRAINABLE rows          : {int(df['is_trainable'].sum()):,}")
    add("")
    add(f"rows sharing text       : {int(dup_mask.sum()):,}")
    kept = df[df["is_excluded"] == 0]
    kept_dup = int((kept["pankati"].duplicated(keep=False) & (kept["pankati"] != "")).sum())
    add(f"  ...still in corpus    : {kept_dup:,}  (repeated scripture -> Step 2 must group)")
    add("")
    add("-- steek availability (all rows) -----------------------------")
    for col in all_cols:
        tag = "  [train]" if col in train_cols else "  [excluded from pairs]"
        add(f"{col:<12}: {int((df[col] != '').sum()):>7,}{tag}")
    add("")
    add(f"ss & bdb both present   : {int(both.sum()):,}")
    add(f"  byte-identical        : {int(identical.sum()):,} "
        f"({100 * identical.sum() / max(both.sum(), 1):.1f}%)")
    add("")
    add("-- Steeks per TRAINABLE Pankati (ss / ft / ms) ---------------")
    tr = df[df["is_trainable"] == 1]
    hist = df["n_steeks_train"].value_counts().sort_index(ascending=False)
    for k in [3, 2, 1, 0]:
        add(f"{k} Steeks             : {int(hist.get(k, 0)):>7,}")
    add("")
    add(f"Estimated Pankati<->Steek positive pairs: {int(tr['n_steeks_train'].sum()):,}")
    add("")
    add("-- metadata --------------------------------------------------")
    for col in ["ang", "shabad_id", "verse_id", "raag_id"]:
        if col in df.columns:
            missing = int(df[col].isna().sum())
            add(f"{col:<12}: {missing:,} missing, {df[col].nunique():,} distinct")
    add("=" * 62)

    report = "\n".join(lines)
    print("\n" + report)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_TXT.write_text(report, encoding="utf-8")

    # Everything flagged, so the regex can be audited before Step 2 trusts it.
    df["_prefix_rule"] = prefix_hit.astype(int)
    df["_lexicon_rule"] = lexicon_hit.astype(int)
    review = (
        df.loc[df["is_excluded"] == 1,
               ["id", "ang", "pankati", "is_structural", "_prefix_rule",
                "_lexicon_rule", "is_duplicate_pankati", "n_steeks_train"]]
        .sort_values(["n_steeks_train", "ang"], ascending=[False, True])
    )
    df = df.drop(columns=["_prefix_rule", "_lexicon_rule"])
    review.to_csv(REVIEW_CSV, index=False, encoding="utf-8-sig")

    df.to_csv(CLEANED_CSV, index=False, encoding="utf-8-sig")

    print(f"\nWrote {CLEANED_CSV}")
    print(f"Wrote {REPORT_TXT}")
    print(f"Wrote {REVIEW_CSV}  ({len(review):,} rows to review)")


if __name__ == "__main__":
    main()
