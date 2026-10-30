"""
Step 2: build contrastive training pairs from the cleaned dataset.

Direction: the STEEK is the query and the PANKATI is the passage. This mirrors
inference -- the user types modern Punjabi and the system returns Gurmukhi
verses -- so the Pankati is what ends up in the FAISS index.

Negatives are deliberately NOT built here. Hard negatives are mined in Step 4
using the baseline BGE-M3 encoder, so v1 trains on in-batch negatives only and
we get a clean v1-vs-v2 ablation.

Reads  : data/processed/cleaned_gurbani.csv
Writes : data/processed/training_pairs.jsonl
         data/processed/val_pairs.jsonl
         data/processed/test_pairs.jsonl
         data/processed/corpus_pankatis.jsonl
         data/processed/pairs_report.txt
"""

import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(PROJECT_ROOT / ".env")

OUT_DIR = PROJECT_ROOT / os.getenv("PROCESSED_DATA_DIR", "data/processed")
CLEANED_CSV = OUT_DIR / "cleaned_gurbani.csv"

TRAIN_STEEKS = [c.strip() for c in os.getenv("TRAIN_STEEK_COLS", "").split(",") if c.strip()]
SEED = int(os.getenv("SEED", "42"))
MIN_STEEK_CHARS = int(os.getenv("MIN_STEEK_CHARS", "15"))
FRACS = {
    "train": float(os.getenv("TRAIN_FRAC", "0.80")),
    "val": float(os.getenv("VAL_FRAC", "0.10")),
    "test": float(os.getenv("TEST_FRAC", "0.10")),
}

OUTPUTS = {
    "train": OUT_DIR / "training_pairs.jsonl",
    "val": OUT_DIR / "val_pairs.jsonl",
    "test": OUT_DIR / "test_pairs.jsonl",
}
CORPUS_JSONL = OUT_DIR / "corpus_pankatis.jsonl"
ROW_SPLITS_CSV = OUT_DIR / "row_splits.csv"
REPORT_TXT = OUT_DIR / "pairs_report.txt"


# --------------------------------------------------------------------------
# Leakage groups
#
# Two rows must not straddle a split boundary if they are (a) lines of the same
# Shabad, or (b) the same Pankati text repeated elsewhere in the Granth. Either
# would let a near-identical verse appear in both train and val, inflating the
# fine-tuned score and making the Phase 8 baseline comparison meaningless.
# Union the two relations so the transitive closure is respected.
# --------------------------------------------------------------------------

class UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, x):
        self.parent.setdefault(x, x)
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:      # path compression
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def build_leak_groups(df: pd.DataFrame) -> pd.Series:
    uf = UnionFind()
    for shabad, group in zip(df["shabad_id"], df["pankati_group_id"]):
        uf.union(("S", shabad), ("P", group))

    # Third relation: the same Steek text sometimes annotates two different
    # Pankatis. Without this, an identical query lands in train with one gold
    # Pankati and in val with another -- the model is then scored on a query it
    # was trained to answer differently. Costs ~0.9% on the largest group.
    for col in TRAIN_STEEKS:
        for shabad, steek in zip(df["shabad_id"], df[col]):
            if len(steek) >= MIN_STEEK_CHARS:
                uf.union(("S", shabad), ("T", steek))

    # A root may be a ("T", <long steek>) key, so map roots to compact integer
    # ids -- deterministic, and keeps the JSONL records small.
    roots = df["shabad_id"].map(lambda s: uf.find(("S", s)))
    ids = {root: i for i, root in enumerate(sorted(set(roots), key=str))}
    return roots.map(ids)


def assign_splits(sizes: dict) -> dict:
    """Greedy bin-packing: largest group first, into whichever split is
    furthest below its quota. Deterministic, no RNG needed."""
    total = sum(sizes.values())
    targets = {name: frac * total for name, frac in FRACS.items()}
    current = {name: 0 for name in FRACS}
    assignment = {}

    # Tie-break on the group key so the result is stable across runs.
    ordered = sorted(sizes.items(), key=lambda kv: (-kv[1], str(kv[0])))
    for group, size in ordered:
        pick = max(current, key=lambda n: (targets[n] - current[n]) / max(targets[n], 1))
        assignment[group] = pick
        current[pick] += size
    return assignment


# --------------------------------------------------------------------------

def main() -> None:
    if not CLEANED_CSV.exists():
        sys.exit(f"Run clean_dataset.py first -- {CLEANED_CSV} not found.")

    df = pd.read_csv(CLEANED_CSV, encoding="utf-8-sig", low_memory=False)
    for col in TRAIN_STEEKS + ["pankati"]:
        df[col] = df[col].fillna("").astype(str)

    corpus_rows = df[df["is_excluded"] == 0]
    trainable = df[df["is_trainable"] == 1].copy()
    print(f"{len(df):,} rows -> {len(trainable):,} trainable")

    trainable["leak_group"] = build_leak_groups(trainable)
    group_sizes = trainable.groupby("leak_group").size().to_dict()
    split_of = assign_splits(group_sizes)
    trainable["split"] = trainable["leak_group"].map(split_of)

    # --- emit pairs -------------------------------------------------------
    pairs = {name: [] for name in FRACS}
    seen = {name: set() for name in FRACS}
    dropped_short = 0
    dropped_dupe = 0
    per_source = defaultdict(int)

    for row in trainable.itertuples(index=False):
        pankati = row.pankati
        split = row.split
        for col in TRAIN_STEEKS:
            steek = getattr(row, col)
            if not steek:
                continue
            if len(steek) < MIN_STEEK_CHARS:
                dropped_short += 1
                continue
            key = (steek, pankati)
            if key in seen[split]:
                dropped_dupe += 1
                continue
            seen[split].add(key)
            source = col.replace("steek_", "")
            per_source[source] += 1
            pairs[split].append({
                "query": steek,
                "pos": [pankati],
                "neg": [],
                "steek_source": source,
                "pankati_id": int(row.id),
                "shabad_id": int(row.shabad_id),
                "ang": int(row.ang),
                "leak_group": int(row.leak_group),
            })

    for name, path in OUTPUTS.items():
        with path.open("w", encoding="utf-8") as f:
            for rec in pairs[name]:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Row -> split map, so evaluation can draw queries from the test split
    # without re-deriving the grouping (and risking drift).
    trainable[["id", "shabad_id", "leak_group", "split"]].to_csv(
        ROW_SPLITS_CSV, index=False, encoding="utf-8-sig"
    )

    # --- emit the retrieval corpus ---------------------------------------
    # Deduplicated by Pankati text: repeated scripture is ONE searchable
    # document, with every occurrence listed so metadata is not lost.
    corpus = {}
    for row in corpus_rows.itertuples(index=False):
        entry = corpus.setdefault(row.pankati, {
            "pankati": row.pankati,
            "pankati_group_id": int(row.pankati_group_id),
            "occurrences": [],
        })
        # Steeks travel with the document: a search result has to show the
        # meaning alongside the verse, and Phase 9 feeds them to the LLM.
        # pss is a padd-arth glossary -- useless for training, useful on screen.
        entry["occurrences"].append({
            "id": int(row.id),
            "ang": int(row.ang),
            "shabad_id": int(row.shabad_id),
            "verse_id": int(row.verse_id),
            "raag_id": None if pd.isna(row.raag_id) else int(row.raag_id),
            "steek_ss": row.steek_ss,
            "steek_ft": row.steek_ft,
            "steek_ms": row.steek_ms,
            "steek_pss": "" if pd.isna(row.steek_pss) else str(row.steek_pss),
        })
    with CORPUS_JSONL.open("w", encoding="utf-8") as f:
        for entry in corpus.values():
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # --- report -----------------------------------------------------------
    lines = []
    add = lines.append
    add("=" * 62)
    add("TRAINING PAIR CONSTRUCTION")
    add("=" * 62)
    add("Direction  : query = Steek, passage = Pankati")
    add(f"Steeks used: {', '.join(TRAIN_STEEKS)}")
    add(f"Negatives  : none (in-batch only; hard negatives mined in Step 4)")
    add("")
    add(f"Trainable rows      : {len(trainable):,}")
    add(f"Leakage groups      : {len(group_sizes):,}")
    add(f"  largest group     : {max(group_sizes.values()):,} rows "
        f"({100 * max(group_sizes.values()) / len(trainable):.2f}% of corpus)")
    add("")
    add("-- split (grouped by shabad_id + repeated Pankati text) ------")
    total_pairs = sum(len(v) for v in pairs.values())
    for name in ["train", "val", "test"]:
        rows = int((trainable["split"] == name).sum())
        add(f"{name:<6}: {rows:>7,} rows  {len(pairs[name]):>7,} pairs "
            f"({100 * len(pairs[name]) / max(total_pairs, 1):5.1f}%)")
    add(f"{'TOTAL':<6}: {len(trainable):>7,} rows  {total_pairs:>7,} pairs")
    add("")
    add("-- pairs by Steek source -------------------------------------")
    for source, count in sorted(per_source.items(), key=lambda kv: -kv[1]):
        add(f"  {source:<4}: {count:>7,}")
    add("")
    add(f"dropped, steek < {MIN_STEEK_CHARS} chars : {dropped_short:,}")
    add(f"dropped, duplicate pair    : {dropped_dupe:,}")
    add("")
    add(f"Retrieval corpus (unique Pankatis): {len(corpus):,}")
    add("=" * 62)

    report = "\n".join(lines)
    print("\n" + report)
    REPORT_TXT.write_text(report, encoding="utf-8")

    for name, path in OUTPUTS.items():
        print(f"Wrote {path}  ({len(pairs[name]):,} pairs)")
    print(f"Wrote {CORPUS_JSONL}  ({len(corpus):,} documents)")
    print(f"Wrote {REPORT_TXT}")


if __name__ == "__main__":
    main()
