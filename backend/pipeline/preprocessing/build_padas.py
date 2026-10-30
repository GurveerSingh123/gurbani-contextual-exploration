"""
Pada step 1: group the Pankatis into padas.

A single Pankati is often half a thought. The complete unit in Gurbani is the
pada: the lines of a Shabad up to a numbered end marker (॥੧॥, ॥੨॥ ...), with the
Rahao line as a unit of its own. Padas average 4 lines / ~48 model tokens, so
they still fit the encoder (99.9% within 256 tokens).

Each pada keeps its lines, so a search over padas can still point at the exact
line. Steeks are joined line by line, so the pada's meaning can be mined from
the commentaries exactly as the line-level gold set was.

The train/val/test split carries over unchanged: the split is grouped by Shabad,
so every line of a pada is in the same part. That is asserted, not assumed.

  python backend/pipeline/preprocessing/build_padas.py

Reads  : data/processed/corpus_pankatis.jsonl, row_splits.csv
Writes : data/processed/padas.jsonl
         data/processed/padas_report.txt
"""

import csv
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(PROJECT_ROOT / ".env")

PROCESSED = PROJECT_ROOT / os.getenv("PROCESSED_DATA_DIR", "data/processed")
CORPUS_JSONL = PROCESSED / "corpus_pankatis.jsonl"
ROW_SPLITS = PROCESSED / "row_splits.csv"
PADAS_JSONL = PROCESSED / "padas.jsonl"
REPORT_TXT = PROCESSED / "padas_report.txt"

# A pada ends at a numbered marker (॥੧॥, ॥੨॥ ... also ॥੧॥ ਰਹਾਉ ॥) or a Rahao line.
PADA_END = re.compile(r"॥\s*[੦-੯0-9]+\s*॥|ਰਹਾਉ")
STEEKS = ["steek_ss", "steek_ms", "steek_ft"]


def main() -> None:
    for p in (CORPUS_JSONL, ROW_SPLITS):
        if not p.exists():
            sys.exit(f"Missing {p}. Run create_training_pairs.py first.")

    docs = [json.loads(l) for l in CORPUS_JSONL.open(encoding="utf-8")]
    split_of = {}
    with ROW_SPLITS.open(encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            split_of[int(r["id"])] = r["split"]

    # Every occurrence of every verse, placed back into its Shabad in order.
    # Lines with no split are the 66 with no Steek at all (Raagmala, heading
    # fragments such as "ਦੁਤੁਕੇ"): nothing to judge them by, so they join no pada.
    lines_of = defaultdict(list)          # shabad_id -> [(verse_id, corpus idx, occurrence)]
    skipped = 0
    for i, d in enumerate(docs):
        for occ in d["occurrences"]:
            if occ["id"] not in split_of:
                skipped += 1
                continue
            lines_of[occ["shabad_id"]].append((occ["verse_id"], i, occ))

    padas = []
    for sid in sorted(lines_of):
        group = []
        ordered = sorted(lines_of[sid], key=lambda t: t[0])
        for n, (_, i, occ) in enumerate(ordered):
            group.append((i, occ))
            if PADA_END.search(docs[i]["pankati"]) or n == len(ordered) - 1:
                padas.append((sid, group))
                group = []

    records, mixed = [], 0
    for pid, (sid, group) in enumerate(padas):
        splits = {split_of.get(occ["id"]) for _, occ in group}
        if len(splits) != 1:
            mixed += 1
        rec = {
            "pada_id": pid,
            "shabad_id": sid,
            "ang": group[0][1]["ang"],
            "raag_id": group[0][1]["raag_id"],
            "split": next(iter(splits)),
            "n_lines": len(group),
            "line_index": [i for i, _ in group],          # rows of corpus_pankatis / embeddings
            "line_ids": [occ["id"] for _, occ in group],  # occurrence ids
            "lines": [docs[i]["pankati"] for i, _ in group],
            "text": " ".join(docs[i]["pankati"] for i, _ in group),
        }
        for col in STEEKS:
            rec[col] = " ".join((occ.get(col) or "").strip() for _, occ in group).strip()
        records.append(rec)

    if mixed:
        sys.exit(f"{mixed} padas span more than one split -- the split is not Shabad-grouped.")

    with PADAS_JSONL.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # --- report -----------------------------------------------------------------
    sizes = Counter(r["n_lines"] for r in records)
    by_split = Counter(r["split"] for r in records)
    has = {c: sum(1 for r in records if r[c]) for c in STEEKS}
    dup_text = len(records) - len({r["text"] for r in records})
    lines = [
        "=" * 62, "PADAS  (lines up to a ॥n॥ marker or a Rahao line)", "=" * 62,
        f"Lines left out     : {skipped} (no Steek and no split: Raagmala, heading fragments)",
        f"Shabads            : {len(lines_of):,}",
        f"Padas              : {len(records):,}",
        f"Lines per pada     : " + ", ".join(f"{k}: {sizes[k]:,}" for k in sorted(sizes)[:8])
        + (" ..." if len(sizes) > 8 else ""),
        f"Mean lines / pada  : {sum(r['n_lines'] for r in records) / len(records):.2f}",
        f"Identical text elsewhere : {dup_text:,} padas",
        "",
        "Split (Shabad-grouped, so every pada is in one part):",
        *[f"  {k:<6}: {v:,}" for k, v in sorted(by_split.items())],
        "",
        "Padas with each Steek present:",
        *[f"  {c:<9}: {v:,} ({100 * v / len(records):.1f}%)" for c, v in has.items()],
        "=" * 62,
    ]
    report = "\n".join(lines)
    print(report)
    REPORT_TXT.write_text(report, encoding="utf-8")
    print(f"\nWrote {PADAS_JSONL}  ({len(records):,} padas)")
    print(f"Wrote {REPORT_TXT}")


if __name__ == "__main__":
    main()
