"""
Build the hand-labelling sheets: an independent check on the mined gold set.

Every score so far comes from a benchmark mined from the same Steeks the model
trained on. A set judged by people breaks that link, and also counts correct
matches the mined set never knew about.

For each of N query verses (random, from the TEST split only, not just from the
mined gold set, so the choice cannot favour the model) this pools the top 10 of
the final model and the top 10 of char-BM25, removes duplicates, and shuffles
them. Labellers never see which system found a verse; that goes to a separate
answer key.

Results follow the product's rules: other Shabads only, one verse per Shabad.

The 50 queries are split across the three team members, and 10 of them are
labelled twice (rotated A-B, B-C, C-A) to measure agreement.

  python backend/pipeline/evaluation/build_label_sheet.py

Reads  : data/processed/corpus_pankatis.jsonl, row_splits.csv
         data/embeddings/<SEARCH_TAG>/{embeddings.npy,corpus_meta.jsonl}
Writes : data/labeling/label_sheet_<name>.xlsx      (one per labeller)
         data/labeling/answer_key_DO_NOT_OPEN.csv   (which system found what)
"""

import argparse
import csv
import json
import os
import random
import re
import sys
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(PROJECT_ROOT / ".env")

PROCESSED = PROJECT_ROOT / os.getenv("PROCESSED_DATA_DIR", "data/processed")
EMBED_DIR = PROJECT_ROOT / os.getenv("EMBEDDING_DIR", "data/embeddings")
OUT_DIR = PROJECT_ROOT / os.getenv("DATA_DIR", "data") / "labeling"
CORPUS_JSONL = PROCESSED / "corpus_pankatis.jsonl"
ROW_SPLITS = PROCESSED / "row_splits.csv"
TAG = os.getenv("SEARCH_TAG", "verse_hn")
SEED = int(os.getenv("SEED", "42"))

TOP = 10
MIN_WORDS = 4            # very short lines ("ਰਹਾਉ ॥" fragments) are not judgeable
LABELLERS = ["Balkaran", "Gurpreet", "Gurveer"]

# --- tokenisers: identical to evaluate.py, so the baseline matches the reported one
_STRIP = re.compile(r"[॥।0-9੦-੯]")
CHAR_N = (3, 4, 5)


def tokenize(text: str):
    return [w for w in _STRIP.sub(" ", text).split() if len(w) > 1]


def char_tokenize(text: str):
    cleaned = "_".join(tokenize(text))
    return [cleaned[i:i + n] for n in CHAR_N for i in range(len(cleaned) - n + 1)]


def top_k(scores, own_shabads, shabad_of, k):
    """Best k other-Shabad verses, at most one per Shabad (the product's rules)."""
    order = np.argsort(-scores)
    out, used = [], set()
    for i in order:
        if scores[i] <= 0:          # sorted, so nothing after this matches either
            break
        shabads = shabad_of[i]
        if shabads & own_shabads or min(shabads) in used:
            continue
        used.add(min(shabads))
        out.append(int(i))
        if len(out) >= k:
            break
    return out


def assignment(n):
    """Query index -> list of labellers. Base split in thirds; 10 queries doubled."""
    third = [n * j // 3 for j in range(4)]
    who = {q: [LABELLERS[j]] for j in range(3) for q in range(third[j], third[j + 1])}
    # 10 queries labelled twice, rotated so the extra load is spread evenly:
    # A's first 4 also go to B, B's first 3 to C, C's first 3 to A.
    for q in range(third[0], third[0] + 4):
        who[q].append(LABELLERS[1])
    for q in range(third[1], third[1] + 3):
        who[q].append(LABELLERS[2])
    for q in range(third[2], third[2] + 3):
        who[q].append(LABELLERS[0])
    return who


INSTRUCTIONS = [
    ("What this is", "Each row pairs a QUERY verse with a CANDIDATE verse found by a search system. "
                     "Judge whether they carry the same meaning (Bhavarth)."),
    ("Your task", "In the 'label' column choose 2, 1 or 0 for every row. Add a note if unsure."),
    ("2 = same meaning", "Both verses carry the same teaching, even if the words are completely different."),
    ("1 = related", "Same theme, but the teaching differs in scope or emphasis "
                    "(e.g. both about Naam, but one about its glory and one about forgetting it)."),
    ("0 = not related", "Different meaning, including verses that share words but teach something "
                        "different or opposite (e.g. 'the cycle of birth and death is STOPPED' vs "
                        "'the cycle NEVER ends')."),
    ("Judge by meaning", "Shared words do not make a match. Use the Sahib Singh Steeks shown beside "
                         "each verse when you are unsure."),
    ("Do not look it up", "Do not try to find out which system found a verse. Rows are shuffled on purpose."),
    ("Work alone", "Do not discuss labels with teammates until everyone has finished; some queries "
                   "are labelled twice to measure agreement."),
    ("When done", "Save the file under the same name and hand it back. Empty labels are skipped."),
]


def write_workbook(path, name, rows):
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.worksheet.datavalidation import DataValidation

    gur = Font(name="Nirmala UI", size=12)
    gur_b = Font(name="Nirmala UI", size=12, bold=True)
    head = Font(name="Calibri", size=11, bold=True, color="FFFFFF")
    head_fill = PatternFill("solid", fgColor="16213F")
    bands = [PatternFill("solid", fgColor="FFFFFF"), PatternFill("solid", fgColor="EEF1F7")]
    label_fill = PatternFill("solid", fgColor="FBEEDD")
    wrap = Alignment(wrap_text=True, vertical="top")

    wb = Workbook()
    ins = wb.active
    ins.title = "Instructions"
    ins["A1"] = f"Labelling sheet for {name}"
    ins["A1"].font = Font(size=16, bold=True)
    ins["A2"] = (f"{len({r['query_no'] for r in rows})} queries, {len(rows)} rows. "
                 "Open the 'Labels' sheet when you have read this.")
    for i, (k, v) in enumerate(INSTRUCTIONS, start=4):
        ins.cell(i, 1, k).font = Font(bold=True)
        ins.cell(i, 2, v).alignment = Alignment(wrap_text=True, vertical="top")
    ins.column_dimensions["A"].width = 22
    ins.column_dimensions["B"].width = 100

    ws = wb.create_sheet("Labels")
    cols = [("row_id", 8), ("query_no", 9), ("query verse", 40), ("query meaning (Sahib Singh)", 45),
            ("candidate verse", 40), ("candidate meaning (Sahib Singh)", 45), ("candidate Ang", 10),
            ("label (2/1/0)", 12), ("note (optional)", 30)]
    for c, (title, width) in enumerate(cols, start=1):
        cell = ws.cell(1, c, title)
        cell.font, cell.fill, cell.alignment = head, head_fill, Alignment(wrap_text=True, vertical="center")
        ws.column_dimensions[cell.column_letter].width = width
    ws.freeze_panes = "C2"

    dv = DataValidation(type="list", formula1='"2,1,0"', allow_blank=True,
                        errorTitle="Use 2, 1 or 0", error="2 = same meaning, 1 = related, 0 = not related")
    ws.add_data_validation(dv)

    band, last_q = 0, None
    for r, row in enumerate(rows, start=2):
        if row["query_no"] != last_q:
            band, last_q = 1 - band, row["query_no"]
        values = [row["row_id"], row["query_no"], row["query"], row["query_ss"],
                  row["candidate"], row["candidate_ss"], row["candidate_ang"], None, None]
        for c, v in enumerate(values, start=1):
            cell = ws.cell(r, c, v)
            cell.alignment = wrap
            cell.fill = bands[band]
            if c in (3, 5):
                cell.font = gur_b
            elif c in (4, 6):
                cell.font = gur
        ws.cell(r, 8).fill = label_fill
        ws.cell(r, 8).alignment = Alignment(horizontal="center", vertical="top")
        dv.add(ws.cell(r, 8))
    wb.save(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", type=int, default=50)
    args = ap.parse_args()

    for p in (CORPUS_JSONL, ROW_SPLITS, EMBED_DIR / TAG / "embeddings.npy"):
        if not p.exists():
            sys.exit(f"Missing {p}.")
    docs = [json.loads(l) for l in CORPUS_JSONL.open(encoding="utf-8")]
    meta = [json.loads(l) for l in (EMBED_DIR / TAG / "corpus_meta.jsonl").open(encoding="utf-8")]
    if docs[0]["pankati"] != meta[0]["pankati"] or len(docs) != len(meta):
        sys.exit("corpus order mismatch -- rebuild the index.")
    emb = np.load(EMBED_DIR / TAG / "embeddings.npy")
    shabad_of = [set(m["shabad_ids"]) for m in meta]

    split = {}
    with ROW_SPLITS.open(encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            split[int(r["id"])] = r["split"]

    def ss(i):
        return (docs[i]["occurrences"][0].get("steek_ss") or "").strip()

    # Query pool: every occurrence in the test split, judgeable length, has a Steek.
    pool = [i for i, d in enumerate(docs)
            if all(split.get(o["id"]) == "test" for o in d["occurrences"])
            and len(tokenize(d["pankati"])) >= MIN_WORDS and ss(i)]
    rng = random.Random(SEED)
    queries = rng.sample(pool, args.queries)
    print(f"test-split verses eligible as queries: {len(pool):,}; sampled {len(queries)}")

    from rank_bm25 import BM25Okapi
    print("building char-BM25 index (a minute or two) ...")
    bm25 = BM25Okapi([char_tokenize(d["pankati"]) for d in docs])

    rows, key, row_id = [], [], 0
    for qn, qi in enumerate(queries, start=1):
        own = shabad_of[qi]
        dense = top_k(emb @ emb[qi], own, shabad_of, TOP)
        lexical = top_k(bm25.get_scores(char_tokenize(docs[qi]["pankati"])), own, shabad_of, TOP)
        pooled = list(dict.fromkeys(dense + lexical))
        rng.shuffle(pooled)
        for ci in pooled:
            row_id += 1
            rows.append({"row_id": row_id, "query_no": qn, "query": docs[qi]["pankati"],
                         "query_ss": ss(qi), "candidate": docs[ci]["pankati"],
                         "candidate_ss": ss(ci), "candidate_ang": docs[ci]["occurrences"][0]["ang"]})
            key.append({"row_id": row_id, "query_no": qn, "query_index": qi, "candidate_index": ci,
                        "dense_rank": dense.index(ci) + 1 if ci in dense else "",
                        "bm25_char_rank": lexical.index(ci) + 1 if ci in lexical else ""})
        if qn % 10 == 0:
            print(f"  {qn}/{len(queries)} queries pooled")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    who = assignment(len(queries))
    for name in LABELLERS:
        mine = [r for r in rows if name in who[r["query_no"] - 1]]
        path = OUT_DIR / f"label_sheet_{name}.xlsx"
        write_workbook(path, name, mine)
        print(f"wrote {path.name}: {len({r['query_no'] for r in mine})} queries, {len(mine)} rows")

    with (OUT_DIR / "answer_key_DO_NOT_OPEN.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(key[0].keys()) + ["labellers"])
        w.writeheader()
        for k in key:
            w.writerow({**k, "labellers": "+".join(who[k["query_no"] - 1])})

    both = sum(1 for k in key if k["dense_rank"] and k["bm25_char_rank"])
    print(f"\n{len(rows)} rows in total, {len(rows) / len(queries):.1f} per query; "
          f"{both} candidates found by both systems")
    print(f"double-labelled queries: {sum(1 for v in who.values() if len(v) > 1)}")


if __name__ == "__main__":
    main()
