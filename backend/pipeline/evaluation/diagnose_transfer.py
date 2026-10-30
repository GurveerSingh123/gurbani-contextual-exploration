"""
Diagnostic: did fine-tuning improve the task it was actually TRAINED on?

We train on Steek -> Pankati but evaluate on Pankati -> Pankati. When the
Pankati->Pankati score does not move, there are two very different causes:

  training worked, but does not transfer   -> train on verse pairs directly
  training did not work at all             -> fix training before mining data

This measures the trained task itself, so the two can be told apart.

    python backend/pipeline/evaluation/diagnose_transfer.py --tag baseline
    python backend/pipeline/evaluation/diagnose_transfer.py --tag finetuned --model models/bge-m3-gurbani

Reads  : data/processed/test_pairs.jsonl
         data/embeddings/<tag>/{embeddings.npy,corpus_meta.jsonl}
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(PROJECT_ROOT / ".env")

PROCESSED = PROJECT_ROOT / os.getenv("PROCESSED_DATA_DIR", "data/processed")
EMBED_DIR = PROJECT_ROOT / os.getenv("EMBEDDING_DIR", "data/embeddings")
SEED = int(os.getenv("SEED", "42"))
K_VALUES = [1, 5, 10, 50]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="baseline", help="which corpus embeddings to search")
    ap.add_argument("--model", default=None,
                    help="model used to encode the Steek queries "
                         "(default: the base model for tag 'baseline')")
    ap.add_argument("--n", type=int, default=1000, help="queries to sample")
    ap.add_argument("--pairs", default=None,
                    help="pairs file (default: test_pairs.jsonl, else val_pairs.jsonl). "
                         "Both are held out from training.")
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    emb_path = EMBED_DIR / args.tag / "embeddings.npy"
    meta_path = EMBED_DIR / args.tag / "corpus_meta.jsonl"

    if args.pairs:
        pairs_path = Path(args.pairs)
    else:
        pairs_path = PROCESSED / "test_pairs.jsonl"
        if not pairs_path.exists():
            pairs_path = PROCESSED / "val_pairs.jsonl"
            print("test_pairs.jsonl not found -- using val_pairs.jsonl "
                  "(also held out from training)")

    for p in (emb_path, meta_path, pairs_path):
        if not p.exists():
            sys.exit(f"Missing {p}")

    emb = np.load(emb_path)
    meta = [json.loads(l) for l in meta_path.open(encoding="utf-8")]
    by_text = {m["pankati"]: i for i, m in enumerate(meta)}
    print(f"corpus: {len(meta):,} verses, tag '{args.tag}'")

    pairs = [json.loads(l) for l in pairs_path.open(encoding="utf-8")]
    # One query per gold verse, so a verse with three Steeks cannot dominate.
    seen, sampled = set(), []
    random.Random(SEED).shuffle(pairs)
    for rec in pairs:
        gold = rec["pos"][0]
        if gold in seen or gold not in by_text:
            continue
        seen.add(gold)
        sampled.append((rec["query"], by_text[gold], rec["steek_source"]))
        if len(sampled) >= args.n:
            break
    print(f"queries: {len(sampled):,} Steeks (held-out test split)")

    model_name = args.model or os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
    print(f"encoding queries with {model_name} ...")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(str(model_name))
    model.max_seq_length = 128
    qv = model.encode([q for q, _, _ in sampled], batch_size=args.batch_size,
                      normalize_embeddings=True, convert_to_numpy=True,
                      show_progress_bar=True).astype(np.float32)

    scores = qv @ emb.T
    order = np.argsort(-scores, axis=1)[:, :max(K_VALUES)]
    gold_idx = np.array([g for _, g, _ in sampled])

    print(f"\n{'=' * 58}")
    print(f"STEEK -> PANKATI  (the task the model was trained on)")
    print(f"{'=' * 58}")
    print(f"model: {model_name}   corpus: {args.tag}")
    ranks = []
    for row, gold in zip(order, gold_idx):
        hit = np.flatnonzero(row == gold)
        ranks.append(int(hit[0]) + 1 if hit.size else 10 ** 9)
    ranks = np.array(ranks)
    for k in K_VALUES:
        print(f"  H@{k:<3}: {100 * (ranks <= k).mean():6.2f}%")
    print(f"  MRR  : {100 * np.mean(1.0 / ranks):6.2f}")

    print("\n  by Steek source:")
    for src in ["ss", "ft", "ms"]:
        mask = np.array([s == src for _, _, s in sampled])
        if mask.any():
            print(f"    {src}: H@10 {100 * (ranks[mask] <= 10).mean():6.2f}%  (n={int(mask.sum())})")
    print("=" * 58)


if __name__ == "__main__":
    main()
