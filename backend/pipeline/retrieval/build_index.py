"""
Step 3b (part 1): encode the Pankati corpus into vectors.

Designed for a slow CPU: work is written to disk in shards as it completes, so
an interrupted run resumes instead of starting over. Encoding all 54,844 verses
on a 2-core laptop takes hours; it is a ONE-TIME cost per model.

  python backend/pipeline/retrieval/build_index.py                  # pretrained baseline
  python backend/pipeline/retrieval/build_index.py --model models/bge-m3-gurbani --tag finetuned

Reads  : data/processed/corpus_pankatis.jsonl
Writes : data/embeddings/<tag>/part_*.npy   (shards, resumable)
         data/embeddings/<tag>/embeddings.npy
         data/embeddings/<tag>/corpus_meta.jsonl
         data/embeddings/<tag>/model.json         (which model built this tag)
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(PROJECT_ROOT / ".env")

PROCESSED = PROJECT_ROOT / os.getenv("PROCESSED_DATA_DIR", "data/processed")
EMBED_DIR = PROJECT_ROOT / os.getenv("EMBEDDING_DIR", "data/embeddings")
CORPUS = PROCESSED / "corpus_pankatis.jsonl"

SHARD = 2000          # verses per checkpoint file
MAX_SEQ = 128         # Pankatis are short; BGE-M3 defaults to 8192 and would crawl


def load_corpus():
    if not CORPUS.exists():
        sys.exit(f"Missing {CORPUS}. Run create_training_pairs.py first.")
    docs = [json.loads(line) for line in CORPUS.open(encoding="utf-8")]
    # Keep one representative occurrence for metadata; the verse text is the key.
    meta = []
    for d in docs:
        first = d["occurrences"][0]
        meta.append({
            "pankati": d["pankati"],
            "pankati_group_id": d["pankati_group_id"],
            "ids": [o["id"] for o in d["occurrences"]],
            "shabad_ids": sorted({o["shabad_id"] for o in d["occurrences"]}),
            "ang": first["ang"],
            "raag_id": first["raag_id"],
        })
    return meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3"))
    ap.add_argument("--tag", default="baseline",
                    help="subfolder under data/embeddings/")
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()

    meta = load_corpus()
    texts = [m["pankati"] for m in meta]
    print(f"corpus: {len(texts):,} unique Pankatis")

    out_dir = EMBED_DIR / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "corpus_meta.jsonl").open("w", encoding="utf-8") as f:
        for m in meta:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")

    # Record which model built this tag. A free-text query must be encoded by the
    # same model as the corpus, or its vector is not comparable to the index.
    model_ref = args.model
    local = Path(args.model).resolve()
    if local.exists():
        try:
            model_ref = local.relative_to(PROJECT_ROOT).as_posix()
        except ValueError:
            model_ref = local.as_posix()
    (out_dir / "model.json").write_text(
        json.dumps({"model": model_ref}, indent=2) + "\n", encoding="utf-8")

    shards = [(i, min(i + SHARD, len(texts))) for i in range(0, len(texts), SHARD)]
    todo = [(s, e) for s, e in shards if not (out_dir / f"part_{s:06d}.npy").exists()]
    print(f"{len(shards) - len(todo)}/{len(shards)} shards already done")

    if todo:
        # Imported here, not at module load: writing corpus_meta.jsonl above must
        # work without sentence-transformers, so BM25 can be scored while the
        # (slow, optional) encoding is still being set up.
        from sentence_transformers import SentenceTransformer

        print(f"loading {args.model} ... (first run downloads ~2.3 GB)")
        model = SentenceTransformer(args.model)
        model.max_seq_length = MAX_SEQ

        started = time.time()
        for n, (s, e) in enumerate(todo, 1):
            vecs = model.encode(
                texts[s:e],
                batch_size=args.batch_size,
                normalize_embeddings=True,     # cosine == dot product
                convert_to_numpy=True,
                show_progress_bar=False,
            ).astype(np.float32)
            np.save(out_dir / f"part_{s:06d}.npy", vecs)
            done = n / len(todo)
            elapsed = time.time() - started
            eta = elapsed / done - elapsed
            print(f"  shard {n}/{len(todo)}  {100 * done:5.1f}%  "
                  f"elapsed {elapsed / 60:.1f}m  eta {eta / 60:.1f}m")

    parts = [np.load(out_dir / f"part_{s:06d}.npy") for s, _ in shards]
    embeddings = np.vstack(parts)
    assert len(embeddings) == len(texts), "shard/corpus size mismatch"
    np.save(out_dir / "embeddings.npy", embeddings)
    print(f"\nWrote {out_dir / 'embeddings.npy'}  shape={embeddings.shape}")
    print("Shards kept for resume; delete part_*.npy to reclaim space.")


if __name__ == "__main__":
    main()
