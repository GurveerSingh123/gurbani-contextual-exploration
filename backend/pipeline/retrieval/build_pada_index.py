"""
Pada step 3: encode every pada's full text into a vector.

This is the "read the whole pada as one input" version of the context idea --
unlike averaging line vectors, the model sees the lines together and can let
them qualify each other.

Long on a CPU (~1.5 h for 18.5k padas on the laptop, ~3 min on the RTX 3060),
so it works in shards of 500 padas: progress and time left are printed after
each shard, and an interrupted run continues where it stopped when restarted
with the same command.

MAX_SEQ is 256 here (128 for single lines): padas have a median of 48 tokens
and 99.9% fit in 256; a longer limit only adds padding.

  python backend/pipeline/retrieval/build_pada_index.py --model models/bge-m3-verse-hn --tag verse_hn
  python backend/pipeline/retrieval/build_pada_index.py --model BAAI/bge-m3 --tag baseline

Reads  : data/processed/padas.jsonl
Writes : data/embeddings/pada_<tag>/part_*.npy      (shards, resumable)
         data/embeddings/pada_<tag>/embeddings.npy  (padas x 1024)
         data/embeddings/pada_<tag>/pada_meta.jsonl
         data/embeddings/pada_<tag>/model.json
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
PADAS_JSONL = PROCESSED / "padas.jsonl"

SHARD = 500        # padas per checkpoint file: small, so progress shows often
MAX_SEQ = 256      # padas: median 48 tokens, 99.9% within 256


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="model folder or Hugging Face id")
    ap.add_argument("--tag", required=True, help="output goes to data/embeddings/pada_<tag>/")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="encode only the first N padas (testing)")
    args = ap.parse_args()

    if not PADAS_JSONL.exists():
        sys.exit(f"Missing {PADAS_JSONL}. Run backend/pipeline/preprocessing/build_padas.py first.")
    padas = [json.loads(l) for l in PADAS_JSONL.open(encoding="utf-8")]
    if args.limit:
        padas = padas[:args.limit]
    texts = [p["text"] for p in padas]

    out_dir = EMBED_DIR / f"pada_{args.tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "pada_meta.jsonl").open("w", encoding="utf-8") as f:
        for p in padas:
            f.write(json.dumps({k: p[k] for k in ("pada_id", "shabad_id", "ang", "raag_id",
                                                  "split", "line_index", "text")},
                               ensure_ascii=False) + "\n")

    model_ref = args.model
    local = Path(args.model).resolve()
    if local.exists():
        try:
            model_ref = local.relative_to(PROJECT_ROOT).as_posix()
        except ValueError:
            model_ref = local.as_posix()
    (out_dir / "model.json").write_text(
        json.dumps({"model": model_ref, "max_seq": MAX_SEQ}, indent=2) + "\n", encoding="utf-8")

    shards = [(i, min(i + SHARD, len(texts))) for i in range(0, len(texts), SHARD)]
    todo = [(s, e) for s, e in shards if not (out_dir / f"part_{s:06d}.npy").exists()]
    print(f"padas: {len(texts):,}  ->  {out_dir}")
    print(f"{len(shards) - len(todo)}/{len(shards)} shards already done")

    if todo:
        from sentence_transformers import SentenceTransformer
        print(f"loading {args.model} ... (takes a minute)")
        model = SentenceTransformer(args.model)
        model.max_seq_length = MAX_SEQ

        started = time.time()
        done_padas = 0
        total_padas = sum(e - s for s, e in todo)
        for n, (s, e) in enumerate(todo, 1):
            vecs = model.encode(texts[s:e], batch_size=args.batch_size, normalize_embeddings=True,
                                convert_to_numpy=True, show_progress_bar=False).astype(np.float32)
            np.save(out_dir / f"part_{s:06d}.npy", vecs)
            done_padas += e - s
            elapsed = time.time() - started
            left = elapsed / done_padas * (total_padas - done_padas)
            print(f"  shard {n:>3}/{len(todo)}  padas {e:>6,}/{len(texts):,}  "
                  f"{100 * done_padas / total_padas:5.1f}%  elapsed {elapsed / 60:5.1f} min  "
                  f"left ~{left / 60:5.1f} min", flush=True)

    parts = [np.load(out_dir / f"part_{s:06d}.npy") for s, _ in shards]
    embeddings = np.vstack(parts)
    assert len(embeddings) == len(texts), "shard/pada count mismatch"
    np.save(out_dir / "embeddings.npy", embeddings)
    print(f"\nWrote {out_dir / 'embeddings.npy'}  shape={embeddings.shape}")
    print("The part_*.npy shards can be deleted now; they only exist to resume.")


if __name__ == "__main__":
    main()
