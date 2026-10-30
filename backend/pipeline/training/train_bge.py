"""
Fine-tune BGE-M3 on Gurbani Pankati <-> Steek pairs.

Run this on the RTX 3060 (12 GB). The 3050's 4 GB cannot hold the optimizer
state for a 568M-parameter model.

    # prototype: ~20 min, enough to show movement
    python backend/pipeline/training/train_bge.py --quick

    # full run
    python backend/pipeline/training/train_bge.py --lora --batch-size 64

    # pada model: longer texts, pada validation set, loss computed in chunks of 8
    # (batch 32 x 256 tokens ran out of memory on a 15 GB T4 without --mini-batch)
    python backend/pipeline/training/train_bge.py --pairs pada_pairs.jsonl --val pada_val_pairs.jsonl \
        --max-seq 256 --mini-batch 8 --lora --batch-size 32 --epochs 5 --output models/bge-m3-pada

Objective: MultipleNegativesRankingLoss. Every other item in the batch acts as
a negative, so BATCH SIZE IS THE TRAINING SIGNAL -- 64 gives each anchor 63
negatives, 8 gives it 7. This is why LoRA is preferred here: it frees the
memory that would otherwise go to optimizer state and buys a bigger batch.

Memory on 12 GB, seq len 128:
    --lora                      batch 64+   (recommended)
    --freeze-layers 12          batch 16
    full fine-tune              batch 8, and it will be tight

Reads  : data/processed/training_pairs.jsonl, val_pairs.jsonl
Writes : models/bge-m3-gurbani/
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(PROJECT_ROOT / ".env")

PROCESSED = PROJECT_ROOT / os.getenv("PROCESSED_DATA_DIR", "data/processed")
MODEL_DIR = PROJECT_ROOT / os.getenv("MODEL_DIR", "models")
BASE_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
SEED = int(os.getenv("SEED", "42"))
# 128 suits single lines (95% within 30 tokens). Padas need --max-seq 256:
# median 48 tokens, 95% within 127, 99.9% within 256.
MAX_SEQ = 128


def load_pairs(paths, limit=None):
    """JSONL file(s) -> {anchor, positive}.

    Accepts a single path or several. Steek->Pankati and Pankati->Pankati pairs
    have the same shape, so they can be mixed in one training set.
    """
    if isinstance(paths, Path):
        paths = [paths]
    anchors, positives, negatives = [], [], []
    for path in paths:
        if not path.exists():
            sys.exit(f"Missing {path}. Run create_training_pairs.py "
                     f"or mine_verse_pairs.py first.")
        n_before = len(anchors)
        with path.open(encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                anchors.append(rec["query"])
                positives.append(rec["pos"][0])
                neg = rec.get("neg") or []
                negatives.append(neg[0] if neg else None)
        print(f"  {path.name}: {len(anchors) - n_before:,} pairs")
    if limit and limit < len(anchors):
        idx = random.Random(SEED).sample(range(len(anchors)), limit)
        anchors = [anchors[i] for i in idx]
        positives = [positives[i] for i in idx]
        negatives = [negatives[i] for i in idx]

    data = {"anchor": anchors, "positive": positives}
    # MultipleNegativesRankingLoss takes an optional third column as an
    # explicit hard negative. The column has to be complete, so it is only
    # added when every record has one -- mixing would silently drop it.
    if negatives and all(n for n in negatives):
        data["negative"] = negatives
        print(f"  -> using explicit hard negatives ({len(negatives):,})")
    elif any(negatives):
        missing = sum(1 for n in negatives if not n)
        print(f"  -> ignoring hard negatives: {missing:,} of {len(negatives):,} "
              f"records lack one (the column must be complete)")
    return data


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default=BASE_MODEL)
    ap.add_argument("--output", default=str(MODEL_DIR / "bge-m3-gurbani"))
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--epochs", type=float, default=1.0)
    # Default depends on the method, resolved below: LoRA adapters start from
    # random values and need a far larger step than full fine-tuning, which is
    # nudging weights that are already good.
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--max-pairs", type=int, default=None)
    ap.add_argument("--pairs", default="training_pairs.jsonl",
                    help="comma-separated files in data/processed/. "
                         "e.g. verse_pairs.jsonl  or  training_pairs.jsonl,verse_pairs.jsonl")
    ap.add_argument("--val", default="val_pairs.jsonl",
                    help="validation file in data/processed/ -- use pada_val_pairs.jsonl "
                         "when training on padas, or the loss shown is for the wrong task")
    ap.add_argument("--max-seq", type=int, default=MAX_SEQ,
                    help="token limit per text: 128 for lines, 256 for padas")
    ap.add_argument("--mini-batch", type=int, default=0,
                    help="compute the loss in chunks of this size (e.g. 8). Keeps the full "
                         "--batch-size worth of in-batch negatives at a fraction of the memory; "
                         "needed for padas, where batch 32 x 256 tokens overflows a 15 GB T4")
    ap.add_argument("--lora", action="store_true",
                    help="LoRA adapters -- least memory, biggest batch")
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--freeze-layers", type=int, default=0,
                    help="freeze the bottom N transformer layers")
    ap.add_argument("--no-fp16", action="store_true")
    ap.add_argument("--no-eval", action="store_true",
                    help="skip validation scoring (use if the trainer objects to "
                         "train having a hard-negative column that val lacks)")
    ap.add_argument("--quick", action="store_true",
                    help="prototype: 20k pairs, LoRA, ~20 min on a 3060")
    args = ap.parse_args()

    if args.quick:
        args.max_pairs = args.max_pairs or 20_000
        args.lora = True
        args.batch_size = max(args.batch_size, 32)

    if args.lr is None:
        args.lr = 1e-4 if args.lora else 2e-5
    print(f"method: {'LoRA' if args.lora else 'full fine-tune'}   lr={args.lr}   "
          f"batch={args.batch_size}")

    # Check the GPU before the heavy imports, so running this on the laptop
    # says what is actually wrong instead of "No module named datasets".
    import torch

    if not torch.cuda.is_available():
        sys.exit("No CUDA device found. Run this on the RTX 3060 (12 GB), "
                 "not the laptop -- see requirements-gpu.txt.")
    gpu = torch.cuda.get_device_properties(0)
    print(f"GPU: {gpu.name}  {gpu.total_memory / 1e9:.1f} GB")
    if gpu.total_memory < 6e9:
        print("WARNING: under 6 GB. Use --lora --batch-size 8, and expect OOM.")

    from datasets import Dataset
    from sentence_transformers import (
        SentenceTransformer, SentenceTransformerTrainer,
        SentenceTransformerTrainingArguments,
    )
    try:                                    # moved in sentence-transformers 6.x
        from sentence_transformers.sentence_transformer import losses
    except ImportError:
        from sentence_transformers import losses

    import transformers
    global _TRANSFORMERS_V5
    _TRANSFORMERS_V5 = int(transformers.__version__.split(".")[0]) >= 5

    random.seed(SEED)
    torch.manual_seed(SEED)

    pair_files = [PROCESSED / n.strip() for n in args.pairs.split(",") if n.strip()]
    print("training pairs:")
    train = load_pairs(pair_files, args.max_pairs)
    val = load_pairs(PROCESSED / args.val, 2000)
    print(f"train pairs: {len(train['anchor']):,}   val pairs: {len(val['anchor']):,}")

    print(f"loading {args.base_model} ...")
    model = SentenceTransformer(args.base_model)
    model.max_seq_length = args.max_seq
    print(f"max sequence length: {args.max_seq} tokens")

    transformer = model[0].auto_model

    if args.lora:
        try:
            from peft import LoraConfig
        except ImportError:
            sys.exit("--lora needs peft:  pip install peft")
        # Attention projections only. "dense" would match 73 modules in
        # XLM-RoBERTa (attention output, both FFN layers, and the unused
        # pooler), which costs memory we would rather spend on batch size.
        cfg = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_r * 2,
            lora_dropout=0.05,
            bias="none",
            target_modules=["query", "key", "value"],
        )
        # Use SentenceTransformer's own PEFT integration, NOT get_peft_model().
        # add_adapter injects adapters into the existing model, so saving and
        # loading stay adapter-aware. Wrapping with get_peft_model() instead
        # produces an object the trainer silently unwraps during checkpointing,
        # and the trained adapters never reach the saved model.
        model.add_adapter(cfg)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"LoRA: {trainable:,} trainable of {total:,} ({100 * trainable / total:.2f}%)")
    elif args.freeze_layers:
        for p in transformer.embeddings.parameters():
            p.requires_grad = False
        for layer in transformer.encoder.layer[:args.freeze_layers]:
            for p in layer.parameters():
                p.requires_grad = False
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"froze bottom {args.freeze_layers} layers -> {trainable:,} trainable")
        # Gradient checkpointing needs at least one input requiring grad; with
        # frozen embeddings there is none, and the backward pass silently
        # produces nothing. This is the standard fix.
        if hasattr(transformer, "enable_input_require_grads"):
            transformer.enable_input_require_grads()

    # Behavioural check, not a weight check: with LoRA the base weights never
    # change (the adapter sits on top), so comparing them would fail on every
    # correct run. What must change is the model's OUTPUT.
    PROBE_TEXT = "ਜਤ ਕਤ ਦੇਖਉ ਤਤ ਰਹਿਆ ਸਮਾਇ ॥"
    probe_before = model.encode(PROBE_TEXT, normalize_embeddings=True)

    if args.mini_batch:
        # Same objective and scale as MultipleNegativesRankingLoss, computed with
        # gradient caching: embeddings in chunks, so memory follows --mini-batch
        # while every anchor still sees all --batch-size candidates as negatives.
        loss = losses.CachedMultipleNegativesRankingLoss(model, mini_batch_size=args.mini_batch)
        print(f"loss: cached MNRL, batch {args.batch_size} computed in chunks of {args.mini_batch}")
    else:
        loss = losses.MultipleNegativesRankingLoss(model)

    out = Path(args.output)
    targs = SentenceTransformerTrainingArguments(
        output_dir=str(out / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        # Transformers v5 wants the ratio passed as a float `warmup_steps`.
        **({"warmup_steps": 0.1} if _TRANSFORMERS_V5 else {"warmup_ratio": 0.1}),
        fp16=not args.no_fp16,
        gradient_checkpointing=not args.lora,   # LoRA rarely needs it
        logging_steps=50,
        eval_strategy="no" if args.no_eval else "steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=2,
        seed=SEED,
        # Spawned workers re-import sentence-transformers on every epoch and
        # every evaluation, which costs more than it saves unless they persist.
        dataloader_num_workers=2,
        dataloader_persistent_workers=True,
        report_to=[],
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=targs,
        train_dataset=Dataset.from_dict(train),
        eval_dataset=None if args.no_eval else Dataset.from_dict(val),
        loss=loss,
    )
    trainer.train()

    # A run that trains happily and then saves an untouched model is the worst
    # failure mode here: everything looks fine until the scores do not move.
    probe_after = model.encode(PROBE_TEXT, normalize_embeddings=True)
    drift = 1.0 - float(probe_before @ probe_after)
    print(f"probe drift: {drift:.6f}  (0 = model output unchanged)")
    if drift < 1e-6:
        sys.exit("ABORTED: the model produces identical embeddings to before "
                 "training. The fine-tuning did not take effect.")

    out.mkdir(parents=True, exist_ok=True)
    model.save(str(out))
    print(f"\nSaved to {out}")

    # Round-trip check: reload from disk and confirm it still behaves as trained.
    # With LoRA, `out` holds only adapter_config.json + adapter_model.safetensors
    # and references BAAI/bge-m3 as the base, so this also proves the reference
    # resolves.
    from sentence_transformers import SentenceTransformer as _ST
    reloaded = _ST(str(out))
    reloaded.max_seq_length = args.max_seq
    agreement = float(probe_after @ reloaded.encode(PROBE_TEXT, normalize_embeddings=True))
    print(f"reload check: cosine {agreement:.6f}  (1.0 = saved correctly)")
    if agreement < 0.999:
        sys.exit("ABORTED: the reloaded model does not match the trained one.")
    tag = out.name.removeprefix("bge-m3-").replace("-", "_")
    print("Next, on this same machine:")
    if "pada" in args.pairs:
        print(f"  python backend/pipeline/retrieval/build_pada_index.py --model {out} --tag {tag}")
        print(f"  python backend/pipeline/evaluation/evaluate_padas.py --linemean verse_hn --dense verse_hn --dense {tag}")
    else:
        print(f"  python backend/pipeline/retrieval/build_index.py --model {out} --tag {tag}")
        print(f"  python backend/pipeline/evaluation/evaluate.py --bm25-char --dense baseline --dense {tag}")


if __name__ == "__main__":
    main()
