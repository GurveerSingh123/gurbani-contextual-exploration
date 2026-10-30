# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project purpose

B.Tech major project: **Semantic Retrieval and Contextual Exploration System** for Sri Guru Granth Sahib (Guru Nanak Dev Engineering College, Ludhiana).

Given a Gurbani **Pankati** (verse line), return other Pankatis that carry the same contextual meaning (**Bhavarth**) — *even when they share no vocabulary* — along with their Steeks, Ang, Raag and Shabad. The primary task is **verse-to-verse** retrieval, not natural-language-query-to-verse; results are deliberately drawn from *different* Shabads, since an adjacent line is something the reader can already see.

## Terminology

| Term | Meaning |
|---|---|
| **Pankati** | a single line of Gurbani (the retrieval unit) |
| **Steek** | a scholarly commentary/explanation of a Pankati |
| **Bhavarth** | contextual/spiritual meaning (what the system retrieves on) |
| **Ang** | page number in Sri Guru Granth Sahib (1–1430) |
| **Shabad** | a hymn; a group of consecutive Pankatis |
| **Raag** | the musical measure a Shabad is set in |
| `ss` / `bdb` / `ft` / `ms` / `pss` | the five Steek sources: Sahib Singh, BaniDB, Faridkot Teeka (Braj), Manmohan Singh, Padd-Arth glossary |

## The central design idea

There are **no verse-to-verse similarity labels** anywhere in the data. Supervision is bootstrapped from the commentaries:

> The semantic problem is hard on the *verses* and easy on the *commentaries*. Two Pankatis that mean the same thing are explained in nearly the same modern-Punjabi words, even when the verses share no vocabulary. So TF-IDF word overlap on Steeks ≈ semantic similarity on verses. The commentators already did the semantic work by hand.

This is used twice: to mine **training pairs** (train split) and to mine the **evaluation gold set** (test split).

## Commands

Setup — pick the file matching the machine; plain `requirements.txt` deliberately omits torch:

```bash
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements-cpu.txt     # laptop, no NVIDIA GPU
pip install -r requirements-gpu.txt     # training machine with CUDA
```

Pipeline, in order. Each step reads the previous step's output and writes new files:

```bash
python backend\pipeline\preprocessing\clean_dataset.py                        # 1. clean + profile
python backend\pipeline\training\create_training_pairs.py                     # 2. corpus + grouped split
python backend\pipeline\evaluation\build_eval_sets.py                         # 3. mine gold benchmark
python backend\pipeline\training\mine_verse_pairs.py                          # 4. mine Pankati<->Pankati positives
python backend\pipeline\training\mine_hard_negatives.py                       # 5. mine hard negatives
python backend\pipeline\retrieval\build_index.py --tag baseline               # 6. encode corpus
python backend\pipeline\evaluation\evaluate.py --bm25 --bm25-char --dense baseline
```

Training (**GPU only** — the script exits with a clear message on a CPU machine):

```bash
python backend\pipeline\training\train_bge.py --pairs verse_pairs_hn.jsonl --lora --batch-size 64 --epochs 5
python backend\pipeline\retrieval\build_index.py --model models\bge-m3-gurbani --tag verse_hn
python backend\pipeline\evaluation\evaluate.py --bm25-char --dense baseline --dense verse_hn
```

Always pass `--output models\<name>` when training; the default path is reused and silently overwrites the previous run.

Interactive search and side-by-side model comparison:

```bash
python backend\pipeline\retrieval\semantic_search.py --tag verse_hn --compare baseline
python backend\pipeline\retrieval\semantic_search.py --tag baseline --pankati "ਜਤ ਕਤ ਦੇਖਉ ਤਤ ਰਹਿਆ ਸਮਾਇ ॥"
```

Diagnostic — separates "training failed" from "training worked but didn't transfer":

```bash
python backend\pipeline\evaluation\diagnose_transfer.py --tag baseline
python backend\pipeline\evaluation\diagnose_transfer.py --tag finetuned --model models\bge-m3-gurbani
```

There is no test suite, linter or build step. Verification is done through the printed reports and the retrieval metrics.

## Architecture

Layout (reorganised 2026-10-28): `frontend/` (website), `backend/app/` (FastAPI server), `backend/pipeline/<stage>/` (the pipeline scripts, formerly `src/`), `common/` (code shared by both — `explain_match.py`), `data/`, `models/`, `docs/`, `tools/data_collection/`. Pipeline scripts locate the project root with `Path(__file__).resolve().parents[3]`; `common/explain_match.py` with `parents[1]` — update these if a script moves.

Independent stage scripts under `backend/pipeline/`, each reading files and writing new files, so any stage can be re-run without repeating the others. Nothing is a library; there are no shared modules besides `common/` (helpers are intentionally duplicated across mining scripts, which are run standalone — keep duplicates in step).

```
data/processed/final_gurbani_dataset_all_steeks.xlsx   (source, 60,360 rows x 13 cols, READ-ONLY)
        │
  clean_dataset.py ──────────> cleaned_gurbani.csv + profile_report.txt
        │                      flagged_structural_review.csv  (audit of exclusions)
        │
  create_training_pairs.py ──> corpus_pankatis.jsonl  (54,844 searchable verses + Steeks)
        │                      training_pairs.jsonl / val_pairs.jsonl / test_pairs.jsonl
        │                      row_splits.csv          (row -> split, so later stages don't re-derive)
        ├──> build_eval_sets.py ─────> eval_semantic_pairs.jsonl   (gold, from TEST split)
        ├──> mine_verse_pairs.py ────> verse_pairs.jsonl           (positives, from TRAIN split)
        │         └─> mine_hard_negatives.py ─> verse_pairs_hn.jsonl  (anchor/positive/negative)
        │
  train_bge.py (GPU) ────────> models/<name>/   LoRA adapter (~5 MB, not a full model)
        │
  build_index.py ────────────> data/embeddings/<tag>/embeddings.npy  (54,844 x 1024 float32)
        │                                        corpus_meta.jsonl
        ├──> semantic_search.py   (interactive retrieval)
        └──> evaluate.py          (scores any number of systems on the gold set)
```

A **tag** is a folder under `data/embeddings/` holding one `embeddings.npy` — the same corpus encoded by a *different* model. Tags are how before/after comparisons work; `build_index.py --tag X` never overwrites another tag.

Two non-obvious behaviours:

- **`build_index.py` sets `MAX_SEQ = 128`.** BGE-M3 defaults to 8192 tokens; Pankatis average ~34 characters, so leaving the default wastes enormous compute on padding. Do not remove this.
- **`semantic_search.py` accepts multiple Pankatis or a whole Shabad** as one query. It splits on `॥`, resolves each verse to its *cached* vector and averages them — which needs no model and avoids the 128-token limit a concatenated Shabad would exceed. Lookup is exact → punctuation-insensitive → partial, so a verse typed without its danda or verse numbers still matches.

## Models and libraries

- **BGE-M3** (`BAAI/bge-m3`) — base embedding model. XLM-RoBERTa-large, 568M params, 24 layers, **1024-dimensional** output, ~2.3 GB download. Chosen for multilingual (incl. Punjabi) retrieval.
- **LoRA** via `peft` — adapters on `query`/`key`/`value` only, r=16. Trains ~0.4% of parameters.
- **sentence-transformers 6.x** — `SentenceTransformerTrainer` + `MultipleNegativesRankingLoss`.
- **FAISS**, **rank_bm25** (word and char 3–5gram baselines), **scikit-learn** (TF-IDF mining), pandas/NumPy, python-dotenv.

Batch size *is* the training signal: `MultipleNegativesRankingLoss` uses the other batch items as negatives, so batch 64 gives 63 negatives and batch 8 gives 7. LoRA exists here largely to free memory for a bigger batch.

## Configuration

All paths and hyperparameters live in `.env` (loaded by `python-dotenv`); **no script hardcodes a path**. `.env` is gitignored; `.env.example` is the template.

`.env` does not currently define the hard-negative knobs — `mine_hard_negatives.py` falls back to its own defaults: `HN_LEXICAL_MIN=0.35`, `HN_LEXICAL_MAX=0.90`, `HN_MEANING_MAX=0.10`, `HN_MARGIN=0.25`, `HN_PER_PAIR=1`.

## Measured results (the project's central finding)

Training on **Pankati↔Steek** pairs raised that task hugely but did **nothing** for verse-to-verse. Training on **Pankati↔Pankati** pairs was the fix, and adding **hard negatives** made it the best system. Scored on the corrected **827-query** gold set (`data/processed/baseline_results.txt`):

| System (tag) | H@1 | H@10 | H@50 | MRR | H@10 on zero-overlap subset |
|---|---|---|---|---|---|
| BM25 (word) | 15.96 | 40.51 | 56.35 | 23.96 | 0.00% |
| BM25 (char 3–5gram) | 16.08 | 40.87 | **59.98** | 23.77 | 0.92% |
| BGE-M3 pretrained (`baseline`) | 14.03 | 29.38 | 42.20 | 19.39 | 1.84% |
| + Steek-pair tuning (`finetuned`) | 13.78 | 28.90 | 43.29 | 19.31 | 4.15% |
| + verse-pair tuning (`verse`) | 17.78 | 37.97 | 54.78 | 24.46 | 3.23% |
| **+ verse pairs + hard negatives (`verse_hn`)** | **18.02** | **41.23** | 58.77 | **25.61** | 5.53% |
| mixed (`mixed`) | 15.36 | 35.55 | 51.27 | 22.01 | 6.45% |

**`verse_hn` is the final model** (`models/bge-m3-verse-hn`). Its H@10 lead over char-BM25 is a statistical tie (~3 of 827 queries); its real advantage is at the top of the ranking (H@1, MRR) and on zero-overlap pairs. Say so in write-ups. Older 919-query numbers (pre-tokenizer-fix gold set, kept in `gpu_results_919q.txt`) are not comparable.

Diagnostic that revealed the cause — Steek→Pankati (the trained task) went 38.00% → 61.60% H@10 while Pankati→Pankati stayed flat. **Aligning a verse to its own commentary does not transfer to placing two similar verses together.** 6,289 verse pairs beat 131,806 Steek pairs because they were the *right* pairs.

Note: pretrained BGE-M3 **loses to keyword search**. That measurement is what justifies Gurbani-specific fine-tuning, and it should not be quietly dropped from write-ups.

## Gurmukhi / Unicode traps

Gurmukhi vowel signs (ਾ ਿ ੁ) are Unicode **combining marks**, which Python does not classify as word characters. This has bitten the project twice and will again:

1. **`re` word boundaries never fire after a matra.** `\b` is unusable; `clean_dataset.py` uses an explicit `(?=[\s॥।]|$)` terminator instead.
2. **scikit-learn's default `token_pattern` destroys Gurmukhi.** `\b\w\w+\b` keeps only runs of consecutive consonants: `"ਗਾਵੈ ਕੋ ਵਿਦਿਆ ਵਿਖਮੁ ਵੀਚਾਰੁ"` reduced to the single fragment `['ਖਮ']`, as did the unrelated `"ਕੋਊ ਬਿਖਮ ਗਾਰ ਤੋਰੈ"` — giving them cosine similarity 1.000. All three mining scripts now set `TOKEN_PATTERN = r"[^\s॥।,—()!?]+"`. **Never revert to the default.**
3. **pdfLaTeX cannot typeset Gurmukhi at all**; the report uses XeLaTeX + `fontspec`. The font must be declared `\newfontfamily\gur{Nirmala UI}[Script=Gurmukhi]` — without `Script=Gurmukhi` the shaper does not run and pre-base matras render after their consonant (`ਵਿਸਰੀ` appears as `ਵਸਿਰੀ`).

Comparing Gurmukhi strings for equality should go through consonant-skeleton normalisation (see `strip_marks` in the mining scripts) because spelling variants abound: `ਬਿਲਾਵਲੁ`/`ਬਿਲਾਵਲ`, `ਭੈਰਉ`/`ਭੈਰੳ`.

## Must not change without asking

- **`data/processed/final_gurbani_dataset_all_steeks.xlsx` is never written to.** Every stage writes new files.
- **`steek_bdb` excluded from training** — byte-identical to `steek_ss` in 99.9% of rows (54,941 of 55,017). Including both double-weights every Pankati that has SS.
- **`steek_pss` excluded from training** — a padd-arth glossary (`ਰੈਣਿ = ਰਾਤ।`, `=` in 49,928 of 50,015 rows), not a paraphrase. Retained for display and the planned LLM stage.
- **Training uses `ss` / `ft` / `ms` only.**
- **The train/val/test split is grouped, never random.** Union-find joins rows sharing a `shabad_id`, an identical Pankati, **or an identical Steek**. Without all three, near-identical verses straddle the split and any baseline-vs-fine-tuned comparison measures leakage rather than learning. Verified by asserting zero overlap on all four keys across all three split pairs.
- **Structural lines are flagged, not deleted.** The source `is_structural` flag catches only 720 of ~5,000 heading lines; `clean_dataset.py` adds prefix rules, a raag lexicon built from `data/metadata/raag_metadata.csv`, and a *scripture guard* (a line ≥30 chars that all three commentators translated is a teaching regardless of its first word — this is what rescues `ਰਾਗੁ ਨਾਦੁ ਸਭੁ ਸਚੁ ਹੈ...` from the raag-heading rule). Everything flagged is exported for audit.
- **Mining thresholds were tuned against the *corrected* tokenizer** (`GOLD_THRESHOLD=0.30`, `VERSE_PAIR_THRESHOLD=0.30`). The older 0.35 values belong to the broken tokenizer and are not comparable.
- **`train_bge.py` uses `model.add_adapter(cfg)`, not `peft.get_peft_model()`.** Wrapping externally produces an object sentence-transformers silently unwraps during checkpointing — training runs fine and the adapters never reach the saved model. The script verifies with a *behavioural* probe (embed a fixed verse before/after, then reload from disk and compare) rather than a weight check, because with `add_adapter` the base weights legitimately never change.

## Current constraints

- **The development laptop cannot train.** Intel i3-1005G1 (2 cores, 1.2 GHz), 11.8 GB RAM, Intel UHD graphics, no CUDA. Fine-tuning BGE-M3 needs ~9 GB for optimizer state alone. Training happens on a separate **RTX 3060 (12 GB)** desktop.
- Encoding the 54,844-verse corpus takes **112 minutes on the laptop CPU** versus ~3 minutes on the 3060. `build_index.py` checkpoints every 2,000 verses into `part_*.npy` shards and resumes, so an interrupted run is cheap. Once `embeddings.npy` exists the shards are redundant duplicates (28 shards of 8 MB, 225 MB in total).
- A query whose Pankati is already in the corpus is answered from the cached vector and **never loads the 2.3 GB model** — this is what makes interactive search viable on the laptop.
- `models/` and `data/embeddings/` are gitignored. **This is not a git repository** (no `.git`).

## Known issues

- **Only `bge-m3-verse-hn` and `bge-m3-mixed` exist under `models/`.** All five embedding tags are local, but `verse` and `finetuned` have no model folder, so free-text search on those tags is refused (verses already in the corpus still work). Each tag records the model that built it in `data/embeddings/<tag>/model.json`; `semantic_search.py` and the API load that model, never a hardcoded one.
- **Never run the reranker evaluation and the web server at the same time.** Together they exceed the laptop's 11.8 GB RAM; it has already killed one run.
- The mined gold set has **missing** relevant verses: TF-IDF only finds pairs whose commentaries share vocabulary, so measured recall is a **floor**, not the true value. It also contains some wrong pairs (e.g. `ਜਾ ਕਾ ਦਾਨੁ ਨਿਖੂਟੈ ਨਾਹੀ` paired with `ਪੂਰਨ ਹੋਤ ਨ ਕਾਰਜ ਮਾਇਆ` — both commentaries say "never end" about unrelated things). A small hand-labelled set would be the independent check.
- The gold set is derived from Steeks and the model trains on Steeks, so the benchmark is not fully independent of the training signal. BM25 is scored on the identical set, which keeps system-to-system comparison fair.
- Retrieval is still partly lexical. For a query containing a common phrase the model returns everything containing it and cannot reliably separate opposite teachings — e.g. `ਜੰਮਣੁ ਮਰਣਾ ਠਾਕਿ ਰਹਾਏ` ("he *stops* the cycle") still surfaces `ਜੰਮਣੁ ਮਰਣੁ ਨ ਚੁਕਈ` ("the cycle *never* ends"). Reducing this is what hard negatives target.
- `tools/data_collection/` (`collect_gurbani.py`, `adding_other_sources.py`, `collecting_raag.py`, `inspection.ipynb`, …) holds the original data-collection tools. They run against the **global** Python, not `.venv`, from the project root (cwd-relative paths), and are not part of the pipeline. The intermediate datasets they produced were moved out to `E:\Major_Project_Gurveer_archive\data_processed_intermediate\` (user decision 2026-10-28).

## Later stages

- **Reranker** (`backend/pipeline/evaluation/evaluate_rerank.py`, `BAAI/bge-reranker-v2-m3`): evaluated and **not adopted**. On 200 queries it made ranking worse on its own (H@10 40.5 → 36.5 on verses, 37.5 on Faridkot Steeks) and changed ≤3 queries when fused with `verse_hn`, at 6–37 s extra per search on this CPU. Modes `pankati` and `ft` are fair on the gold set; `ss` is **optimistic**, because the gold set was mined from SS and MS commentary similarity. The LLM explanation verdict is the second pass instead.
- **Explanation layer** (`common/explain_match.py`): an LLM explains a pair from its four Steeks only and returns a verdict (same / related / different). **Free only — the user will not pay for an API.** It talks to any OpenAI-compatible endpoint set by `LLM_BASE_URL` / `LLM_MODEL` / `LLM_API_KEY` in `.env`; default is Groq's free tier with `openai/gpt-oss-120b` (~30 requests/min, ~1,000/day, 8K tokens/min). Gemini's free tier is the fallback if Punjabi output is weak. Replies are validated and retried once; answers are cached in `data/processed/explanations_cache.jsonl`.
- **Web interface** — two folders: `backend/` (FastAPI: `app/config.py`, `app/engine.py`, `app/main.py`) and `frontend/` (HTML + Tailwind CSS 3.4.17 + vanilla JS modules, dark glowing theme; `assets/css/app.css` is compiled — rebuild with `npm run build:css` in `frontend/`). Run from the project root: `python -m uvicorn backend.app.main:app --port 8000` (backend also serves the frontend; ready in ~5 s). It **searches PADAS**, not single lines (user decision 2026-10-28). A typed line is mapped to its pada; several lines/a Shabad average their padas; free text goes through the pada model. Main results = char-BM25 + pada model (`PADA_TAG=pada` → `models/bge-m3-pada`) merged by RRF (k=60), one pada per Shabad; each card highlights the line closest to the query line (line vectors from `SEARCH_TAG`). A second section, **"Same meaning, different words"** (`DIFF_K`), lists the pada model's best matches sharing **no word by consonant skeleton** with the query — fusion buries these, and they are the project's distinctive result. "Why this match?" explains pada vs pada with SS/MS/FT only (no glossary, to stay under Groq's 8K tokens/min). The API imports `explain_match` rather than copying it — the one deliberate exception to "no shared modules". The line-level CLI (`semantic_search.py`) is unchanged.
- **Ask a question** (`backend/app/ask.py`, `/api/ask`, second tab on the page with its **own input box** — a question typed into the Gurbani box is searched as free text and loads the 2.3 GB model on CPU, very slow): an English or Punjabi question → the LLM lists Punjabi/Gurbani terms → word-BM25 over each pada's **Steeks** (SS+MS+text; a question shares no words with verses) → `gpt-oss-20b` picks 2–3 of 15 candidates → `gpt-oss-120b` (reasoning *medium*) explains each **whole Shabad** (read with its Rahao), gives a direct answer and, when the question asks for help, "What you can do" steps — all in the **question's language** (Gurmukhi in the question → Punjabi, else English). Guardrails in the prompt forbid practices Gurbani rejects and correct idioms (ਆਠ ਪਹਰ = at all times, ਨਿੰਦਾ = slandering others) — all were real errors. LLM calls use `max_retries=0` and a 60 s timeout: the SDK's silent rate-limit retries made the page hang; now a 429 waits ≤12 s once, else tells the reader. The model may return the Ang as the Shabad id; `_ids_by_position` maps it back. Cached in `data/processed/ask_cache.jsonl` keyed by model + `PROMPT_VERSION`; bump it when prompts change. gpt-oss Punjabi is understandable but imperfect; `ASK_LLM_*` in `.env` can point steps 1 and 4 at Gemini's free tier.

## Deliverables

`mid term report/` holds the college submission: `Midterm_report.tex` (**compile with XeLaTeX**, not pdfLaTeX), its PNG figures, and `references.bib`. `docs/project-documentation.html` is a plain-language project write-up.

### How the report figures are produced

Every figure in the report is a **PNG**, not inline LaTeX — a deliberate choice, and necessary for anything containing Gurmukhi.

| Figure group | Produced by |
|---|---|
| `fig_diag_*.png` — the 11 design diagrams | TikZ in a `standalone` document → XeLaTeX → PDF → rasterised at 600 DPI with **pymupdf**. The generator (`diagrams.py`) was **lost** with a temp folder; the PNGs are the only copy. |
| `fig_dataset_*`, `fig_code_*`, `fig_*_report*`, `fig_output_*` | HTML → **headless Chrome** `--screenshot --force-device-scale-factor=2` → auto-cropped with **Pillow**; code highlighting via **pygments** |

Two things to know before regenerating any of them:

- **`pygments`, `pillow` and `pymupdf` are installed in `.venv` but are absent from all three `requirements*.txt` files.** Recreating the venv from requirements will silently break figure generation. Install them explicitly.
- **Headless Edge does not work** for screenshots on this machine (produces no file regardless of flags). Use `C:\Program Files (x86)\Google\Chrome\Application\chrome.exe`, and pass paths without spaces — the report folder name contains one.

Chrome and XeLaTeX both shape Gurmukhi correctly. Pillow is used only for cropping; neither Pillow nor .NET `System.Drawing` does complex-script shaping reliably, so never render Gurmukhi text through them.
