# PROJECT_STATE.md

Handoff document between Claude Code sessions. Read `CLAUDE.md` first for stable
project context (architecture, terminology, design rules). This file records
**where the work currently stands**.

Last updated: 2026-10-22

---

## 1. Objective

Given a Gurbani **Pankati** (verse line), retrieve other Pankatis carrying the
same contextual meaning (**Bhavarth**) — even when they share no vocabulary —
with their Steeks, Ang, Raag and Shabad. Verse-to-verse retrieval is the primary
task; results are drawn from *different* Shabads.

Two deliverables run in parallel:

- **the system** (`backend/pipeline/`) — data pipeline, fine-tuned embedding model, retrieval
- **the midterm report** (`mid term report/`) — college submission, XeLaTeX

---

## 2. Current architecture

Seven stage scripts, each reading files and writing new files; no shared library.

```
final_gurbani_dataset_all_steeks.xlsx  (READ-ONLY source, 60,360 rows)
  → clean_dataset.py          → cleaned_gurbani.csv (+ profile & audit reports)
  → create_training_pairs.py  → corpus_pankatis.jsonl (54,844 verses)
                                training/val/test_pairs.jsonl, row_splits.csv
  → build_eval_sets.py        → eval_semantic_pairs.jsonl   (gold, TEST split)
  → mine_verse_pairs.py       → verse_pairs.jsonl            (positives, TRAIN split)
  → mine_hard_negatives.py    → verse_pairs_hn.jsonl         (anchor/pos/neg triplets)
  → train_bge.py   [GPU only] → models/<name>/  LoRA adapter (~5 MB)
  → build_index.py            → data/embeddings/<tag>/embeddings.npy (54,844 × 1024)
  → evaluate.py / semantic_search.py / diagnose_transfer.py
```

Supervision is bootstrapped from the commentaries: two Pankatis are a pair when
**two independent commentators** (Sahib Singh, Manmohan Singh) explain them
alike. The same rule builds training pairs (train split) and the gold benchmark
(test split).

---

## 3. Implementation status

| Stage | Status |
|---|---|
| Data cleaning & profiling | ✅ complete |
| Corpus + leakage-free split | ✅ complete, zero-leakage verified |
| Gold benchmark mining | ✅ complete (827 queries) |
| Verse-pair + hard-negative mining | ✅ complete (5,717 / 5,343) |
| Baseline measurement | ✅ complete |
| LoRA fine-tuning | ✅ complete; **final model = `verse_hn`** (`models/bge-m3-verse-hn`, verified) |
| Re-evaluation on corrected 827-query gold set | ✅ done (2026-10-22) |
| Reranker stage | ✅ evaluated; **not adopted** — no gain over verse_hn, slow on CPU (§6) |
| LLM explanation layer | ✅ built and **tested live on Groq** (2026-10-24): envy pair → same; "cycle stopped" vs "never ends" → different; vs "cycle ends" → same |
| Web / FastAPI interface | ✅ rebuilt 2026-10-28 as `backend/` (FastAPI) + `frontend/` (Tailwind, dark glowing theme); pada search; tested |
| Midterm report + slides | ⏸ on hold until the user says; numbers in them are the OLD 919-query run |

---

## 4. Completed

**Data preparation**
- 60,360 rows → 5,077 excluded as structural/heading → **55,217 trainable**
- Retrieval corpus: **54,844** unique Pankatis, deduplicated, Steeks attached
- Split: 131,806 / 16,457 / 16,515 pairs at exactly 80/10/10
- Grouped split via union-find on `shabad_id` + identical Pankati + identical
  Steek. **All 12 cross-split overlap checks return 0.**

**Benchmark**
- 827 queries, 1,758 gold pairs, 18.9% with zero shared vocabulary
- Cross-Shabad by construction; two-commentator agreement required

**Training data**
- 5,717 verse-to-verse positives (19.1% zero-overlap)
- 5,343 of those with a mined hard negative (93.5% coverage)

**Measurement** — the project's central result (full detail in `CLAUDE.md`):
pretrained BGE-M3 *loses* to keyword search (19.37 vs 27.09 H@10); Steek-pair
training changed nothing on the target task; verse-pair training won on every
metric (28.29 H@10, 3.57% on the zero-overlap subset).

**Report** — 36 pages, compiles with **0 errors** under XeLaTeX. All 11 design
diagrams are now PNGs (600 DPI) rather than inline TikZ.

---

## 5. Currently being worked on

Building out the system past the embedding model (user: "proceed with the
project, leave the report and slide part until I say").

### 5a. Results on the corrected 827-query gold set (authoritative)

`data/processed/baseline_results.txt` (old 919-query run kept as
`gpu_results_919q.txt` — not comparable):

| System | H@1 | H@5 | H@10 | H@50 | MRR | zero-overlap H@10 (217 q) |
|---|---|---|---|---|---|---|
| BM25 word | 15.96 | 32.89 | 40.51 | 56.35 | 23.96 | 0.00 |
| BM25 char 3–5 | 16.08 | 31.80 | 40.87 | **59.98** | 23.77 | 0.92 |
| pretrained | 14.03 | 25.15 | 29.38 | 42.20 | 19.39 | 1.84 |
| finetuned (Steek pairs) | 13.78 | 25.39 | 28.90 | 43.29 | 19.31 | 4.15 |
| verse | 17.78 | 32.16 | 37.97 | 54.78 | 24.46 | 3.23 |
| **verse_hn** | **18.02** | **33.74** | **41.23** | 58.77 | **25.61** | 5.53 |
| mixed | 15.36 | 28.90 | 35.55 | 51.27 | 22.01 | **6.45** |

How to state it honestly: verse_hn vs char-BM25 at H@10 is a **statistical
tie** (≈3 queries of 827; ±1.7 pt noise). The real gains are at the top of the
ranking (H@1, MRR) and on zero-overlap pairs (6× BM25). mixed's zero-overlap
lead over verse_hn is 14 vs 12 queries — noise. **The narrative changed from
the old run: hard negatives now help, and plain `verse` falls below BM25.**

### 5b. Models and embeddings now local

`data/embeddings/{baseline,finetuned,verse,verse_hn,mixed}` (all verified:
identical corpus_meta, unit-norm, finite). `models/bge-m3-verse-hn` and
`models/bge-m3-mixed` reproduce their stored vectors at cosine 1.000. No model
folder exists for `verse` or `finetuned` — not needed (verse_hn wins).
`mixed` = 40,000 anchor/positive pairs, 3 epochs, batch 64 (training mix not
recorded). Each tag folder now has `model.json` naming the model that built it.

---

## 6. Exact current problem

### System (active)

1. **Reranker: evaluated, NOT adopted (2026-10-26).** 200 queries, top 20
   reranked (`data/processed/rerank_results_fair.txt`):

   | System | H@1 | H@10 | MRR |
   |---|---|---|---|
   | verse_hn alone | 15.00 | 40.50 | 23.57 |
   | rerank, verses | 15.50 | 36.50 | 21.84 |
   | rerank+fuse, verses | 15.50 | 40.50 | 23.17 |
   | rerank, Faridkot Steek | 10.50 | 37.50 | 19.23 |
   | rerank+fuse, Faridkot | 16.50 | 42.00 | 23.69 |

   The reranker alone makes ranking **worse** (it has never seen Gurbani). Fused
   with verse_hn it changes ≤3 queries of 200 — noise — at ~6–15 s extra per
   search on this CPU. Decision: keep search dense-only; use the LLM verdict as
   the second pass instead. `ss` mode not run (optimistic by construction, and
   moot given the above). Out-of-memory lesson still stands: never run heavy
   jobs and the web server together.
2. **Explanations work** — Groq key is in `.env` (tested 2026-10-24, ~5–15 s per
   new pair, cached after). Minor accuracy slips seen: glossed over a Faridkot
   Steek that disagreed; one key word respelled. The web "Why this match?"
   button has not yet been tried live.
3. Visible retrieval failure to keep in mind: `ਜੰਮਣੁ ਮਰਣਾ ਠਾਕਿ ਰਹਾਏ` ("cycle
   stopped") returns "cycle continues" verses at ranks 1, 2, 9. The explanation
   verdict (`different`) and/or reranker are meant to catch this.

### Report (on hold — do not touch until the user says)

**`diagrams.py` is LOST.** The old session scratchpad holding it was cleaned up
between 2026-10-05 and 2026-10-22. The 11 `fig_diag_*.png` files are intact, but
changing any diagram now means rewriting its TikZ from scratch.

The report and slides still carry the **old 919-query** numbers and the claim
that `verse` (not `verse_hn`) is the winner — both must be updated when the user
resumes report work (see §5a).

### (a) DFD level-1 `results` label — resolved, one judgement call left

The DFD level-1 diagram originally had a badly placed `results` flow label: at
`pos=0.9` with `left` anchoring on the leftward return segment, which pushed it
outside the drawing so it looked clipped and detached. The user asked about it.

**Resolved.** The label was removed from the generator and the PNG re-rendered
from that source — verified by inspecting `fig_diag_dfd1.png` directly, which
now shows the return arrow with no label. Source and PNG agree.

(An earlier note in this file claimed they were out of sync, inferred from
timestamps. That was wrong — confirmed by looking at the image.)

Remaining judgement call only: **DFD convention labels every data flow**, so the
examiner may expect `results` on the 6.0 → User arrow. If it should come back,
place it on the short horizontal segment so it stays inside the drawing:

```latex
\draw[flow] (p6.west) -- node[lbl, above, pos=0.55]{results} ++(-2.4,0) |- (user.west);
```

then re-render. Ask the user; do not decide unilaterally.

### (b) Table of Contents page numbers are each 1 too high

The user has updated the ToC, but a fresh compile disagrees by exactly +1 on
every entry:

| Section | in ToC | actual |
|---|---|---|
| System Requirements | 5 | **4** |
| Software Requirement Analysis | 7 | **6** |
| Software Design | 10 | **9** |
| Testing Module | 20 | **19** |
| Performance of the Project Developed | 23 | **22** |
| Output Screens | 32 | **31** |

Introduction (1) is correct. The offset is uniform (+1), which suggests the
values were taken from a build one page longer. The List of Figures was not
re-verified and probably carries the same offset. References does not appear in
the `.aux` section list (it is added via `\addcontentsline`) — read its page
from the compiled PDF.

Document is currently **36 pages**, compiling with **0 errors**.

**Do not fix this unsolicited** — the user is mid-edit and every edit shifts
these. Offer, don't apply. To regenerate the true values:

```bash
cd "mid term report"; xelatex Midterm_report.tex; xelatex Midterm_report.tex
Select-String -Path Midterm_report.aux -Pattern "contentsline \{section\}"
```

### (c) Three figure-toolchain packages are missing from `requirements*.txt`

`pygments`, `pillow` and `pymupdf` are installed in `.venv` but appear in none
of the three requirements files. They are needed to regenerate the report
figures (syntax highlighting, PNG cropping, PDF→PNG rasterisation). **Recreating
the venv from `requirements-cpu.txt` would silently break figure generation.**
Not fixed — the user did not ask for requirements changes. Worth offering.

---

## 7. Files recently modified

| File | Change |
|---|---|
| `CLAUDE.md` | **created** this session |
| `PROJECT_STATE.md` | **created** this session |
| `mid term report/Midterm_report.tex` | 11 TikZ blocks → `\includegraphics`; 27 missing `\begin{figure}[H]` restored; `[Script=Gurmukhi]` added to the font declaration. User has since filled the title page and edited the ToC. |
| `mid term report/fig_diag_*.png` | all 11 diagrams redesigned and re-rendered |
| `mid term report/fig_output_*.png`, `fig_code_*.png`, `fig_dataset_*.png`, `fig_*_report.png` | 17 figures generated |
| `backend/pipeline/evaluation/build_eval_sets.py` | Gurmukhi `TOKEN_PATTERN` fix; speaker-formula stripping |
| `backend/pipeline/training/mine_verse_pairs.py` | **created**; tokenizer fix |
| `backend/pipeline/training/mine_hard_negatives.py` | **created**; `LEX_MAX` upper bound added |
| `backend/pipeline/training/train_bge.py` | `add_adapter` instead of `get_peft_model`; LoRA learning rate; behavioural probe + reload check; `--pairs`, `--no-eval` |
| `backend/pipeline/evaluation/diagnose_transfer.py` | **created** |
| `.env` | corrected thresholds (0.30), verse-pair and gold-set knobs |
| `mid term report/Midterm_presentation.pptx` + `build_presentation.js` | 15-slide panel deck (pptxgenjs); objectives verbatim from report. Old 919-query numbers. |
| *2026-10-22 session* | |
| `backend/pipeline/retrieval/build_index.py` | writes `data/embeddings/<tag>/model.json` (which model built the tag) |
| `backend/pipeline/retrieval/semantic_search.py` | free text now uses the tag's own model (was hardcoded `models/bge-m3-gurbani` for every tag); model cached per session; default tag from `SEARCH_TAG`; no fake `(+0)` Ang gap |
| `backend/pipeline/evaluation/evaluate_rerank.py` | **created** — cross-encoder reranking in `pankati` / `ft` / `ss` modes + RRF fusion; report rewritten after each mode |
| `common/explain_match.py` | **created** — Claude explanation grounded in all four Steeks, JSON schema output, verdict same/related/different, cached in `data/processed/explanations_cache.jsonl` |
| `src/api/app.py`, `src/web/index.html` | **created** — FastAPI backend + single-page UI (`python -m uvicorn src.api.app:app --port 8000`) |
| `.env`, `.env.example` | `SEARCH_TAG=verse_hn`, `LLM_BASE_URL` (Groq), `LLM_MODEL=openai/gpt-oss-120b` — free providers only, user will not pay |
| `requirements.txt` | `anthropic>=2.1`, `fastapi`, `uvicorn[standard]` |
| `data/embeddings/*/model.json` | baseline → `BAAI/bge-m3`, verse_hn → `models/bge-m3-verse-hn`, mixed → `models/bge-m3-mixed` |
| `.claude/launch.json` | `gurbani-web` preview config |

---

## 8. Important decisions made

1. **Train on Pankati↔Pankati pairs, not Pankati↔Steek.** The original design
   aligned each verse with its own commentary and hoped verse-verse similarity
   would emerge. Measurement showed it does not. This was the single most
   important change in the project.
2. **Mining thresholds re-tuned to 0.30** after the tokenizer fix. The earlier
   0.35 values were calibrated against broken tokenization and are not
   comparable.
3. **Hard negatives need an upper bound as well as a lower one.**
   `HN_LEXICAL_MAX=0.90` stops the miner selecting a verse's own repeat in
   another Shabad as its negative.
4. **Verification must be behavioural, not weight-based.** With
   `model.add_adapter`, base weights legitimately never change, so a weight
   fingerprint aborts every correct run. `train_bge.py` embeds a fixed probe
   verse before/after training, then reloads from disk and compares.
5. **Report diagrams as PNG, not inline TikZ** — the user's explicit preference,
   and it decouples diagram iteration from report compilation.
6. **Report tables are floats with captions on top; figures keep captions
   below.** Converting tables to `table[H]` was required because a top caption
   in a bare `center` block gets orphaned from its table across a page break.
7. **XeLaTeX, not pdfLaTeX** — pdfLaTeX cannot typeset Gurmukhi at all.

---

## 9. Experiments performed

| # | Experiment | Result |
|---|---|---|
| 1 | BM25 word-level baseline | H@10 23.18, **0.00%** on zero-overlap (definitional, not evidence) |
| 2 | BM25 char 3–5gram baseline | H@10 **27.09** — strongest lexical baseline |
| 3 | Pretrained BGE-M3 | H@10 19.37 — **loses to keyword search** |
| 4 | LoRA on 20k Steek pairs, lr 2e-5 | eval loss 0.475; H@10 18.50 — **no change** |
| 5 | Same, lr 1e-4 | eval loss **0.301** — LoRA needs the higher rate |
| 6 | Transfer diagnostic | Steek→Pankati **38.0 → 61.6%** while Pankati→Pankati stayed flat — training worked, skill didn't transfer |
| 7 | LoRA on 6,289 verse pairs, 5 epochs | H@10 **28.29**, MRR 16.77, zero-overlap **3.57%** — beats every baseline |
| 8 | Two-commentator agreement filter | Rejected the `ਕਬੀਰ ਆਖਦਾ ਹੈ` formulaic class of false pairs |
| 9 | Speaker-formula stripping | Dropped a known-bad pair 0.525 → 0.000 **and raised** coverage 16% → 18% |
| 10 | Gold threshold sweep (0.25/0.30/0.35/0.40/0.45) | 0.30 chosen: 827 queries with visibly cleaner pairs |
| 11 | Hard-negative availability probe | 295 of 300 sampled verses have lexical lookalikes with different meaning |
| 12 | Whole-Shabad query (averaged vectors) | Better than any single verse — averaging cancels incidental wording |
| 13 | Intra-Shabad cohesion | cosine 0.545 within a Shabad vs 0.511 random — pretrained model barely distinguishes them |

| 14 | Hard-negative training (`verse_hn`), measured on GPU, 919 q | H@10 26.66 vs verse 28.29; zero-overlap 4.05% vs 3.57% |
| 15 | **All tags re-scored on the corrected 827-query gold set** | verse_hn best on H@1/5/10, MRR, nDCG — see §5a. Rows 1–7 above are the OLD 919-query numbers |
| 16 | Model ↔ embedding consistency check | verse-hn and mixed folders reproduce their stored vectors at cosine 1.000 |
| 17 | Reranker timing (bge-reranker-v2-m3, CPU) | ~16 s/query (verses), ~37 s/query (Steeks), 50 candidates |

| 18 | Reranker, 200 q, top 20 (`rerank_results_fair.txt`) | Worse alone; fused ≤3 queries better. Not adopted |
| 19 | **Context vectors** (`evaluate_context.py` → `context_results.txt`): line blended with mean of its pada / Shabad, no retraining | Blends tie or lose slightly (line+0.5 Shabad H@10 41.23 = line-only). Pada-only H@10 6.77, Shabad-only 3.26 — far fewer word-matches (75–86% of top-10 share no word) but they lose the *specific* line. Caveat: the gold set is line-level, so it is biased against context; a fair test needs pada-level text encoding + a pada-level benchmark (user question 2026-10-27) |

Rows 1–7 were measured on the old 919-query gold set; do not mix them with §5a.

---

## 10. Things that failed, and why

| Failure | Cause |
|---|---|
| Steek-pair fine-tuning (exp. 4/5) | Wrong objective. Aligning a verse to its commentary does not place two similar verses together. |
| `merge_and_unload()` crash | Used `peft.get_peft_model()` to wrap the model. sentence-transformers silently unwraps it during checkpointing, so the trained adapters never reached the saved model. Fixed with `model.add_adapter()`. |
| First safety check aborted every run | Compared base weights, but LoRA leaves base weights untouched by design. Replaced with a behavioural probe. |
| `\b` in regex never matched after a matra | Gurmukhi vowel signs are combining marks, not word characters. Replaced with an explicit `(?=[\s॥।]|$)` terminator. |
| TF-IDF scored unrelated verses at **1.000** | scikit-learn's default `token_pattern` kept only consonant runs; `"ਗਾਵੈ ਕੋ ਵਿਦਿਆ ਵਿਖਮੁ ਵੀਚਾਰੁ"` and `"ਕੋਊ ਬਿਖਮ ਗਾਰ ਤੋਰੈ"` both reduced to `['ਖਮ']`. **This invalidated the first gold set and every number measured against it.** |
| Gurmukhi rendered with matras after their consonant | `\newfontfamily` without `[Script=Gurmukhi]` — the OpenType shaper never ran. |
| My figure-replacement script deleted 28 `\begin{figure}[H]` lines | Split the file on the delimiter and never re-inserted it. Caught by a diff check, restored from backup, redone by operating on `tikzpicture` spans directly. **Always diff against a backup after a scripted edit to the report.** |
| Headless Edge screenshot | Produced no file regardless of flags. Headless **Chrome** works; that is what generated the PNG figures. |
| Browser-pane screenshots for report figures | The pane caps output at ~800 px and screenshots cannot be written to disk — unusable for print figures. Replaced with headless Chrome at `--force-device-scale-factor=2`, auto-cropped with Pillow. |
| Inferring file state from timestamps | Concluded `fig_diag_dfd1.png` was out of sync with its generator because the mtimes were close. It was not. **Open the artifact and look at it**; mtime proves nothing about content. |
| PIL / .NET for rendering Gurmukhi text | Neither does complex-script shaping reliably without HarfBuzz. Browser (Chrome) and XeLaTeX both shape correctly; Pillow is used only for cropping. |
| Reranker eval killed by out-of-memory | Web server (second BGE-M3 + vectors) running alongside the cross-encoder exceeded 11.8 GB. Run heavy jobs alone. Script now saves after each mode. |
| `curl` from Git Bash with Gurmukhi query args | Arguments were mangled before reaching the server (every query returned the same results). Test the API from Python `urllib` instead. |
| `diagrams.py` lost | Left in a session scratchpad; flagged but not copied. **Copy generators into the repo the moment they are written.** |
| PowerPoint COM automation | "Interface not registered" on this machine (32- and 64-bit, early and late binding). Slides cannot be rendered for QA here; no LibreOffice either. |

---

## 11. Remaining tasks

**System**
1. ~~Try the web "Why this match?" button live~~ — **done 2026-10-26**, works
   (fresh Groq call: `ਜੰਮਣੁ ਮਰਣਾ ਹੈ ਸੰਸਾਰੁ` → RELATED with an accurate "differs"
   note; cached pair rendered instantly).
   **Cleanup done 2026-10-26** (to Recycle Bin, ~390 MB): baseline `part_*.npy`
   shards (verified identical to embeddings.npy first), both `models/*/checkpoints/`,
   `cache.dat`, `__pycache__/`. The folder guide (`docs/project-guide.html`) still
   lists them — fix in the documentation pass.
2. Optional: LLM verification pass that checks a search's top results and flags
   `different` verdicts in the web page (~10 free requests per search).
0. **PADA DIRECTION (active, user-led, 2026-10-27/28).** User argued a single
   Pankati lacks context; the pada (lines up to ॥n॥ / Rahao, mean 3 lines,
   median 48 tokens, 99.9% ≤ 256) is the unit of meaning. Built and tested:
   `build_padas.py` (18,506 padas; 66 no-Steek lines excluded) →
   `build_pada_eval.py` (pada gold, threshold **0.15** chosen from samples;
   502 queries / 999 pairs, only 8% zero-word → biased toward keyword search) →
   `build_pada_index.py` (user ran it, ~1.5 h on laptop) → `evaluate_padas.py`:
   char-BM25 48.80 H@10 / MRR 31.11; line-mean verse_hn 30.68 / 20.85;
   **pada text verse_hn 39.04 / 25.06, zero@10 10.94% (7 of 64)**.
   Next: `mine_pada_pairs.py` → 3,534 positives (pada_pairs.jsonl), 2,227 with
   hard negatives (pada_pairs_hn.jsonl). Sample reading showed pada-level hard
   negatives are unreliable (some teach the SAME thing as the anchor), so train
   A = positives only (main) and B = with negatives (comparison). `train_bge.py`
   gained `--max-seq` (256 for padas) and `--val`. Everything for the GPU
   desktop is in `E:\gurbani-transfer-pada\` with `README_GPU.txt`.
   **Labelling paused** (user decision) until the pada result decides the unit.
   **Trained on Colab T4 (2026-10-28)** — needed `pip uninstall torchao` (peft
   0.21 rejects Colab's torchao 0.10) and `--mini-batch 8` (cached MNRL; batch 32
   × 256 tokens OOM'd). Model A loss 2.26→1.23, eval 2.30/2.34, reload 1.000.
   Pada gold results (502 q):

   | System | H@1 | H@10 | H@50 | MRR | zero@10 | no-word |
   |---|---|---|---|---|---|---|
   | char-BM25 | 23.11 | **48.80** | **66.53** | **31.11** | 0.00 | 0.3 |
   | pada text: verse_hn (untrained on padas) | 19.12 | 39.04 | 53.59 | 25.06 | 10.94 | 12.8 |
   | **pada text: pada (A, positives only)** | **20.52** | 42.63 | 61.95 | 28.11 | 9.38 | 8.8 |
   | pada text: pada_hn (B, + hard negatives) | 19.92 | 39.44 | 59.56 | 26.79 | 7.81 | 9.5 |

   A beats B → pada-level hard negatives hurt, as the sample reading predicted.
   Training helps (+3.6 H@10, +8.4 H@50) but A stays ~6 H@10 below BM25 on a
   gold set where 92% of pairs share words. Models in `models/bge-m3-pada{,-hn}`,
   embeddings in `data/embeddings/pada_pada{,_hn}`, logs in data/processed.
   **Hybrid (RRF k=60, untuned; `--fuse` in evaluate.py / evaluate_padas.py):**

   | Level | System | H@1 | H@10 | H@50 | MRR | zero@10 |
   |---|---|---|---|---|---|---|
   | lines (827 q) | char-BM25 | 16.08 | 40.87 | 59.98 | 23.77 | 0.92 |
   | | verse_hn | **18.02** | 41.23 | 58.77 | 25.61 | **5.53** |
   | | fused | 16.69 | **45.34** | **65.54** | **25.97** | 1.38 |
   | padas (502 q) | char-BM25 | 23.11 | 48.80 | 66.53 | 31.11 | 0.00 |
   | | pada (A) | 20.52 | 42.63 | 61.95 | 28.11 | **9.38** |
   | | fused | **23.90** | **51.79** | **71.91** | **33.25** | 0.00 |

   Fusion wins overall at both levels (+4.1 H@10 lines ≈ 34 queries; +3.0 padas
   ≈ 15 queries) but buries meaning-only matches (zero-overlap hits collapse,
   no-word 0.7%): a verse only the model finds gets half an RRF score. Line and
   pada numbers are on different gold sets — not comparable across levels.
   Line results in `fusion_results.txt`; `baseline_results.txt` untouched.
   **Decision (user, 2026-10-28): padas + two-section results, built into the
   website.** `src/api/app.py` rewritten for padas (hybrid main list + "same
   meaning, different words" section + best-line highlight + pada explanations);
   `index.html` rebuilt. "No shared words" uses consonant skeletons — exact
   spelling let ਧਨੁ results through for a ਧਨ query. Tested live: `ਨਵ ਨਿਧਿ`
   query now returns the "wealth of Naam" theme; `ਜੰਮਣੁ ਮਰਣਾ ਠਾਕਿ ਰਹਾਏ` (old
   opposite-meaning failure) now tops with a "die in the Shabad, liberated while
   alive" pada; pada explanation via Groq ~8 s, verdict "related".
   **Next: rebuild the human-labelling sheets at pada level** (main + different
   words + keyword-only candidates), then resume the team's labelling.
3. **Hand-labelled check set — sheets built 2026-10-26, PAUSED (see 0).**
   `backend/pipeline/evaluation/build_label_sheet.py` → `data/labeling/label_sheet_{Balkaran,
   Gurpreet,Gurveer}.xlsx` (344 / 374 / 379 rows) + `answer_key_DO_NOT_OPEN.csv`.
   50 random test-split queries (not drawn from the mined gold set), top 10 of
   verse_hn + top 10 of char-BM25 pooled and shuffled, blind; 18.3 candidates
   per query because the two systems share only 86 of ~1,000 results. 10 queries
   double-labelled (rotated A-B, B-C, C-A) for agreement. Labels 2/1/0.
   **Still to write:** a scoring script (H@10, MRR, nDCG on human labels;
   Cohen's kappa on the 10 shared queries; each system's unique finds).
4. Optional: paired bootstrap test verse_hn vs char-BM25 ("tie vs win").
5. Optional cleanup: `models/*/checkpoints/` (only for resuming training) and
   `data/embeddings/baseline/part_*.npy` (duplicates of embeddings.npy).

**Report / slides (only when the user says)**
6. Replace 919-query numbers with §5a; winner is `verse_hn`; state the H@10 tie.
7. Still open from before: DFD `results` label question; ToC +1 offset;
   `pygments`/`pillow`/`pymupdf` missing from the requirements files.

---

## 12. Next recommended steps

1. Start the web server and test "Why this match?" live.
2. Decide with the user on the LLM verification pass (§11.2) and the
   hand-labelled check set (§11.3).
3. Never run heavy CPU/RAM jobs while the web server is up (§10).

---

## 13. Important commands

```bash
# activate
cd E:\Major_Project_Gurveer; .\.venv\Scripts\Activate.ps1

# full data pipeline (CPU, minutes except build_index)
python backend\pipeline\preprocessing\clean_dataset.py
python backend\pipeline\training\create_training_pairs.py
python backend\pipeline\evaluation\build_eval_sets.py
python backend\pipeline\training\mine_verse_pairs.py
python backend\pipeline\training\mine_hard_negatives.py

# encode + score  (build_index is ~112 min on this CPU, ~3 min on the 3060)
python backend\pipeline\retrieval\build_index.py --tag baseline
python backend\pipeline\evaluation\evaluate.py --bm25 --bm25-char --dense baseline

# train (GPU machine only)
python backend\pipeline\training\train_bge.py --pairs verse_pairs_hn.jsonl --lora --batch-size 64 --epochs 5 --output models\bge-m3-hn

# interactive search (default tag verse_hn), two models side by side
python backend\pipeline\retrieval\semantic_search.py --compare baseline

# reranker evaluation (CPU: ~20 min per 200 queries in verse mode)
python backend\pipeline\evaluation\evaluate_rerank.py --sample 200 --candidates 20 --mode pankati --mode ft

# explanation (needs a free LLM_API_KEY in .env, Groq by default)
python common\explain_match.py --query "ਵਿਸਰੀ ਤਿਸੈ ਪਰਾਈ ਤਾਤਾ" --top 3

# web interface -> http://localhost:8000
python -m uvicorn src.api.app:app --port 8000

# report
cd "mid term report"; xelatex Midterm_report.tex; bibtex Midterm_report; xelatex Midterm_report.tex; xelatex Midterm_report.tex
```

Diagram PNGs were generated by `diagrams.py`, which **no longer exists** — its
temp folder was cleaned up. The PNGs in `mid term report/` are the only copy.

---

## 14. Assumptions and constraints

- **This laptop cannot train.** i3-1005G1 (2 cores, 1.2 GHz), 11.8 GB RAM, Intel
  UHD, no CUDA. Fine-tuning BGE-M3 needs ~9 GB for optimizer state alone.
  Training happens on a separate **RTX 3060 (12 GB)** desktop with limited
  availability — plan GPU sessions to produce model *and* embeddings before
  leaving.
- Corpus encoding: **112 min** on this CPU vs ~3 min on the 3060. Resumable in
  2,000-verse shards.
- Queries already in the corpus are answered from cached vectors and never load
  the 2.3 GB model — this is what keeps search usable locally.
- **Not a git repository.** No history to fall back on; take explicit backups
  before scripted edits to the report.
- `models/` and `data/embeddings/` are gitignored.
- The user edits `Midterm_report.tex` directly and has asked that **only
  explicitly requested changes** be made to it. Verify with a diff afterwards.
- Benchmark caveats that must survive into any write-up: measured recall is a
  **floor** (the mined gold set has missing relevant verses), it contains some
  wrong pairs, and it is derived from the same Steeks the model trains on.
- Report must stay **≥ 35 pages** (currently 36 — only one page of margin).
