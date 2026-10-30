# Semantic Retrieval and Contextual Exploration System for Sri Guru Granth Sahib Ji

B.Tech major project — Department of CSE, Guru Nanak Dev Engineering College, Ludhiana.
**Team:** Balkaran Singh, Gurpreet Kaur, Gurveer Singh · **Guide:** Dr. Parminder Singh

Given a line of Gurbani, the system finds passages elsewhere in Sri Guru Granth Sahib Ji that
carry the same meaning (*Bhavarth*) — even when they share no words — with their Steeks, Ang,
Raag and Shabad. It can also answer a question ("How can I remove my ego?", "ਮਨ ਨੂੰ ਸ਼ਾਂਤੀ ਕਿਵੇਂ
ਮਿਲੇ?") with the Shabads that answer it, explained in the language it was asked in.

There are no verse-to-verse similarity labels anywhere in the data. Supervision comes from the
commentaries: two verses that mean the same thing are explained in nearly the same modern-Punjabi
words, so agreement between independent Steeks is used to mine both the training pairs and the
evaluation set.

---

## Project structure

```
Major_Project_Gurveer/
├── frontend/                 Website: HTML + Tailwind CSS + JavaScript (dark, glowing theme)
│   ├── index.html
│   ├── assets/               compiled CSS, JS modules, icon
│   └── src/input.css         Tailwind source (rebuild: npm run build:css)
│
├── backend/
│   ├── app/                  FastAPI server: search engine, question answering, explanations
│   └── pipeline/             Data and model pipeline, one standalone script per step
│       ├── preprocessing/    clean the dataset, build padas
│       ├── training/         grouped split, mine training pairs, fine-tune BGE-M3 (GPU)
│       ├── retrieval/        encode the corpus, command-line search
│       └── evaluation/       gold sets and scoring
│
├── common/                   Code shared by the server and the pipeline (LLM explanation layer)
├── data/
│   ├── raw/                  original BaniDB downloads, one JSON per Ang
│   ├── metadata/             Raag list
│   ├── processed/            cleaned dataset, padas, pairs, gold sets, result reports, caches
│   ├── embeddings/<tag>/     the corpus encoded by each model (one folder per model)
│   └── labeling/             human-labelling sheets
├── models/                   fine-tuned LoRA adapters (~26 MB each)
├── docs/                     project documentation
├── tools/data_collection/    the original scripts that downloaded and assembled the dataset
├── mid term report/          college submission (XeLaTeX)
│
├── .env.example              every path and setting (copy to .env)
└── requirements*.txt         Python dependencies
```

Every stage reads files and writes new files, so any step can be re-run on its own. No script
hard-codes a path: everything comes from `.env`.

---

## Run the website

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-cpu.txt         # laptop (no NVIDIA GPU); requirements-gpu.txt with CUDA
pip install -r backend/requirements.txt
copy .env.example .env                      # then add a free Groq key as LLM_API_KEY (optional)

python -m uvicorn backend.app.main:app --port 8000
```

Open <http://localhost:8000>. The server is ready in about 10 seconds.

- **Search by Gurbani** — type a Pankati, several, or a whole Shabad. Results combine keyword
  search and the fine-tuned meaning model; a second section lists matches that share *no word*
  with the query, checked against the Steeks.
- **Ask a question** — in English or Punjabi; returns two or three whole Shabads with an
  explanation, and practical guidance when the question asks for it.

Explanations use a free LLM (Groq). Without `LLM_API_KEY` the search still works; explanations
and questions are switched off.

---

## Pipeline

Run from the project root, in order. Each step reads the previous step's output.

| # | Command | Produces |
|---|---|---|
| 1 | `python backend/pipeline/preprocessing/clean_dataset.py` | `cleaned_gurbani.csv`, audit of excluded lines |
| 2 | `python backend/pipeline/training/create_training_pairs.py` | corpus, grouped train/val/test split |
| 3 | `python backend/pipeline/evaluation/build_eval_sets.py` | gold evaluation pairs (test split) |
| 4 | `python backend/pipeline/training/mine_verse_pairs.py` | verse↔verse training pairs (train split) |
| 5 | `python backend/pipeline/training/mine_hard_negatives.py` | pairs with hard negatives |
| 6 | `python backend/pipeline/retrieval/build_index.py --tag baseline` | `data/embeddings/baseline/` |
| 7 | `python backend/pipeline/evaluation/evaluate.py --bm25-char --dense baseline` | results report |

Padas (the unit the website searches):

| Command | Produces |
|---|---|
| `python backend/pipeline/preprocessing/build_padas.py` | `padas.jsonl` |
| `python backend/pipeline/evaluation/build_pada_eval.py` | pada gold set |
| `python backend/pipeline/training/mine_pada_pairs.py` | pada training pairs |
| `python backend/pipeline/retrieval/build_pada_index.py --model models/bge-m3-pada --tag pada` | `data/embeddings/pada_pada/` |
| `python backend/pipeline/evaluation/evaluate_padas.py --bm25-char --dense pada --fuse pada` | pada results |

Training needs a CUDA GPU (RTX 3060 or Google Colab T4):

```bash
python backend/pipeline/training/train_bge.py --pairs verse_pairs_hn.jsonl --lora --batch-size 64 --epochs 5 --output models/<name>
```

The source workbook `data/processed/final_gurbani_dataset_all_steeks.xlsx` is never modified.

---

## Results

Verse-to-verse retrieval on the 827-query gold set (`data/processed/baseline_results.txt`):

| System | Hit@1 | Hit@10 | MRR |
|---|---|---|---|
| BM25 keyword search (char 3–5-gram) | 16.08 | 40.87 | 23.77 |
| BGE-M3 pretrained | 14.03 | 29.38 | 19.39 |
| + Steek-pair fine-tuning | 13.78 | 28.90 | 19.31 |
| + verse-pair fine-tuning | 17.78 | 37.97 | 24.46 |
| **+ verse pairs + hard negatives (final line model)** | **18.02** | **41.23** | **25.61** |

Pretrained BGE-M3 loses to keyword search, which is what justifies Gurbani-specific fine-tuning.
Training on verse↔commentary pairs did not transfer to verse↔verse; training on verse↔verse pairs
did. The fine-tuned model's Hit@10 lead over keyword search is a statistical tie; its advantage is
at the top of the ranking and on pairs with no shared words. At pada level, keyword search and
the pada model combined by reciprocal-rank fusion reach Hit@10 51.8 (keyword alone 48.8).

---

## Key data decisions

- **Training uses Steeks 1–3 only** (Sahib Singh, Manmohan Singh, Faridkot Teeka). BaniDB's Steek
  is byte-identical to Sahib Singh's in 99.9% of rows; the padd-arth glossary is not a paraphrase.
- **The train/val/test split is grouped, never random.** Rows sharing a Shabad, an identical
  Pankati or an identical Steek stay together; otherwise near-identical verses straddle the split
  and the comparison measures leakage, not learning.
- **Structural lines (headings) are flagged, not deleted**, and exported for audit.
- **Gurmukhi needs its own tokenizer.** Vowel signs are Unicode combining marks: `\b` and
  scikit-learn's default token pattern break Gurmukhi words apart.

## Caveats

- The gold set is mined from Steeks, and the models train on Steeks, so it is not fully
  independent of the training signal. All systems are scored on the identical set.
- Pairs whose Steeks use different vocabulary are missed, so measured recall is a floor.
- LLM explanations are a study aid, not an authority; read the Bani and the full Steeks.
