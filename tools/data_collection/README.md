# Data collection tools

The original scripts that downloaded Sri Guru Granth Sahib Ji from the BaniDB API and assembled
the source dataset, `data/processed/final_gurbani_dataset_all_steeks.xlsx`. They are kept for
provenance; the pipeline in `backend/pipeline/` starts from that finished workbook and does not
need them.

| File | Step |
|---|---|
| `collect_gurbani.py` | Downloads every Ang into `data/raw/ang_*.json` and builds the first dataset (Sahib Singh's Steek) |
| `adding_other_sources.py` | Adds the other Steeks from the raw files |
| `collecting_metadata.py` | Fetches each Shabad's metadata and enriches the dataset |
| `collecting_raag.py` | Fetches the Raag and writer lists into `data/processed/raag_metadata.csv` (the copy the pipeline reads is `data/metadata/raag_metadata.csv`) |
| `adding_raag_id_script.py` | Attaches a Raag id to every line |
| `banidb_raag_source.txt`, `banidb_raags_source.txt` | Raag lists as returned by BaniDB |
| `banidb_test.py`, `inspection.py`, `inspection.ipynb` | Exploration of the `banidb` Python library's API |

Notes:

- Run them **from the project root** (`python tools/data_collection/collect_gurbani.py`): their
  paths (`data/raw`, `data/processed`) are relative to it.
- They were written for the global Python installation, not the project's `.venv`.
- The intermediate datasets they produced (`gurbani_full_dataset*.csv`, `gurbani_enriched_*`,
  `gurbani_semantic_dataset.csv`) were moved out of the project to
  `E:\Major_Project_Gurveer_archive\data_processed_intermediate\`. Copy them back into
  `data/processed/` before re-running a step that reads one.
