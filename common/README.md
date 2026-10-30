# Common

Code shared by the server (`backend/app`) and the pipeline's command line.

| File | What it does |
|---|---|
| `explain_match.py` | The LLM explanation layer: given two passages, explains from their Steeks only whether they carry the same meaning, and returns a verdict (same / related / different). Uses any free OpenAI-compatible endpoint set in `.env` (Groq by default); answers are cached in `data/processed/explanations_cache.jsonl`. |

It is imported by the server rather than copied, so the website and the command line always use
the same prompt. It can also be run directly:

```bash
python common/explain_match.py --query "ਵਿਸਰੀ ਤਿਸੈ ਪਰਾਈ ਤਾਤਾ" --top 3
```
