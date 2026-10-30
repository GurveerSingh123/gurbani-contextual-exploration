"""
Step 4: explain WHY two Pankatis were retrieved together.

The retriever returns verses that score close in vector space; it cannot say
what they share, and it is sometimes fooled by wording (a verse saying the cycle
of birth and death is STOPPED can surface one saying it NEVER ends). This stage
gives an LLM both verses with every Steek the dataset holds for them, and asks
for the shared Bhavarth -- grounded in those commentaries, not in its own
reading -- plus a verdict: same / related / different. The verdict doubles as a
check on the retriever's false positives.

  python common/explain_match.py --query "ਵਿਸਰੀ ਤਿਸੈ ਪਰਾਈ ਤਾਤਾ" --match "ਛਾਡਿ ਵਿਡਾਣੀ ਤਾਤਿ ਮੂੜੇ"
  python common/explain_match.py --query "ਵਿਸਰੀ ਤਿਸੈ ਪਰਾਈ ਤਾਤਾ" --top 3

Both verses must be in the corpus (that is where the Steeks come from).
Answers are cached, so the same pair is never paid for twice.

Needs LLM_API_KEY in .env (Groq's free tier by default; see LLM_BASE_URL / LLM_MODEL).

Reads  : data/processed/corpus_pankatis.jsonl, data/metadata/raag_metadata.csv
         data/embeddings/<tag>/{embeddings.npy,corpus_meta.jsonl}   (--top only)
Writes : data/processed/explanations_cache.jsonl
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

PROCESSED = PROJECT_ROOT / os.getenv("PROCESSED_DATA_DIR", "data/processed")
EMBED_DIR = PROJECT_ROOT / os.getenv("EMBEDDING_DIR", "data/embeddings")
RAAG_META = PROJECT_ROOT / os.getenv("RAAG_METADATA", "data/metadata/raag_metadata.csv")
CORPUS_JSONL = PROCESSED / "corpus_pankatis.jsonl"
CACHE_JSONL = PROCESSED / "explanations_cache.jsonl"

# Any OpenAI-compatible chat endpoint. Defaults to Groq's free tier; Gemini,
# OpenRouter and Cerebras work by changing these three values in .env.
BASE_URL = os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")
MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-120b")
API_KEY_VAR = "LLM_API_KEY"

STEEKS = [  # (column, how the prompt names it)
    ("steek_ss", "Prof. Sahib Singh (Punjabi)"),
    ("steek_ms", "Manmohan Singh (Punjabi)"),
    ("steek_ft", "Faridkot Teeka (Braj)"),
    ("steek_pss", "Padd-arth word glossary"),
]

SYSTEM = """You help readers of Sri Guru Granth Sahib understand why a search \
system placed two passages together. Each passage is a single Pankati or a pada \
(a few consecutive Pankatis); below, "Pankati" means either.

You are given two Pankatis, each with its Ang, Raag and the commentaries \
(Steeks) available for it. Work only from those Steeks:
- Every statement about what a Pankati means must be supported by its Steeks. \
Do not add interpretations, stories or context the Steeks do not give.
- If a Steek is missing or the Steeks disagree, say so rather than filling the gap.
- Quote Gurmukhi words exactly as they appear in the input.

Then judge the pair:
- "same": both Pankatis carry the same teaching.
- "related": they share a theme, but the teachings differ in scope or emphasis.
- "different": they were matched on shared wording or a surface theme, but \
their meanings differ or are opposed. Say plainly what differs.

Write the explanation for a reader who knows Punjabi but is not a scholar. \
Keep the tone respectful, and refer to Gurbani in its own terms.

Reply with a single JSON object and nothing else, with exactly these keys:
  "verdict": "same", "related" or "different"
  "shared_bhavarth_pa": the shared meaning in one or two sentences of Punjabi \
in Gurmukhi script (if the verdict is "different", each verse's meaning instead)
  "shared_bhavarth_en": the same in English
  "explanation_en": two to four English sentences on how each Pankati expresses \
the meaning, naming the Steek each point comes from
  "key_concepts": a list of up to five Gurmukhi words or short phrases that \
carry the shared meaning
  "difference": what differs between the two Pankatis, or "" if nothing"""

SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["same", "related", "different"]},
        "shared_bhavarth_pa": {
            "type": "string",
            "description": "The shared meaning in one or two sentences of Punjabi (Gurmukhi script). "
                           "If the verdict is 'different', state each verse's meaning instead.",
        },
        "shared_bhavarth_en": {"type": "string",
                               "description": "The same, in English."},
        "explanation_en": {
            "type": "string",
            "description": "Two to four sentences: how each Pankati expresses the meaning, "
                           "naming the Steek each point comes from.",
        },
        "key_concepts": {
            "type": "array", "items": {"type": "string"},
            "description": "Up to five Gurmukhi words or short phrases that carry the shared meaning.",
        },
        "difference": {"type": "string",
                       "description": "What differs between the two Pankatis; empty string if nothing."},
    },
    "required": ["verdict", "shared_bhavarth_pa", "shared_bhavarth_en",
                 "explanation_en", "key_concepts", "difference"],
    "additionalProperties": False,
}

_NOISE = re.compile(r"[॥।0-9੦-੯]+")


def lookup_key(text: str) -> str:
    """Loose key for corpus lookup: nobody types the danda and verse numbers."""
    return " ".join(_NOISE.sub(" ", text).split())


def load_raag_names() -> dict:
    if not RAAG_META.exists():
        return {}
    df = pd.read_csv(RAAG_META, encoding="utf-8")
    return {int(r.raag_id): str(r.raag)
            for r in df.itertuples() if not pd.isna(r.raag_id)}


def find(docs, by_text, by_key, text):
    """Exact -> punctuation-insensitive -> unique partial. Returns index or None."""
    if text in by_text:
        return by_text[text]
    key = lookup_key(text)
    if key in by_key:
        return by_key[key]
    if len(key) >= 8:
        partial = [i for k, i in by_key.items() if key in k]
        if partial:
            return min(partial, key=lambda i: len(docs[i]["pankati"]))
    return None


def describe(label, doc, raag_names, columns=STEEKS) -> str:
    occ = doc["occurrences"][0]
    raag = raag_names.get(occ["raag_id"], "unknown") if occ["raag_id"] else "unknown"
    lines = [f"<{label}>",
             f"Pankati: {doc['pankati']}",
             f"Ang {occ['ang']}, {raag}, Shabad {occ['shabad_id']}"]
    for col, name in columns:
        text = (occ.get(col) or "").strip()
        lines.append(f"{name}: {text if text else '(not available)'}")
    lines.append(f"</{label}>")
    return "\n".join(lines)


def load_cache() -> dict:
    cache = {}
    if CACHE_JSONL.exists():
        for line in CACHE_JSONL.open(encoding="utf-8"):
            row = json.loads(line)
            cache[(row["model"], row["query"], row["match"])] = row["explanation"]
    return cache


def explain(client, query_doc, match_doc, raag_names, cache, columns=STEEKS) -> dict:
    """columns: which Steeks to send. Padas pass the three paraphrases only --
    the word glossary roughly doubles the prompt, and Groq's free tier caps a
    request at 8K tokens per minute."""
    # Different Steek labels give differently worded answers, so they are cached apart.
    model_key = MODEL if columns is STEEKS else f"{MODEL}|{'/'.join(n for _, n in columns)}"
    key = (model_key, query_doc["pankati"], match_doc["pankati"])
    if key in cache:
        return cache[key]

    import openai
    prompt = (describe("query", query_doc, raag_names, columns) + "\n\n"
              + describe("match", match_doc, raag_names, columns)
              + "\n\nWhy were these two Pankatis retrieved together? Judge the pair.")

    result, problem = None, None
    for _ in range(2):                  # free models sometimes miss the format; retry once
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "system", "content": SYSTEM},
                          {"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.2,
                max_completion_tokens=4000,
            )
        except openai.AuthenticationError:
            sys.exit(f"The API key was rejected. Check {API_KEY_VAR} in .env.")
        except openai.RateLimitError:
            sys.exit("The free tier's rate limit was reached. Wait a minute and try again.")
        except openai.APIStatusError as e:
            sys.exit(f"API error {e.status_code}: {e.message}")
        except openai.APIConnectionError:
            sys.exit(f"Could not reach {BASE_URL}. Check the internet connection.")

        choice = response.choices[0]
        if choice.finish_reason == "length":
            problem = "The explanation was cut off."
            continue
        result, problem = parse(choice.message.content or "")
        if result is not None:
            break
    if result is None:
        return {"error": f"{problem} Try again, or switch LLM_MODEL in .env."}

    cache[key] = result
    with CACHE_JSONL.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"model": key[0], "query": key[1], "match": key[2],
                            "explanation": result}, ensure_ascii=False) + "\n")
    return result


def parse(text: str):
    """JSON reply -> (result, None) if it has the expected shape, else (None, why)."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None, "The model did not return valid JSON."
    props = SCHEMA["properties"]
    missing = [k for k in SCHEMA["required"] if k not in data]
    if missing:
        return None, f"The model's answer lacked {', '.join(missing)}."
    if data["verdict"] not in props["verdict"]["enum"]:
        return None, f"Unknown verdict '{data['verdict']}'."
    if not isinstance(data["key_concepts"], list):
        data["key_concepts"] = [str(data["key_concepts"])]
    return {k: data[k] for k in SCHEMA["required"]}, None


def make_client():
    import openai
    return openai.OpenAI(api_key=os.getenv(API_KEY_VAR), base_url=BASE_URL)


def top_matches(qi, tag, k):
    """Top-k cross-Shabad results for a corpus verse, from cached vectors."""
    meta_path, emb_path = EMBED_DIR / tag / "corpus_meta.jsonl", EMBED_DIR / tag / "embeddings.npy"
    for p in (meta_path, emb_path):
        if not p.exists():
            sys.exit(f"Missing {p}. Run build_index.py --tag {tag} first.")
    meta = [json.loads(l) for l in meta_path.open(encoding="utf-8")]
    emb = np.load(emb_path, mmap_mode="r")
    scores = emb @ emb[qi]
    exclude = set(meta[qi]["shabad_ids"])
    out, used = [], set()
    for i in np.argsort(-scores):
        shabads = set(meta[i]["shabad_ids"])
        if shabads & exclude or min(shabads) in used:
            continue
        used.add(min(shabads))
        out.append((int(i), float(scores[i])))
        if len(out) >= k:
            break
    return out


def show(rank, doc, score, result):
    occ = doc["occurrences"][0]
    head = f"{rank}. " if rank else ""
    sc = f"[{score:.3f}]  " if score is not None else ""
    print(f"\n{head}{sc}{doc['pankati']}   (Ang {occ['ang']})")
    if "error" in result:
        print(f"   {result['error']}")
        return
    print(f"   verdict : {result['verdict'].upper()}")
    print(f"   ਭਾਵਾਰਥ  : {result['shared_bhavarth_pa']}")
    print(f"   meaning : {result['shared_bhavarth_en']}")
    print(f"   why     : {result['explanation_en']}")
    if result["key_concepts"]:
        print(f"   key     : {', '.join(result['key_concepts'])}")
    if result["difference"]:
        print(f"   differs : {result['difference']}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", required=True, help="a Pankati from the corpus")
    ap.add_argument("--match", default=None, help="a second Pankati to compare with")
    ap.add_argument("--top", type=int, default=0,
                    help="instead of --match, explain the top N search results")
    ap.add_argument("--tag", default=os.getenv("SEARCH_TAG", "verse_hn"))
    args = ap.parse_args()
    if bool(args.match) == bool(args.top):
        ap.error("give exactly one of --match or --top")

    if not os.getenv(API_KEY_VAR):
        sys.exit(f"{API_KEY_VAR} is not set. Add it to .env (never to .env.example).")
    if not CORPUS_JSONL.exists():
        sys.exit(f"Missing {CORPUS_JSONL}. Run create_training_pairs.py first.")

    docs = [json.loads(l) for l in CORPUS_JSONL.open(encoding="utf-8")]
    by_text = {d["pankati"]: i for i, d in enumerate(docs)}
    by_key = {}
    for i, d in enumerate(docs):
        by_key.setdefault(lookup_key(d["pankati"]), i)

    qi = find(docs, by_text, by_key, args.query)
    if qi is None:
        sys.exit("Query Pankati not found in the corpus (its Steeks are needed).")

    if args.match:
        mi = find(docs, by_text, by_key, args.match)
        if mi is None:
            sys.exit("Match Pankati not found in the corpus (its Steeks are needed).")
        pairs = [(None, mi, None)]
    else:
        pairs = [(r, i, s) for r, (i, s) in enumerate(top_matches(qi, args.tag, args.top), 1)]

    client = make_client()
    raag_names = load_raag_names()
    cache = load_cache()

    print(f"QUERY: {docs[qi]['pankati']}   (Ang {docs[qi]['occurrences'][0]['ang']})")
    print(f"model: {MODEL}  via {BASE_URL}")
    for rank, mi, score in pairs:
        show(rank, docs[mi], score, explain(client, docs[qi], docs[mi], raag_names, cache))


if __name__ == "__main__":
    main()
