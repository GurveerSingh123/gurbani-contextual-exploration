"""
One-line reasons + a verdict for every result card, from ONE LLM request.

Two jobs in one call:
  * a short reason shown on each card ("what teaching does it share?")
  * a verdict -- same / related / different -- which the frontend uses to FILTER
    the "same meaning, different words" section: the meaning model proposes a
    pool of word-free candidates, and only those judged same or related are
    shown. Without this check the section filled every slot, including weak
    matches the reason line itself called unrelated.

A full explanation per card would cost ~10+ requests and well over Groq's free
~8K tokens a minute. So all cards go in one request, from Steek 1 only
(trimmed harder for big batches), to a smaller model with its own free quota
(LLM_SUMMARY_MODEL). The detailed "Why this match?" keeps the larger model.
Every answer is cached, so a repeated search costs nothing.
"""

import json
import os
import threading

from . import config

SUMMARY_MODEL = os.getenv("LLM_SUMMARY_MODEL", "openai/gpt-oss-20b")
CACHE_KEY = f"{SUMMARY_MODEL}|v2"            # v2 = with verdicts; v1 rows are ignored
CACHE_JSONL = config.PROCESSED / "summaries_cache.jsonl"
VERDICTS = ("same", "related", "different")

SYSTEM = """You help readers of Sri Guru Granth Sahib Ji see why a search \
returned each passage for their query. You get the QUERY passage and several \
numbered CANDIDATE passages, each with a short commentary (Steek).

For every candidate, based only on the Steeks:
- "verdict": "same" if it carries the same teaching as the query, "related" if \
it shares the theme but the teaching differs in scope or emphasis, "different" \
if the teaching is unrelated or opposed.
- "reason": ONE short English sentence (at most 30 words) saying what teaching \
it shares with the query -- or, for "different", plainly what differs.
Do not add interpretations the Steeks do not give. Be respectful.

Reply with a single JSON object and nothing else:
{"items": [{"id": <candidate number>, "verdict": "...", "reason": "..."}, ...]}"""

_lock = threading.Lock()


def _load_cache() -> dict:
    cache = {}
    if CACHE_JSONL.exists():
        for line in CACHE_JSONL.open(encoding="utf-8"):
            row = json.loads(line)
            if row.get("model") == CACHE_KEY and row.get("verdict") in VERDICTS:
                cache[(row["query"], row["pada_id"])] = {"reason": row["reason"], "verdict": row["verdict"]}
    return cache


def _passage(engine, pada_id, chars) -> tuple[str, str]:
    p = engine.padas[pada_id]
    steek = " ".join((engine.occ.get(oid, {}).get("steek_ss") or "").strip() for oid in p["line_ids"])
    return p["text"], steek[:chars]


def reasons(engine, client, query_pada_ids, query_text, pada_ids) -> dict:
    """-> {"items": {pada_id: {"reason", "verdict"}}, "error": str|None}"""
    if query_pada_ids:
        texts = [_passage(engine, q, 520) for q in query_pada_ids]
        q_text = " ".join(t for t, _ in texts)
        q_steek = " ".join(s for _, s in texts)[:520]
    else:                                           # free text: no Steek to give
        q_text, q_steek = query_text, "(free text, no Steek)"

    with _lock:
        cache = _load_cache()
    out = {pid: cache[(q_text, pid)] for pid in pada_ids if (q_text, pid) in cache}
    todo = [pid for pid in pada_ids if pid not in out]
    if not todo:
        return {"items": out, "error": None}

    # ~260 Steek characters per candidate fits 12 in one request; bigger batches
    # get shorter excerpts so a search stays inside the free per-minute limit.
    chars = 260 if len(todo) <= 12 else 180
    lines = [f"QUERY: {q_text}", f"Steek: {q_steek}", ""]
    for n, pid in enumerate(todo, 1):
        text, steek = _passage(engine, pid, chars)
        lines += [f"CANDIDATE {n}: {text}", f"Steek: {steek}", ""]

    import openai
    data, problem = None, None
    for _ in range(2):                  # one retry: a cut-off answer is rejected as invalid JSON
        try:
            resp = client.chat.completions.create(
                model=SUMMARY_MODEL,
                messages=[{"role": "system", "content": SYSTEM},
                          {"role": "user", "content": "\n".join(lines)}],
                response_format={"type": "json_object"},
                temperature=0.2,
                # gpt-oss "thinks" first and that counts against the limit; low
                # effort is plenty for one-line judgements and keeps room to answer.
                reasoning_effort="low",
                max_completion_tokens=4000,
            )
            data = json.loads(resp.choices[0].message.content or "{}")
            break
        except openai.RateLimitError:
            return {"items": out, "error": "The free LLM limit was reached; try again in a minute."}
        except (openai.APIError, json.JSONDecodeError) as e:
            problem = f"Could not check the matches ({type(e).__name__})."
    if data is None:
        return {"items": out, "error": problem}

    new = {}
    for item in data.get("items", []):
        try:
            n = int(item["id"])
            verdict = str(item["verdict"]).strip().lower()
            reason = str(item["reason"]).strip()
        except (KeyError, TypeError, ValueError):
            continue
        if 1 <= n <= len(todo) and reason and verdict in VERDICTS:
            new[todo[n - 1]] = {"reason": reason, "verdict": verdict}
    if new:
        with _lock, CACHE_JSONL.open("a", encoding="utf-8") as f:
            for pid, v in new.items():
                f.write(json.dumps({"model": CACHE_KEY, "query": q_text, "pada_id": pid, **v},
                                   ensure_ascii=False) + "\n")
    out.update(new)
    return {"items": out, "error": None}
