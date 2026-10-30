"""
Ask a question ("how do I meditate?", "ਹਉਮੈ ਕਿਵੇਂ ਦੂਰ ਹੋਵੇ?") and get two or three
whole Shabads of Gurbani that answer it, each fully explained, in the language
the question was asked in, with practical guidance when the question asks for help.

A question and a verse share almost no words, so the question is matched against
the STEEKS -- modern Punjabi prose -- not the verses:

  1. terms    the LLM restates the question in Punjabi and lists the words a Steek
              (or Gurbani) uses for its core ideas: ego -> ਹਉਮੈ, ਅਹੰਕਾਰ, ਆਪਾ-ਭਾਵ
  2. search   keyword search (BM25) over every pada's Steeks -> 15 candidates,
              one per Shabad
  3. pick     a smaller model reads the candidates' Steeks and picks the two or
              three that actually answer the question (none, if none do)
  4. explain  the larger model explains each WHOLE Shabad for a seeker -- read with
              its Rahao (the Shabad's central idea) -- answers the question directly,
              and, if the question asks for help, gives guidance as Gurbani teaches it

The Steeks are used to FIND the Shabads and to keep the explanation faithful; the
explanation itself goes beyond paraphrasing them. It may quote only lines it was
given. Answers are cached, so a repeated question costs nothing.

Free only: Groq by default (LLM_*). Groq's gpt-oss writes understandable but
imperfect Punjabi; set ASK_LLM_BASE_URL / ASK_LLM_MODEL / ASK_LLM_API_KEY to use
another OpenAI-compatible free endpoint (e.g. Google Gemini) for steps 1 and 4.
"""

import json
import os
import re
import threading
import time

from . import config

QUESTION_MAX = 300
CALL_TIMEOUT = 60       # seconds per LLM call
RETRY_WAIT_MAX = 12     # on a rate limit, wait and retry once only if the server asks for <= this
CANDIDATES = 15
MAX_SHABADS = 3
_GURMUKHI = re.compile(r"[\u0A00-\u0A7F]")
PICK_MODEL = os.getenv("LLM_SUMMARY_MODEL", "openai/gpt-oss-20b")
CACHE_JSONL = config.PROCESSED / "ask_cache.jsonl"
PROMPT_VERSION = "v3"     # bump when the prompts change, so old cached answers are not reused

TERMS = """A reader asks a question about spiritual life. You will search Sri Guru \
Granth Sahib Ji, whose every line has a modern Punjabi commentary (Steek) by Prof. \
Sahib Singh. Give the words such a Steek, or the Gurbani itself, uses for the CORE \
concepts of THIS question (for example, for ego one would give ਹਉਮੈ, ਅਹੰਕਾਰ, ਆਪਾ-ਭਾਵ; \
for death ਮੌਤ, ਜਮ, ਕਾਲ -- use these only as a guide to the kind of words wanted). \
Only real Punjabi/Gurbani words; no English, no generic words (ਜੀਵਨ, ਮਨੁੱਖ, ਕਰਨਾ).
Reply with one JSON object: {"punjabi": "<the question as one plain Punjabi \
sentence>", "terms": ["<8-14 words or two-word phrases>"]}"""

PICK = """A reader asked a QUESTION. Below are numbered CANDIDATE passages (padas) of \
Sri Guru Granth Sahib Ji, each with its Steek (commentary). Pick the two or three \
candidates whose teaching most directly answers the question, best first -- a third \
only if it clearly adds to the answer. Judge only from the Steeks.
Reply with one JSON object: {"ids": [n, ...]} -- an empty list if none answers it."""

EXPLAIN = """You are a gentle, learned guide helping a seeker understand Sri Guru \
Granth Sahib Ji. The seeker asked a QUESTION. You are given two or three SHABADS of \
Gurbani that answer it. For each: the whole Shabad, the PADA in it that speaks most \
directly to the question with its Steek (commentary), and the Shabad's RAHAO, which \
carries the Shabad's central idea.

Write for someone who came seeking spiritual understanding, not a word-for-word \
gloss: explain what Guru Sahib teaches in each Shabad, why, and what it means for \
the seeker's question. Stay faithful to Gurbani and to Gurmat as the Steeks present \
it. Quote Gurbani ONLY from the lines given, exactly; never invent lines, Ang \
numbers, stories or teachings. Be reverent: say "Guru Sahib" and "Sri Guru Granth \
Sahib Ji".

If the question asks for help -- how to do something, how to overcome something, \
what to do -- give GUIDANCE: concrete steps a person can take, each drawn from these \
Shabads. Guidance must follow Gurmat as Gurbani itself teaches it: Naam Simran \
(remembering Waheguru with love), reading, singing and reflecting on Gurbani, the \
company of the Sangat, Seva, honest living (Kirat), sharing, and Ardas. NEVER suggest \
pictures or images of the Guru, idols, mantras counted on beads, fixed numbers of \
repetitions, fasting, pilgrimages or rituals -- Gurbani rejects these. Read Gurbani's \
idioms by their sense, as the Steek does: ਆਠ ਪਹਰ / ਅਠੇ ਪਹਰ means "at all times", not \
eight times; ਚਰਣ (the Lord's/Guru's feet) means humble, loving devotion, not a \
physical object of focus; ਨਿੰਦਾ means slandering others, not self-criticism. If the question only asks to understand something, \
guidance may be an empty list.

Write EVERYTHING in {language}. {language_rule}

Reply with one JSON object:
{{"answer": "<the direct answer to the question, drawn from these Shabads, 4-6 sentences>",
 "shabads": [{{"id": <shabad number>,
   "key_line": "<the one line of this Shabad that answers the question most directly, copied exactly>",
   "explanation": "<what this Shabad teaches, its central idea, and what it says to the seeker's question; at least 5 sentences>"}}],
 "guidance": ["<one concrete step, 1-2 sentences, saying which Shabad it comes from>", "... 3-5 steps, or []"]}}"""

LANGUAGE_RULES = {
    "Punjabi": "Pure, simple, correct modern Punjabi in Gurmukhi script, as Prof. Sahib "
               "Singh's Steek writes it -- lean on the Steek's own wording. NEVER use English "
               "words or Latin letters.",
    "English": "Clear, warm English. A Gurbani term may appear transliterated with its "
               "meaning, e.g. Haumai (ego).",
}

_lock = threading.Lock()


class AskError(Exception):
    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


def _clients():
    """(client, model) for writing (terms + explanation) and for picking."""
    import openai
    base = os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")
    # No silent retries: on a rate limit the SDK would otherwise wait and retry on
    # its own (with a 10-minute timeout), and the page just kept "searching".
    # _chat decides instead, and tells the reader.
    opts = {"max_retries": 0, "timeout": CALL_TIMEOUT}
    groq = openai.OpenAI(api_key=os.getenv("LLM_API_KEY"), base_url=base, **opts)
    if os.getenv("ASK_LLM_API_KEY"):
        writer = openai.OpenAI(api_key=os.getenv("ASK_LLM_API_KEY"),
                               base_url=os.getenv("ASK_LLM_BASE_URL", base), **opts)
        return (writer, os.getenv("ASK_LLM_MODEL", "gemini-2.5-flash")), (groq, PICK_MODEL)
    return (groq, os.getenv("ASK_LLM_MODEL") or os.getenv("LLM_MODEL", "openai/gpt-oss-120b")), (groq, PICK_MODEL)


def available() -> bool:
    return bool(os.getenv("LLM_API_KEY"))


def writer_model() -> str:
    return _clients()[0][1] if available() else ""


def _retry_after(err) -> float:
    try:
        return float(err.response.headers.get("retry-after", "60"))
    except (AttributeError, TypeError, ValueError):
        return 60.0


def _chat(client, model, system, user, max_tokens, effort="low") -> dict:
    import openai
    extra = {"reasoning_effort": effort} if "gpt-oss" in model else {}
    started = time.time()
    for attempt in range(2):
        try:
            resp = client.chat.completions.create(
                model=model, temperature=0.3, max_completion_tokens=max_tokens,
                response_format={"type": "json_object"},
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}], **extra)
            break
        except openai.RateLimitError as e:
            wait = _retry_after(e)
            if attempt == 0 and wait <= RETRY_WAIT_MAX:
                print(f"  ask: {model} rate-limited, waiting {wait:.0f} s", flush=True)
                time.sleep(wait)
                continue
            raise AskError(f"The free LLM limit for this minute is used up. Please ask again in about "
                           f"{max(5, round(wait))} seconds.", 429)
        except openai.APITimeoutError:
            raise AskError("The LLM service did not answer in time. Please try again.", 504)
        except openai.APIError as e:
            raise AskError(f"The LLM service failed ({type(e).__name__}). Try again.", 502)
    usage = getattr(resp, "usage", None)
    print(f"  ask: {model} {time.time() - started:.1f} s"
          + (f", {usage.prompt_tokens}+{usage.completion_tokens} tokens" if usage else ""), flush=True)
    try:
        return json.loads(resp.choices[0].message.content or "{}")
    except json.JSONDecodeError:
        return {}


def _norm(q: str) -> str:
    return " ".join(re.sub(r"[?？!.,।॥]+", " ", q.lower()).split())


def _load_cache(model) -> dict:
    out = {}
    if CACHE_JSONL.exists():
        for line in CACHE_JSONL.open(encoding="utf-8"):
            row = json.loads(line)
            if row.get("model") == model:
                out[row["question"]] = row["answer"]
    return out


def _excerpt(text, chars):
    text = (text or "").strip()
    return text if len(text) <= chars else text[:chars].rsplit(" ", 1)[0] + " …"


def _valid(ans: dict, n_chosen: int) -> bool:
    shabads = ans.get("shabads")
    if not isinstance(shabads, list) or not shabads or not str(ans.get("answer") or "").strip():
        return False
    for sh in shabads:
        if not isinstance(sh, dict) or not str(sh.get("id", "")).isdigit() or not 1 <= int(sh["id"]) <= n_chosen:
            return False
        if not str(sh.get("explanation") or "").strip():
            return False
    return isinstance(ans.get("guidance", []), list)


def _ids_by_position(ans: dict, angs: list) -> dict:
    """The model sometimes answers with the Ang number ("SHABAD 1 (Ang 29)" -> id 29)
    instead of the Shabad number. Map Ang -> number; failing that, use list order."""
    shabads = ans.get("shabads")
    if not isinstance(shabads, list):
        return ans
    for pos, sh in enumerate(shabads):
        if not isinstance(sh, dict):
            continue
        raw = str(sh.get("id", "")).strip()
        n = int(raw) if raw.isdigit() else 0
        if not 1 <= n <= len(angs):
            n = angs.index(n) + 1 if n in angs else (pos + 1 if pos < len(angs) else 0)
        sh["id"] = n
    return ans


def _scrub(text: str) -> str:
    """Last resort: drop bracketed English glosses like '(Attachment)' from Punjabi."""
    text = re.sub(r"\s*\([^)]*[A-Za-z][^)]*\)", "", text)
    return " ".join(text.split())


def ask(engine, question: str) -> dict:
    started = time.time()
    question = " ".join(question.split())
    if not question:
        raise AskError("Type a question.")
    if len(question) > QUESTION_MAX:
        raise AskError(f"Keep the question under {QUESTION_MAX} characters.")
    if not available():
        raise AskError("Questions need the free LLM: add LLM_API_KEY to .env and restart the server.", 503)
    (writer, wmodel), (picker, pmodel) = _clients()

    key = _norm(question)
    with _lock:
        cached = _load_cache(f"{wmodel}|{PROMPT_VERSION}").get(key)
    if cached is None:
        print(f"ask: {question!r}", flush=True)
        cached = _answer(engine, question, writer, wmodel, picker, pmodel)
        print(f"ask: answered in {time.time() - started:.1f} s", flush=True)
        with _lock, CACHE_JSONL.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"model": f"{wmodel}|{PROMPT_VERSION}", "question": key, "answer": cached}, ensure_ascii=False) + "\n")

    shabads = []
    for item in cached["shabads"]:
        view = engine.pada_view(item["pada_id"], item.get("key_line"))
        shabads.append({**view, "explanation": item["explanation"]})
    return {"question": question, "language": cached.get("language"), "read_as": cached.get("read_as"),
            "found": bool(shabads), "answer": cached.get("answer"), "guidance": cached.get("guidance", []),
            "shabads": shabads, "model": wmodel, "seconds": round(time.time() - started, 1)}


def _answer(engine, question, writer, wmodel, picker, pmodel) -> dict:
    # Answer in the language the question was asked in.
    language = "Punjabi" if _GURMUKHI.search(question) else "English"

    # 1. Punjabi search terms
    terms = _chat(writer, wmodel, TERMS, question, 1500)
    read_as = str(terms.get("punjabi") or "").strip()
    words = [str(t) for t in terms.get("terms") or [] if str(t).strip()]
    query = " ".join([read_as, *words, question])     # a Punjabi question also matches as typed
    empty = {"language": language, "read_as": read_as, "shabads": []}

    # 2. Steek search
    cands = engine.question_candidates(query, CANDIDATES)
    if not cands:
        return empty

    # 3. pick the two or three that answer it
    lines = [f"QUESTION: {question}", ""]
    for n, pid in enumerate(cands, 1):
        p = engine.padas[pid]
        lines += [f"CANDIDATE {n}: {p['text']}", f"Steek: {_excerpt(p.get('steek_ss'), 260)}", ""]
    ids = _chat(picker, pmodel, PICK, "\n".join(lines), 2000).get("ids") or []
    chosen = []
    for n in ids:
        if str(n).isdigit() and 1 <= int(n) <= len(cands) and cands[int(n) - 1] not in chosen:
            chosen.append(cands[int(n) - 1])
    chosen = chosen[:MAX_SHABADS]
    if not chosen:
        return empty

    # 4. explain each whole Shabad, with its Rahao as the central idea
    lines = [f"QUESTION: {question}", ""]
    for n, pid in enumerate(chosen, 1):
        p = engine.padas[pid]
        shabad = " ".join(engine.padas[q]["text"] for q in engine.shabad_padas[p["shabad_id"]])
        lines += [f"SHABAD {n} -- from Ang {p['ang']}: {_excerpt(shabad, 1500)}",
                  f"PADA: {p['text']}",
                  f"Steek: {_excerpt(p.get('steek_ss'), 600)}"]
        rahao = engine.rahao_pada(p["shabad_id"])
        if rahao is not None and rahao != pid:
            r = engine.padas[rahao]
            lines.append(f"RAHAO: {r['text']} -- Steek: {_excerpt(r.get('steek_ss'), 300)}")
        lines.append("")
    prompt = "\n".join(lines)
    system = EXPLAIN.format(language=language, language_rule=LANGUAGE_RULES[language])
    # "medium" thinking: at "low", gpt-oss misread Gurbani idioms (ਆਠ ਪਹਰ as "eight
    # times a day") and suggested practices Gurbani rejects.
    # Only an unusable answer is asked for again: re-asking over a stray English word
    # doubled the tokens and ran into the free per-minute limit; _scrub tidies those.
    angs = [engine.padas[pid]["ang"] for pid in chosen]
    ans = _ids_by_position(_chat(writer, wmodel, system, prompt, 4500, "medium"), angs)
    if not _valid(ans, len(chosen)):
        ans = _ids_by_position(_chat(writer, wmodel, system, prompt, 4500, "medium"), angs)
    if not _valid(ans, len(chosen)):
        raise AskError("The explanation came back incomplete. Please ask again.", 502)

    tidy = _scrub if language == "Punjabi" else (lambda t: " ".join(t.split()))
    out, seen = [], set()
    for item in ans["shabads"]:
        pid = chosen[int(item["id"]) - 1]
        if pid in seen:
            continue
        seen.add(pid)
        out.append({"pada_id": pid, "key_line": str(item.get("key_line") or ""),
                    "explanation": tidy(str(item["explanation"]))})
    guidance = [tidy(str(g)) for g in ans.get("guidance") or [] if str(g).strip()]
    return {"language": language, "read_as": read_as, "answer": tidy(str(ans["answer"])),
            "guidance": guidance, "shabads": out}
