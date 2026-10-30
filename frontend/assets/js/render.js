// HTML builders for the results. Every piece of text from the server goes through esc().
export const esc = (s) =>
  String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

// The server sends each line's Steeks as a list in a fixed order; the footer says
// which commentary each number is. Names are deliberately not shown on the cards.
const STEEK_LABELS = ["Steek 1", "Steek 2", "Steek 3"];

const icon = {
  book: '<svg class="h-4 w-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20M4 19.5A2.5 2.5 0 0 0 6.5 22H20V2H6.5A2.5 2.5 0 0 0 4 4.5v15z"/></svg>',
  spark: '<svg class="h-4 w-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 3l1.9 5.8L20 10l-6.1 1.2L12 17l-1.9-5.8L4 10l6.1-1.2z"/></svg>',
  pin: '<svg class="h-3.5 w-3.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 21s-7-6.5-7-12a7 7 0 0 1 14 0c0 5.5-7 12-7 12z"/><circle cx="12" cy="9" r="2.5"/></svg>',
  music: '<svg class="h-3.5 w-3.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M9 18V5l12-2v13"/><circle cx="6" cy="18" r="3"/><circle cx="18" cy="16" r="3"/></svg>',
};

export function loading() {
  const card = `
    <div class="glass p-6">
      <div class="skeleton h-5 w-4/5"></div><div class="skeleton mt-3 h-5 w-3/5"></div>
      <div class="mt-5 flex gap-3"><div class="skeleton h-3 w-20"></div><div class="skeleton h-3 w-24"></div><div class="skeleton h-3 w-16"></div></div>
      <div class="skeleton mt-4 h-3 w-full"></div><div class="skeleton mt-2 h-3 w-11/12"></div>
    </div>`;
  return `<div class="space-y-4">${card.repeat(3)}</div>`;
}

export function error(message) {
  return `
    <div class="glass border-rose-500/30 p-5 text-sm text-rose-200" style="box-shadow:0 0 0 1px rgba(244,63,94,.25),0 0 32px -8px rgba(244,63,94,.4)">
      <div class="font-semibold text-rose-300">Search failed</div>
      <p class="mt-1 text-rose-200/90">${esc(message)}</p>
    </div>`;
}

function context(readAs) {
  if (readAs.mode !== "granth") {
    return `
      <div class="glass p-5 text-sm text-mist-300">
        <div class="flex items-center gap-2 font-semibold text-white">${icon.spark} Read as free text</div>
        <p class="mt-1 text-mist-400">This text was not found in Sri Guru Granth Sahib Ji, so the meaning model read it directly.</p>
      </div>`;
  }
  const typed = new Set(readAs.lines);
  const padas = readAs.padas.map((p) => `
      <div class="space-y-0.5">${p.lines.map((l) => `<p class="pline ${typed.has(l) ? "pline-best" : ""}">${esc(l)}</p>`).join("")}</div>
      <div class="mt-2 flex flex-wrap gap-3 text-xs text-mist-500">
        <span class="inline-flex items-center gap-1">${icon.pin} Ang ${p.ang}</span>
        ${p.raag ? `<span class="inline-flex items-center gap-1">${icon.music} ${esc(p.raag)}</span>` : ""}
      </div>`).join('<div class="my-4 border-t border-white/5"></div>');
  const title = readAs.padas.length > 1
    ? `Your lines come from ${readAs.padas.length} padas, searched together`
    : "Your line, read with its pada";
  const titlePa = readAs.padas.length > 1 ? "ਤੁਹਾਡੀਆਂ ਪੰਕਤੀਆਂ, ਪਦਿਆਂ ਸਮੇਤ" : "ਤੁਹਾਡੀ ਪੰਕਤੀ, ਉਸ ਦੇ ਪਦੇ ਸਮੇਤ";
  const note = [readAs.note, readAs.missed.length ? `not found: ${readAs.missed.join(", ")}` : ""].filter(Boolean).join(" · ");
  return `
    <div class="glass p-5 sm:p-6">
      <div class="mb-3 flex flex-wrap items-center gap-x-3 gap-y-1">
        <span class="flex items-center gap-2 text-xs font-semibold uppercase tracking-[.14em] text-kesri-300">${icon.book} ${esc(title)}</span>
        <span class="font-gurmukhi text-sm text-mist-400">${titlePa}</span>
      </div>
      ${padas}
      ${note ? `<p class="mt-3 text-xs text-mist-500">${esc(note)}</p>` : ""}
    </div>`;
}

function foundTag(r) {
  if (r.shared_words === 0) return '<span class="tag-nowords">no shared words</span>';
  if (r.found_by.length === 2) return '<span class="tag-both">meaning + words</span>';
  if (r.found_by[0] === "meaning") return '<span class="tag-meaning">meaning</span>';
  if (r.found_by[0] === "words") return '<span class="tag-words">words</span>';
  return "";
}

// opts.item: {reason, verdict} already known (verified "different words" cards);
// opts.unverified: the LLM check could not run, so no reason and a "not verified" tag.
function card(r, i, key, variant, canExplain, opts = {}) {
  const teal = variant === "teal";
  const reasonBox = opts.unverified ? "" : `
          <div class="reason ${opts.item ? "reason-ready" : ""}" id="reason-${key}" data-pada="${r.pada_id}">
            <span class="reason-icon">${icon.spark}</span>
            <span class="reason-text">${opts.item ? esc(opts.item.reason) : '<span class="skeleton inline-block h-3 w-64 max-w-full align-middle"></span>'}</span>
          </div>`;
  const verdictSlot = opts.unverified
    ? '<span class="tag bg-white/5 text-mist-400 ring-1 ring-white/10">not verified</span>'
    : `<span class="verdict-slot" data-pada="${r.pada_id}">${opts.item ? verdictBadge(opts.item.verdict) : ""}</span>`;
  const best = r.lines.find((l) => l.best) || r.lines[0];
  const lines = r.lines.map((l) => `<p class="pline ${l.best ? `pline-best ${teal ? "pline-best-teal" : ""}` : ""}">${esc(l.text)}</p>`).join("");
  const allSteeks = r.lines.map((l) => `
      <div class="rounded-xl border border-white/5 bg-night-950/40 p-3">
        <p class="font-gurbani text-base font-semibold text-white">${esc(l.text)}</p>
        ${l.steeks.map((s, n) => s ? `<p class="steek mt-1.5"><span class="steek-label">${STEEK_LABELS[n]}</span>${esc(s)}</p>` : "").join("")}
      </div>`).join("");
  return `
    <li class="glass glass-hover ${teal ? "glass-teal" : ""} animate-rise p-5 sm:p-6" style="animation-delay:${i * 45}ms">
      <div class="flex gap-4">
        <div class="flex h-9 w-9 shrink-0 items-center justify-center rounded-xl font-mono text-sm font-semibold ${teal ? "bg-teal-500/15 text-teal-300 shadow-glow-teal" : "bg-kesri-500/15 text-kesri-300 shadow-glow-kesri"}">${i + 1}</div>
        <div class="min-w-0 flex-1">
          <div class="space-y-0.5">${lines}</div>
          <div class="mt-3 flex flex-wrap items-center gap-x-4 gap-y-2 text-xs text-mist-400">
            <span class="inline-flex items-center gap-1">${icon.pin} Ang ${r.ang}${r.ang_gap !== null ? ` <span class="text-mist-500">(${r.ang_gap} away)</span>` : ""}</span>
            <span class="inline-flex items-center gap-1">${icon.music} ${esc(r.raag || "Raag unknown")}</span>
            <span>Shabad ${r.shabad_id}</span>
            <span class="font-mono text-mist-300">meaning ${r.similarity.toFixed(3)}</span>
            ${foundTag(r)}
            ${verdictSlot}
          </div>
          ${best.steeks[0] ? `<p class="steek mt-3"><span class="steek-label">Steek</span>${esc(best.steeks[0])}</p>` : ""}
          ${reasonBox}
          <div class="mt-4 flex flex-wrap gap-2">
            <button type="button" class="btn-ghost btn-small" data-toggle="${key}">All Steeks</button>
            ${canExplain ? `<button type="button" class="btn-ghost btn-small" data-why="${key}" data-pada="${r.pada_id}">${icon.spark} Why this match?</button>` : ""}
          </div>
          <div id="steeks-${key}" class="mt-3 space-y-2" hidden>${allSteeks}</div>
          <div id="why-${key}"></div>
        </div>
      </div>
    </li>`;
}

function verdictBadge(v) {
  return ["same", "related", "different"].includes(v) ? `<span class="verdict-${v}">${v}</span>` : "";
}

export function results(data) {
  const canExplain = data.read_as.padas.length > 0;
  const main = data.results.map((r, i) => card(r, i, `m${i}`, "kesri", canExplain)).join("");
  // The "different words" section is filled later by differentSection(), once the
  // LLM check has judged the candidates. Until then only a quiet status line shows.
  const checking = data.different_candidates.length
    ? `<p class="mt-12 flex items-center gap-2 text-sm text-mist-500"><span class="h-1.5 w-1.5 rounded-full bg-teal-400 animate-breathe"></span>Checking matches that use different words…</p>`
    : "";
  return `
    ${context(data.read_as)}
    <div class="mt-10 flex flex-wrap items-end justify-between gap-2">
      <div>
        <h2 class="section-title">Closest in meaning</h2>
        <p class="section-gurmukhi">ਸਭ ਤੋਂ ਨੇੜਲਾ ਭਾਵਾਰਥ</p>
        <p class="section-sub">Ranked by the meaning model and keyword search together. The glowing line in each pada is the one closest to yours.</p>
      </div>
      <span class="font-mono text-xs text-mist-500">${data.results.length} padas · ${data.seconds}s</span>
    </div>
    <ol class="mt-5 space-y-4">${main}</ol>
    <div id="diff-section">${checking}</div>`;
}

// "Same meaning, different words": the meaning model proposed the candidates;
// show only those the LLM judged same or related (up to the chosen top N). If none
// pass, return "" -- the section does not appear at all. If the check could not run,
// fall back to the model's top N, tagged "not verified".
export function differentSection(data, items, checkFailed) {
  const canExplain = data.read_as.padas.length > 0;
  const limit = data.different_limit;
  let cards = "";
  let sub = "";
  if (checkFailed) {
    cards = data.different_candidates.slice(0, limit)
      .map((r, i) => card(r, i, `d${i}`, "teal", canExplain, { unverified: true })).join("");
    sub = "Proposed by the meaning model; the check could not run, so these are not verified.";
  } else {
    const kept = data.different_candidates
      .filter((r) => ["same", "related"].includes(items[r.pada_id]?.verdict))
      .slice(0, limit);
    cards = kept.map((r, i) => card(r, i, `d${i}`, "teal", canExplain, { item: items[r.pada_id] })).join("");
    sub = "Proposed by the meaning model and checked against the Steeks. These padas share no word with your query, so keyword search could never find them.";
  }
  if (!cards) return "";
  return `
    <div class="mt-14">
      <h2 class="section-title"><span class="text-teal-300">Same meaning,</span> different words</h2>
      <p class="section-gurmukhi">ਇਕੋ ਭਾਵ, ਵੱਖਰੇ ਸ਼ਬਦ</p>
      <p class="section-sub">${sub}</p>
    </div>
    <ol class="mt-5 space-y-4">${cards}</ol>`;
}

export function explanation(e) {
  if (e.error) return `<div class="mt-4 rounded-xl border border-white/10 bg-night-950/50 p-4 text-sm text-mist-300">${esc(e.error)}</div>`;
  const verdict = ["same", "related", "different"].includes(e.verdict) ? e.verdict : "related";
  return `
    <div class="mt-4 space-y-2 rounded-xl border border-white/10 bg-night-950/50 p-4 animate-rise">
      <span class="verdict-${verdict}">${esc(e.verdict)}</span>
      <p class="font-gurmukhi text-base leading-relaxed text-white">${esc(e.shared_bhavarth_pa)}</p>
      <p class="text-sm text-mist-200">${esc(e.shared_bhavarth_en)}</p>
      <p class="text-sm leading-relaxed text-mist-400">${esc(e.explanation_en)}</p>
      ${e.difference ? `<p class="text-sm text-mist-400"><b class="text-mist-200">Differs:</b> ${esc(e.difference)}</p>` : ""}
    </div>`;
}

// Fill (or remove) the main cards' reason lines and verdict badges once
// /api/summaries answers. items: {pada_id: {reason, verdict}}.
export function reasons(items) {
  for (const el of document.querySelectorAll(".reason:not(.reason-ready)")) {
    const item = items[el.dataset.pada];
    if (item) {
      el.querySelector(".reason-text").textContent = item.reason;
      el.classList.add("reason-ready");
    } else {
      el.remove();                     // no reason: show nothing rather than a stuck placeholder
    }
  }
  for (const slot of document.querySelectorAll(".verdict-slot:empty")) {
    const item = items[slot.dataset.pada];
    if (item) slot.innerHTML = verdictBadge(item.verdict);
  }
}

// --- "Ask a question" ------------------------------------------------------------
export function askLoading() {
  return `
    <div class="glass p-6 animate-rise">
      <div class="flex flex-wrap items-center gap-3 text-sm text-mist-300">
        <span class="h-2 w-2 rounded-full bg-kesri-400 animate-breathe"></span>
        Finding the Shabads that answer your question, and reflecting on them…
        <span id="ask-timer" class="ml-auto font-mono text-xs text-mist-500">0 s</span>
      </div>
      <div class="skeleton mt-5 h-4 w-11/12"></div><div class="skeleton mt-2 h-4 w-4/5"></div><div class="skeleton mt-2 h-4 w-3/5"></div>
    </div>
    <div class="mt-4">${loading()}</div>`;
}

// Headings follow the language the question was asked in (so does the answer).
const WORDS = {
  English: { teaches: "What Gurbani teaches", from: "From Sri Guru Granth Sahib Ji",
             fromSub: (n) => `${n} Shabads that answer your question. The glowing line answers it most directly.`,
             explanation: "Explanation", guidance: "What you can do", steeks: "Steeks of this pada" },
  Punjabi: { teaches: "ਗੁਰਬਾਣੀ ਦਾ ਉਪਦੇਸ਼", from: "ਸ੍ਰੀ ਗੁਰੂ ਗ੍ਰੰਥ ਸਾਹਿਬ ਜੀ ਵਿੱਚੋਂ",
             fromSub: (n) => `ਤੁਹਾਡੇ ਸਵਾਲ ਦਾ ਉੱਤਰ ਦੇਣ ਵਾਲੇ ${n} ਸ਼ਬਦ। ਚਮਕਦੀ ਪੰਕਤੀ ਸਭ ਤੋਂ ਸਿੱਧਾ ਉੱਤਰ ਦਿੰਦੀ ਹੈ।`,
             explanation: "ਵਿਆਖਿਆ", guidance: "ਤੁਸੀਂ ਕੀ ਕਰ ਸਕਦੇ ਹੋ", steeks: "ਇਸ ਪਦੇ ਦੇ ਸਟੀਕ" },
};

const SHABAD_PREVIEW = 14;     // longer Shabads open with a "show the whole Shabad" button

function answerShabad(s, i, w, pa) {
  const key = `a${i}`;
  const text = pa ? "font-gurmukhi text-[1.05rem] leading-[1.95]" : "text-[.98rem] leading-relaxed";
  // The whole Shabad; inside the pada that answers, the key line glows.
  const rows = s.shabad.flatMap((part) => part.this
    ? s.lines.map((l) => ({ text: l.text, cls: l.best ? "pline pline-best" : "pline text-kesri-200" }))
    : part.lines.map((l) => ({ text: l, cls: "pline text-mist-300" })));
  const long = rows.length > SHABAD_PREVIEW;
  const lines = rows.map((r, n) => `<p class="${r.cls}" ${long && n >= SHABAD_PREVIEW ? `data-more="${key}" hidden` : ""}>${esc(r.text)}</p>`).join("");
  const allSteeks = s.lines.map((l) => `
      <div class="rounded-xl border border-white/5 bg-night-950/40 p-3">
        <p class="font-gurbani text-base font-semibold text-white">${esc(l.text)}</p>
        ${l.steeks.map((st, n) => st ? `<p class="steek mt-1.5"><span class="steek-label">${STEEK_LABELS[n]}</span>${esc(st)}</p>` : "").join("")}
      </div>`).join("");
  return `
    <li class="glass glass-hover animate-rise p-5 sm:p-7" style="animation-delay:${i * 80}ms">
      <div class="flex gap-4">
        <div class="flex h-9 w-9 shrink-0 items-center justify-center rounded-xl bg-kesri-500/15 font-mono text-sm font-semibold text-kesri-300 shadow-glow-kesri">${i + 1}</div>
        <div class="min-w-0 flex-1">
          <div class="flex flex-wrap items-center gap-x-4 gap-y-2 text-xs text-mist-400">
            <span class="inline-flex items-center gap-1">${icon.pin} Ang ${s.ang}</span>
            <span class="inline-flex items-center gap-1">${icon.music} ${esc(s.raag || "Raag unknown")}</span>
            <span>Shabad ${s.shabad_id}</span>
          </div>
          <div class="mt-3 space-y-0.5">${lines}</div>
          ${long ? `<button type="button" class="btn-ghost btn-small mt-3" data-more-btn="${key}">${icon.book} Show the whole Shabad (${rows.length} lines)</button>` : ""}
          <div class="answer-block mt-6">
            <h4 class="answer-block-title">${w.explanation}</h4>
            <p class="${text} text-mist-100">${esc(s.explanation)}</p>
          </div>
          <div class="mt-5 flex flex-wrap gap-2">
            <button type="button" class="btn-ghost btn-small" data-toggle="${key}">All Steeks</button>
          </div>
          <div id="steeks-${key}" class="mt-3 space-y-2" hidden>${allSteeks}</div>
        </div>
      </div>
    </li>`;
}

export function answer(data) {
  const pa = data.language === "Punjabi";
  const w = WORDS[pa ? "Punjabi" : "English"];
  const text = pa ? "font-gurmukhi text-[1.12rem] leading-[1.95]" : "text-[1.05rem] leading-relaxed";
  const question = `
    <div class="glass p-5 sm:p-6">
      <div class="flex items-center gap-2 text-xs font-semibold uppercase tracking-[.14em] text-kesri-300">${icon.spark} Your question</div>
      <p class="mt-2 text-lg text-white ${pa ? "font-gurmukhi" : ""}">${esc(data.question)}</p>
    </div>`;
  if (!data.found) {
    return `${question}
      <div class="glass mt-6 p-6 text-sm text-mist-300">
        <p class="font-semibold text-white">No Shabad that clearly answers this was found.</p>
        <p class="mt-1 font-gurmukhi text-mist-400">ਇਸ ਸਵਾਲ ਦਾ ਸਪਸ਼ਟ ਉੱਤਰ ਦੇਣ ਵਾਲਾ ਸ਼ਬਦ ਨਹੀਂ ਮਿਲਿਆ।</p>
        <p class="mt-3 text-mist-400">Try asking about a spiritual theme: ego, Naam, anger, fear, death, humility, grace, peace of mind.</p>
      </div>`;
  }
  const guidance = data.guidance.length ? `
      <div class="glass glass-teal mt-6 p-5 sm:p-7 animate-rise">
        <h3 class="flex items-center gap-2 font-display text-lg font-semibold text-teal-200 ${pa ? "font-gurmukhi" : ""}">${icon.spark} ${w.guidance}</h3>
        <ol class="mt-4 space-y-3">
          ${data.guidance.map((g, n) => `
            <li class="flex gap-3">
              <span class="mt-0.5 flex h-6 w-6 shrink-0 items-center justify-center rounded-lg bg-teal-500/15 font-mono text-xs font-semibold text-teal-300">${n + 1}</span>
              <p class="${text} text-mist-100">${esc(g)}</p>
            </li>`).join("")}
        </ol>
      </div>` : "";
  return `
    ${question}
    <div class="mt-10">
      <h2 class="section-title ${pa ? "font-gurmukhi" : ""}">${w.teaches}</h2>
    </div>
    <div class="glass answer-glow mt-5 p-5 sm:p-7 animate-rise">
      <p class="${text} text-white">${esc(data.answer)}</p>
    </div>
    ${guidance}
    <div class="mt-12">
      <h3 class="section-title text-xl ${pa ? "font-gurmukhi" : ""}">${w.from}</h3>
      <p class="section-sub ${pa ? "font-gurmukhi" : ""}">${w.fromSub(data.shabads.length)}</p>
    </div>
    <ol class="mt-5 space-y-5">${data.shabads.map((s, i) => answerShabad(s, i, w, pa)).join("")}</ol>
    <p class="mt-6 text-xs leading-relaxed text-mist-500">
      Found through the Steeks; explained by ${esc(data.model)} from the Steeks and the whole Shabad. This is a study aid:
      for understanding, read the Bani itself and the full Steeks. <span class="font-mono">${data.seconds}s</span>
    </p>`;
}

export function notice(text) {
  return `<div class="mt-4 rounded-xl border border-white/10 bg-night-950/50 p-4 text-sm text-mist-400">${esc(text)}</div>`;
}
