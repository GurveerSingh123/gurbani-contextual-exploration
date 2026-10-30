// Page wiring: status, search form, result interactions.
import { api } from "./api.js";
import * as view from "./render.js";

const $ = (s) => document.querySelector(s);
const state = { health: null, last: null, topN: 5, query: "", mode: "search", views: {} };
const LIMIT = { search: 4000, ask: 300 };

// --- status ----------------------------------------------------------------
async function loadHealth() {
  try {
    const h = await api.health();
    state.health = h;
    $("#status-dot").className = "h-2 w-2 rounded-full bg-emerald-400 shadow-[0_0_10px_rgba(52,211,153,.9)] animate-breathe";
    $("#status-text").textContent = "Model online";
    for (const el of document.querySelectorAll("[data-stat]")) {
      const v = h[el.dataset.stat];
      if (typeof v === "number") el.textContent = v.toLocaleString("en-IN");
    }
    $("#footer-explain").textContent = h.explain
      ? `Explanations by ${h.explain_model}, grounded only in the Steeks.`
      : "Explanations are off: add LLM_API_KEY to .env and restart the server.";
  } catch (e) {
    $("#status-dot").className = "h-2 w-2 rounded-full bg-rose-400";
    $("#status-text").textContent = "Server offline";
    $("#footer-explain").textContent = e.message;
  }
}

// --- search ----------------------------------------------------------------
async function runSearch() {
  const q = input().value.trim();
  if (!q) { input().focus(); return; }
  if (state.mode === "ask") return runAsk(q);
  const out = $("#results");
  state.query = q;
  setBusy(true);
  out.innerHTML = view.loading();
  out.scrollIntoView({ behavior: "smooth", block: "start" });
  try {
    state.last = await api.search(q, state.topN, state.topN);
    out.innerHTML = view.results(state.last);
    loadReasons(state.last);              // not awaited: results show first, reasons fade in
  } catch (e) {
    out.innerHTML = view.error(e.message);
  } finally {
    setBusy(false);
  }
}

// One request for every card's reason + verdict (see backend/app/summaries.py).
// The verdicts also decide which "different words" candidates are shown.
async function loadReasons(data) {
  const ids = [...data.results, ...data.different_candidates].map((r) => r.pada_id);
  if (!ids.length) return;
  const finish = (items, checkFailed) => {
    if (state.last !== data) return;                 // a newer search replaced this one
    view.reasons(items);
    const slot = document.getElementById("diff-section");
    if (slot) slot.innerHTML = view.differentSection(data, items, checkFailed);
  };
  if (!state.health?.explain) { finish({}, true); return; }
  try {
    const res = await api.summaries(data.read_as.padas.map((p) => p.pada_id), ids,
                                    data.read_as.mode === "granth" ? "" : data.query);
    finish(res.items || {}, Boolean(res.error));
  } catch {
    finish({}, true);
  }
}

// --- ask a question -----------------------------------------------------------
// Its own box (#ask-q) and its own results, so a question never goes to the Gurbani
// search (where free text loads the large model on the CPU -- very slow).
async function runAsk(q) {
  const out = $("#results");
  setBusy(true);
  out.innerHTML = view.askLoading();
  out.scrollIntoView({ behavior: "smooth", block: "start" });
  const started = Date.now();
  const timer = setInterval(() => {               // show that it is working, and for how long
    const el = document.getElementById("ask-timer");
    if (el) el.textContent = `${Math.round((Date.now() - started) / 1000)} s`;
  }, 1000);
  try {
    out.innerHTML = view.answer(await api.ask(q));
  } catch (e) {
    out.innerHTML = view.error(e.message);
  } finally {
    clearInterval(timer);
    setBusy(false);
  }
}

const input = () => (state.mode === "ask" ? $("#ask-q") : $("#q"));

// Each tab keeps its own text and its own last results.
function setMode(mode) {
  if (mode === state.mode) return;
  state.views[state.mode] = $("#results").innerHTML;
  state.mode = mode;
  for (const b of document.querySelectorAll("[data-mode-btn]")) {
    const on = b.dataset.modeBtn === mode;
    b.classList.toggle("mode-active", on);
    b.setAttribute("aria-selected", String(on));
  }
  for (const el of document.querySelectorAll("[data-mode]")) el.hidden = el.dataset.mode !== mode;
  $("#results").innerHTML = state.views[mode] || "";
  setBusy(false);
  input().focus();
}

function updateCount() {
  $("#count").textContent = `${$("#q").value.length} / ${LIMIT.search}`;
  $("#ask-count").textContent = `${$("#ask-q").value.length} / ${LIMIT.ask}`;
}

function setTopN(n) {
  state.topN = n;
  for (const b of document.querySelectorAll("[data-topn]")) {
    const on = Number(b.dataset.topn) === n;
    b.classList.toggle("topn-active", on);
    b.setAttribute("aria-checked", String(on));
  }
  if (state.last && state.query) runSearch();      // re-run the current search at the new size
}

function setBusy(busy) {
  const ask = state.mode === "ask";
  $("#go").disabled = busy;
  $("#go-label").textContent = busy ? (ask ? "Reflecting…" : "Searching…") : (ask ? "Ask" : "Search by meaning");
}

async function explain(button) {
  const key = button.dataset.why;
  const box = document.getElementById(`why-${key}`);
  if (!state.health?.explain) {
    box.innerHTML = view.notice("Explanations are off: add LLM_API_KEY to .env and restart the server.");
    return;
  }
  button.disabled = true;
  box.innerHTML = view.notice("Reading the Steeks of both padas…");
  try {
    const ids = state.last.read_as.padas.map((p) => p.pada_id);
    box.innerHTML = view.explanation(await api.explain(ids, Number(button.dataset.pada)));
  } catch (e) {
    box.innerHTML = view.notice(e.message);
  } finally {
    button.disabled = false;
  }
}

// --- events ----------------------------------------------------------------
$("#search-form").addEventListener("submit", (e) => { e.preventDefault(); runSearch(); });
for (const box of [$("#q"), $("#ask-q")]) {
  box.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); runSearch(); }
  });
  box.addEventListener("input", updateCount);
}
$("#clear").addEventListener("click", () => {
  input().value = "";
  if (state.mode === "search") { state.last = null; state.query = ""; }
  updateCount();
  $("#results").innerHTML = "";
  input().focus();
});
for (const b of document.querySelectorAll("[data-example]")) {
  b.addEventListener("click", () => {
    input().value = b.dataset.example;            // a whole pada, one line per row -- or a question
    updateCount();
    runSearch();
  });
}
for (const b of document.querySelectorAll("[data-mode-btn]")) {
  b.addEventListener("click", () => setMode(b.dataset.modeBtn));
}
for (const b of document.querySelectorAll("[data-topn]")) {
  b.addEventListener("click", () => setTopN(Number(b.dataset.topn)));
}
// Result buttons are created later, so listen on the container.
$("#results").addEventListener("click", (e) => {
  const toggle = e.target.closest("[data-toggle]");
  if (toggle) {
    const box = document.getElementById(`steeks-${toggle.dataset.toggle}`);
    box.hidden = !box.hidden;
    toggle.textContent = box.hidden ? "All Steeks" : "Hide Steeks";
    return;
  }
  const more = e.target.closest("[data-more-btn]");
  if (more) {                                     // a long Shabad: reveal the rest of it
    for (const l of document.querySelectorAll(`[data-more="${more.dataset.moreBtn}"]`)) l.hidden = false;
    more.remove();
    return;
  }
  const why = e.target.closest("[data-why]");
  if (why) explain(why);
});

loadHealth();
