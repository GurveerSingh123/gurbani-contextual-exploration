// Calls to the backend API. When the frontend is served by the backend (the normal
// setup), the API is on the same origin; set window.API_BASE to point elsewhere.
const BASE = window.API_BASE || "";

async function getJSON(path) {
  let res;
  try {
    res = await fetch(BASE + path);
  } catch {
    throw new Error("The server is not reachable. Start it with: python -m uvicorn backend.app.main:app --port 8000");
  }
  const body = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(body.detail || `Request failed (${res.status})`);
  return body;
}

export const api = {
  health: () => getJSON("/api/health"),
  search: (q, k = 5, d = 5) => getJSON(`/api/search?q=${encodeURIComponent(q)}&k=${k}&d=${d}`),
  summaries: (queryPadaIds, padaIds, q = "") =>
    getJSON(`/api/summaries?query=${encodeURIComponent(queryPadaIds.join(","))}` +
            `&padas=${encodeURIComponent(padaIds.join(","))}&q=${encodeURIComponent(q)}`),
  ask: (q) => getJSON(`/api/ask?q=${encodeURIComponent(q)}`),
  explain: (queryPadaIds, matchPadaId) =>
    getJSON(`/api/explain?query=${encodeURIComponent(queryPadaIds.join(","))}&match=${encodeURIComponent(matchPadaId)}`),
};
