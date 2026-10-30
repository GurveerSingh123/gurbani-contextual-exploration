# Frontend — Gurbani Bhavarth Search website

Static website: HTML + **Tailwind CSS** + plain JavaScript modules. No build step is needed to
**run** it — the backend serves this folder. Start the backend (see `../backend/README.md`) and
open <http://localhost:8000>.

## Files

| File | What it does |
|---|---|
| `index.html` | The page: header with live status, hero, search card, results area, "How it works", About |
| `assets/css/app.css` | The compiled stylesheet (generated — do not edit by hand) |
| `assets/js/api.js` | Calls to the backend (`/api/health`, `/api/search`, `/api/explain`) |
| `assets/js/render.js` | Builds the result cards, the pada context card, explanations; escapes all text |
| `assets/js/app.js` | Wires the page: search form, examples, "All Steeks", "Why this match?" |
| `assets/img/logo.svg` | Logo / favicon |
| `src/input.css` | Tailwind source: glass cards, glows, buttons, tags, Gurmukhi line styles |
| `tailwind.config.js` | Theme: night background, kesri and teal glow colours, fonts, animations |
| `package.json` | Tailwind version and the build scripts |

## Changing the styling

Only needed if you edit `src/input.css`, `tailwind.config.js`, or the classes in `index.html` / `assets/js`:

```powershell
cd frontend
npm install          # once; installs Tailwind 3.4.17 into node_modules (not committed)
npm run build:css    # rebuilds assets/css/app.css
npm run watch:css    # or: rebuild automatically while editing
```

Fonts (Plus Jakarta Sans, Noto Sans/Serif Gurmukhi) load from Google Fonts when online. Offline,
Gurmukhi falls back to Windows' **Nirmala UI**, so the site still reads correctly at a demo
without internet.

## Serving it separately (optional)

Any static server works, e.g. VS Code Live Server on port 5500. Then set the API location before
`app.js` loads (`<script>window.API_BASE = "http://localhost:8000"</script>`); the backend already
allows `localhost:5500` in `CORS_ORIGINS`.
