/** Tailwind theme for Gurbani Bhavarth Search: deep night background, kesri (saffron)
 *  and teal glows. Build with `npm run build:css` (see README.md). */
module.exports = {
  content: ["./index.html", "./assets/js/**/*.js"],
  // Built from a template string in render.js (`verdict-${...}`), so the scanner cannot see them.
  safelist: ["verdict-same", "verdict-related", "verdict-different"],
  theme: {
    extend: {
      colors: {
        night: { 950: "#05070d", 900: "#080c16", 850: "#0b1120", 800: "#0f1628", 700: "#18213a", 600: "#243052" },
        kesri: { 200: "#ffe2bd", 300: "#ffc98a", 400: "#ffad5c", 500: "#f6923a", 600: "#e0761c", 700: "#b85c12" },
        teal: { 300: "#7ff0e0", 400: "#3ee0cc", 500: "#16c3b0", 600: "#0f9e90" },
        mist: { 100: "#eef2fb", 200: "#d5ddef", 300: "#aab6d3", 400: "#7f8cad", 500: "#5c6888" },
      },
      fontFamily: {
        display: ['"Plus Jakarta Sans"', '"Segoe UI"', "system-ui", "sans-serif"],
        sans: ['"Plus Jakarta Sans"', '"Segoe UI"', "system-ui", "sans-serif"],
        gurmukhi: ['"Noto Sans Gurmukhi"', '"Nirmala UI"', '"Raavi"', "sans-serif"],
        gurbani: ['"Noto Serif Gurmukhi"', '"Nirmala UI"', '"Raavi"', "serif"],
        mono: ['"JetBrains Mono"', "Consolas", "monospace"],
      },
      boxShadow: {
        "glow-kesri": "0 0 0 1px rgba(246,146,58,.35), 0 0 10px -3px rgba(246,146,58,.45)",
        "glow-teal": "0 0 0 1px rgba(22,195,176,.35), 0 0 10px -3px rgba(22,195,176,.45)",
        "glow-soft": "0 10px 40px -12px rgba(0,0,0,.7), inset 0 1px 0 rgba(255,255,255,.05)",
      },
      keyframes: {
        shimmer: { "100%": { transform: "translateX(100%)" } },
        breathe: { "0%,100%": { opacity: ".55" }, "50%": { opacity: "1" } },
        rise: { "0%": { opacity: "0", transform: "translateY(8px)" }, "100%": { opacity: "1", transform: "none" } },
      },
      animation: {
        shimmer: "shimmer 1.6s infinite",
        breathe: "breathe 2.4s ease-in-out infinite",
        rise: "rise .45s ease-out both",
      },
    },
  },
  plugins: [],
};
