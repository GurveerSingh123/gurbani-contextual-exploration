// Builds the midterm panel presentation as a structured .pptx.
const path = require("path");
const pptxgen = require("pptxgenjs");
const { applyTheme } = require("C:/Users/Lenovo/AppData/Roaming/Claude/local-agent-mode-sessions/skills-plugin/cb2ffdcc-bafe-4021-8de8-da4e2a2af135/1dd1f198-c884-4549-8dae-b3b23d380dd1/skills/pptx/scripts/apply_theme.js");

const REPORT = "E:/Major_Project_Gurveer/mid term report";
const OUT = path.join(REPORT, "Midterm_presentation.pptx");
const img = (f) => path.join(REPORT, f);

// ink navy + kesri (saffron) accent; good/bad semantic pair
const HEX = {
  ink: "16213F", ink2: "4A5573", paper: "F3F5F9", rule: "D9DEEA", white: "FFFFFF",
  kesri: "D9822B", kesriSoft: "FBEEDD", good: "2F7A4B", goodSoft: "E3F1E7",
  bad: "B4532A", badSoft: "F8E6DD", grey: "AEB6CA",
};
const THEME = {
  name: "Gurbani Kesri",
  headFontFace: "Cambria",
  bodyFontFace: "Calibri",
  colors: {
    dk1: HEX.ink, lt1: HEX.white, dk2: HEX.ink2, lt2: HEX.paper,
    accent1: HEX.kesri, accent2: HEX.ink, accent3: HEX.good, accent4: HEX.bad,
    accent5: HEX.grey, accent6: HEX.rule, hlink: HEX.kesri, folHlink: HEX.ink2,
  },
};

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.333 x 7.5
pres.title = "Semantic Retrieval and Contextual Exploration System";
pres.author = "Balkaran Singh, Gurpreet Kaur, Gurveer Singh";
pres.subject = "B.Tech major project midterm presentation";
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
const C = pres.SchemeColor;
// scheme aliases
const INK = C.text1, INK2 = C.text2, PAPER = C.background2, WHITE = C.background1;
const KESRI = C.accent1, GOOD = C.accent3, BAD = C.accent4, GREY = C.accent5, RULE = C.accent6;

const GUR = "Nirmala UI";
const MONO = "Consolas";
const W = 13.333, MX = 0.6, CW = W - 2 * MX;

// ---------- layouts ----------
pres.defineSlideMaster({
  title: "CONTENT",
  background: { color: WHITE },
  objects: [
    { placeholder: { options: { name: "eyebrow", type: "body", x: MX, y: 0.42, w: CW, h: 0.32, fontSize: 12, bold: true, color: KESRI, charSpacing: 2, margin: 0, valign: "middle" }, text: "" } },
    { placeholder: { options: { name: "title", type: "title", x: MX, y: 0.78, w: CW, h: 1.0, fontFace: "Cambria", fontSize: 30, bold: true, color: INK, margin: 0, valign: "top" }, text: "" } },
    { text: { text: "Semantic Retrieval and Contextual Exploration System  ·  GNDEC Ludhiana", options: { x: MX, y: 7.05, w: 8, h: 0.3, fontSize: 10, color: INK2, margin: 0 } } },
  ],
  slideNumber: { x: 12.1, y: 7.05, w: 0.63, h: 0.3, fontSize: 10, color: INK2, align: "right", margin: 0 },
});
pres.defineSlideMaster({
  title: "DARK",
  background: { color: INK },
  objects: [],
});

// ---------- helpers ----------
const g = (text, o = {}) => ({ text, options: Object.assign({ fontFace: GUR }, o) });
function content(section, eyebrow, title, notes) {
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: section });
  s.addText(eyebrow.toUpperCase(), { placeholder: "eyebrow" });
  s.addText(title, { placeholder: "title" });
  if (notes) s.addNotes(notes);
  return s;
}
function txt(s, text, o) {
  s.addText(text, Object.assign({ isTextBox: true, margin: 0, fontSize: 16, color: INK, valign: "top" }, o));
}
function bullets(s, items, o) {
  const runs = [];
  items.forEach((it, i) => {
    const parts = Array.isArray(it) ? it : [{ text: it }];
    parts.forEach((p, j) => {
      const opt = Object.assign({}, p.options || {});
      if (j === 0) { opt.bullet = { indent: 16 }; opt.paraSpaceAfter = o.gap || 10; }
      if (j === parts.length - 1 && i < items.length - 1) opt.breakLine = true;
      runs.push({ text: p.text, options: opt });
    });
  });
  s.addText(runs, Object.assign({ isTextBox: true, margin: 0, fontSize: 16, color: INK, valign: "top" }, o));
}
function card(s, x, y, w, h, fill, name) {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.08, fill: { color: fill || PAPER }, line: { type: "none" }, objectName: name });
}
function badge(s, x, y, label, d) {
  d = d || 0.55;
  s.addShape(pres.shapes.OVAL, { x, y, w: d, h: d, fill: { color: KESRI }, line: { type: "none" } });
  s.addText(label, { isTextBox: true, x, y, w: d, h: d, margin: 0, align: "center", valign: "middle", fontFace: "Cambria", fontSize: d > 0.5 ? 20 : 14, bold: true, color: WHITE });
}
function tag(s, x, y, w, label, good) {
  s.addText(label.toUpperCase(), { isTextBox: true, x, y, w, h: 0.28, margin: 0, align: "center", valign: "middle", fontSize: 10, bold: true, charSpacing: 1, color: good ? GOOD : BAD, fill: { color: good ? HEX.goodSoft : HEX.badSoft } });
}
function stat(s, x, y, w, big, label, color, bigSize) {
  txt(s, big, { x, y, w, h: 0.75, fontFace: "Cambria", fontSize: bigSize || 36, bold: true, color: color || INK, valign: "bottom" });
  txt(s, label, { x, y: y + 0.85, w, h: 1.1, fontSize: 14, color: INK2 });
}
const th = (t, o = {}) => ({ text: t, options: Object.assign({ bold: true, fontSize: 11, color: INK2, charSpacing: 1, fill: { color: PAPER } }, o) });

// =====================================================================
// 1. Title
pres.addSection({ title: "Opening" });
{
  const s = pres.addSlide({ masterName: "DARK", sectionTitle: "Opening" });
  s.addText("ਜਤ ਕਤ ਦੇਖਉ ਤਤ ਰਹਿਆ ਸਮਾਇ", { isTextBox: true, x: 2.4, y: 0.25, w: 10.6, h: 1.3, margin: 0, align: "right", fontFace: GUR, fontSize: 58, color: WHITE, transparency: 92, objectName: "watermark verse" });
  txt(s, "B.TECH MAJOR PROJECT  ·  MIDTERM EVALUATION", { x: 0.8, y: 1.15, w: 9, h: 0.35, fontSize: 13, bold: true, charSpacing: 2, color: KESRI });
  txt(s, "Semantic Retrieval and Contextual Exploration System", { x: 0.8, y: 1.6, w: 9.4, h: 1.9, fontFace: "Cambria", fontSize: 44, bold: true, color: WHITE });
  txt(s, "Finding Pankatis of Sri Guru Granth Sahib that carry the same Bhavarth, even when they share no words", { x: 0.8, y: 3.55, w: 8.8, h: 0.8, fontSize: 19, color: "C9D1E6" });
  s.addTable([
    [{ text: "Balkaran Singh" }, { text: "2302494", options: { fontFace: MONO, color: "9AA6C6" } }],
    [{ text: "Gurpreet Kaur" }, { text: "2302530", options: { fontFace: MONO, color: "9AA6C6" } }],
    [{ text: "Gurveer Singh" }, { text: "2302533", options: { fontFace: MONO, color: "9AA6C6" } }],
  ], { x: 0.8, y: 4.75, w: 3.6, colW: [2.2, 1.4], fontSize: 16, color: "E8ECF6", margin: [2, 0, 2, 0], border: { type: "none" }, rowH: 0.36 });
  s.addText([{ text: "Under the guidance of  ", options: { color: "C9D1E6" } }, { text: "Dr. Parminder Singh", options: { bold: true, color: WHITE } }], { isTextBox: true, x: 0.8, y: 5.95, w: 6, h: 0.4, margin: 0, fontSize: 16 });
  s.addShape(pres.shapes.OVAL, { x: 11.05, y: 4.6, w: 1.6, h: 1.6, fill: { color: WHITE }, line: { type: "none" } });
  s.addImage({ path: img("logo-70-Edited.png"), x: 11.15, y: 4.7, w: 1.4, h: 1.4, altText: "GNDEC logo" });
  txt(s, "Department of Computer Science and Engineering\nGuru Nanak Dev Engineering College, Ludhiana\nOctober 2026", { x: 6.6, y: 4.95, w: 4.25, h: 1.0, fontSize: 13, color: "C9D1E6", align: "right" });
  s.addNotes("Introduce the team and guide. One sentence on the project: given one line of Gurbani, find other lines with the same meaning, even when no words are shared.");
}

// 2. Introduction
pres.addSection({ title: "Introduction" });
{
  const s = content("Introduction", "Introduction", "Two Pankatis, seven hundred Angs apart, teach the same thing in different words",
    "Read both lines. Point out that they share no word, yet both teach giving up envy. This is the core task: the system must connect lines like these.");
  const cards = [["ANG 189", "ਵਿਸਰੀ ਤਿਸੈ ਪਰਾਈ ਤਾਤਾ", "Envy of others is forgotten"], ["ANG 889", "ਛਾਡਿ ਵਿਡਾਣੀ ਤਾਤਿ ਮੂੜੇ", "Fool, give up envy of others"]];
  cards.forEach((c, i) => {
    const x = MX + i * 6.2;
    card(s, x, 2.05, 5.93, 2.35);
    txt(s, c[0], { x: x + 0.4, y: 2.3, w: 5, h: 0.3, fontFace: MONO, fontSize: 13, color: INK2 });
    txt(s, c[1], { x: x + 0.4, y: 2.7, w: 5.2, h: 0.9, fontFace: GUR, fontSize: 32, bold: true });
    txt(s, c[2], { x: x + 0.4, y: 3.7, w: 5.2, h: 0.4, fontSize: 16, italic: true, color: INK2 });
  });
  txt(s, "0 shared words", { x: MX, y: 4.85, w: 5.9, h: 0.6, fontFace: "Cambria", fontSize: 28, bold: true, color: BAD });
  txt(s, "Keyword search cannot connect these two lines, however it is tuned.", { x: MX, y: 5.5, w: 5.6, h: 0.8, fontSize: 17, color: INK2 });
  txt(s, "Same Bhavarth", { x: MX + 6.2, y: 4.85, w: 5.9, h: 0.6, fontFace: "Cambria", fontSize: 28, bold: true, color: GOOD });
  txt(s, "A system that understands meaning should place them next to each other.", { x: MX + 6.2, y: 5.5, w: 5.6, h: 0.8, fontSize: 17, color: INK2 });
}

// 4. Objectives (verbatim from report Section 1.2)
{
  const s = content("Introduction", "Objectives", "Objectives",
    "These are the three objectives exactly as stated in the report.");
  txt(s, "Following are the objectives for the project work:", { x: MX, y: 1.75, w: CW, h: 0.4, fontSize: 18, color: INK2 });
  const objs = [
    [{ text: "To develop a semantic representation framework for capturing the contextual meaning (" }, { text: "Bhavarth", options: { italic: true } }, { text: ") of Gurbani text." }],
    [{ text: "To develop a semantic retrieval system for identifying and ranking Gurbani Pankatis based on contextual and thematic similarity." }],
    [{ text: "To develop an intelligent Gurbani information system integrating semantic retrieval with Gurbani metadata, Raag information, and relevant digital resources." }],
  ];
  objs.forEach((o, i) => {
    const y = 2.45 + i * 1.45;
    card(s, MX, y, CW, 1.25);
    badge(s, MX + 0.35, y + 0.33, String(i + 1), 0.6);
    s.addText(o, { isTextBox: true, x: MX + 1.3, y: y + 0.1, w: CW - 1.7, h: 1.05, margin: 0, fontSize: 20, color: INK, valign: "middle" });
  });
}

// 5. Feasibility
pres.addSection({ title: "Feasibility and background" });
{
  const s = content("Feasibility and background", "Feasibility study", "The project is feasible on four counts, and the first phase shows it",
    "Technical: model and data already exist; LoRA makes training fit one GPU. Economic: everything is free and open source. Operational: search runs on a laptop. Schedule: phase 1 is done.");
  const cols = [
    ["Technical", "Feasible", ["BGE-M3 already supports Punjabi; no model is built from scratch", "60,360 Pankatis with five Steeks each are already collected", "LoRA trains about 0.4% of parameters, so training fits a 12 GB RTX 3060", "FAISS searches 54,844 vectors in under a second"]],
    ["Economic", "Feasible", ["All software is open source: PyTorch, sentence-transformers, PEFT, FAISS", "The base model is a free download", "No paid API or cloud service", "A student laptop and one consumer GPU; a training run takes about 10 minutes"]],
    ["Operational", "Feasible", ["Runs on an ordinary laptop: stored verses use cached vectors, so the model is never loaded", "Accepts a verse without its danda, a partial verse, or a whole Shabad", "Each result shows Ang, Raag and Steek, so readers can check the match"]],
    ["Schedule", "On track", ["Phase 1 complete: clean data, benchmark, and a fine-tuned model that beats keyword search", "Phase 2 builds on finished stages", "Stages are independent scripts, each testable alone"]],
  ];
  const w = 2.9, gap = 0.177;
  cols.forEach((c, i) => {
    const x = MX + i * (w + gap);
    card(s, x, 2.0, w, 4.45);
    txt(s, c[0], { x: x + 0.25, y: 2.2, w: 1.6, h: 0.4, fontSize: 19, bold: true, valign: "middle" });
    tag(s, x + w - 1.05, 2.26, 0.85, c[1], true);
    bullets(s, c[2], { x: x + 0.25, y: 2.8, w: w - 0.45, h: 3.55, fontSize: 14, color: INK2, gap: 8 });
  });
  txt(s, "Ethical: the system retrieves existing Gurbani and never generates scripture. Explanations come from the Steeks of named commentators.", { x: MX, y: 6.55, w: CW, h: 0.35, fontSize: 13, italic: true, color: INK2 });
}

// 6. Literature survey
{
  const s = content("Feasibility and background", "Literature survey", "What existing work provides, and what it leaves open",
    "Dense retrieval and multilingual embeddings exist, but none cover archaic Gurmukhi, and dense retrieval normally needs labelled pairs that do not exist here.");
  const rows = [
    ["Sentence-BERT", "Reimers & Gurevych, 2019", "Siamese networks trained with a contrastive objective", "General-domain; the training recipe we adapt"],
    ["Dense Passage Retrieval", "Karpukhin et al., 2020", "Dense vectors outperform BM25 for open-domain retrieval", "Needs labelled pairs; none exist for Gurbani"],
    ["LaBSE, Multilingual E5", "Feng et al., 2022; Wang et al., 2024", "Multilingual sentence embeddings, 100+ languages", "Modern web text, not archaic Gurmukhi"],
    ["IndicNLPSuite, Samanantar, IndicSBERT", "2020 – 2023", "Corpora and models for Indian languages incl. Punjabi", "Contemporary Punjabi; no scripture"],
    ["BGE-M3", "Chen et al., 2024", "Multilingual model built for retrieval; our base model", "Unadapted, it loses to BM25 on Gurbani"],
    ["LoRA", "Hu et al., 2022", "Low-rank adapters for cheap fine-tuning", "Makes training fit one GPU"],
    ["BM25, FAISS", "Robertson 2009; Douze et al., 2024", "Keyword ranking; fast vector search", "Our baseline and our index"],
  ];
  const data = [[th("WORK"), th("CONTRIBUTION"), th("GAP FOR GURBANI")]];
  rows.forEach((r) => data.push([
    { text: [{ text: r[0], options: { bold: true, breakLine: true } }, { text: r[1], options: { fontSize: 11, color: INK2 } }] },
    { text: r[2] }, { text: r[3], options: { color: INK2 } },
  ]));
  s.addTable(data, { x: MX, y: 1.95, w: CW, colW: [3.9, 4.5, 3.73], fontSize: 13, color: INK, valign: "middle", margin: [4, 8, 4, 8], border: { type: "solid", pt: 0.75, color: HEX.rule }, rowH: 0.6 });
}

// 8. Key insight
pres.addSection({ title: "Methodology" });
{
  const s = content("Methodology", "Methodology  ·  Key insight", "The commentators already did the semantic work",
    "The verses share nothing, but their Steeks share three words. Word overlap on Steeks stands in for meaning on verses. We use this twice: for training pairs and for the gold benchmark.");
  txt(s, "Meaning is hard to see in the verses and easy to see in the Steeks. Two Pankatis that mean the same thing are explained in nearly the same modern Punjabi.", { x: MX, y: 1.7, w: CW, h: 0.65, fontSize: 17, color: INK2 });
  const G = (t) => ({ text: t, options: { fontFace: GUR, fontSize: 20 } });
  const hl = { fill: { color: HEX.kesriSoft } };
  s.addTable([
    [th(""), th("TEXT"), th("SHARED WORDS")],
    [{ text: "Pankati A", options: { bold: true } }, G("ਵਿਸਰੀ ਤਿਸੈ ਪਰਾਈ ਤਾਤਾ"), { text: "none", options: { rowspan: 2, bold: true, color: BAD, align: "center" } }],
    [{ text: "Pankati B", options: { bold: true } }, G("ਛਾਡਿ ਵਿਡਾਣੀ ਤਾਤਿ ਮੂੜੇ")],
    [{ text: "Steek of A", options: Object.assign({ bold: true }, hl) }, { text: "ਉਸ ਨੂੰ ਹੋਰਨਾਂ ਨਾਲ ਈਰਖਾ ਕਰਨੀ ਭੁੱਲ ਜਾਂਦੀ ਹੈ", options: Object.assign({ fontFace: GUR, fontSize: 20 }, hl) }, { text: "ਈਰਖਾ · ਕਰਨੀ · ਨਾਲ", options: Object.assign({ rowspan: 2, bold: true, color: GOOD, align: "center", fontFace: GUR, fontSize: 18 }, hl) }],
    [{ text: "Steek of B", options: Object.assign({ bold: true }, hl) }, { text: "ਹੇ ਮੂਰਖ! ਦੂਜਿਆਂ ਨਾਲ ਈਰਖਾ ਕਰਨੀ ਛੱਡ ਦੇ", options: Object.assign({ fontFace: GUR, fontSize: 20 }, hl) }],
  ], { x: MX, y: 2.55, w: CW, colW: [2.0, 7.13, 3.0], fontSize: 15, color: INK, valign: "middle", margin: [4, 10, 4, 10], border: { type: "solid", pt: 0.75, color: HEX.rule }, rowH: [0.42, 0.65, 0.65, 0.65, 0.65] });
  txt(s, [{ text: "A pair counts only when " }, { text: "two independent commentators", options: { bold: true, color: INK } }, { text: " (Sahib Singh and Manmohan Singh) both explain the verses alike, and the verses come from different Shabads. The same rule mines " }, { text: "training pairs", options: { bold: true, color: INK } }, { text: " from the train split and a " }, { text: "gold benchmark", options: { bold: true, color: INK } }, { text: " from the test split." }], { x: MX, y: 5.95, w: CW, h: 0.8, fontSize: 15, color: INK2 });
}

// 9. Dataset
{
  const s = content("Methodology", "Methodology  ·  Dataset and split", "From 60,360 rows to a clean, leakage-free corpus",
    "The source flag caught only 720 of about 5,000 heading lines; our detector finds the rest. The split is grouped, not random: union-find joins rows sharing a Shabad, an identical Pankati or an identical Steek, so near-duplicates cannot straddle train and test.");
  const steps = [["60,360", "rows in the source workbook, five Steeks each", INK], ["−5,077", "structural and heading lines, flagged for audit, not deleted", BAD], ["55,217", "trainable rows retained", INK], ["54,844", "unique Pankatis in the retrieval corpus", KESRI]];
  const w = 2.75, gap = 0.38;
  steps.forEach((st, i) => {
    const x = MX + i * (w + gap);
    stat(s, x, 1.85, w, st[0], st[1], st[2], 38);
    if (i < 3) s.addShape(pres.shapes.CHEVRON, { x: x + w + 0.08, y: 2.35, w: 0.22, h: 0.3, fill: { color: GREY }, line: { type: "none" } });
  });
  txt(s, "GROUPED SPLIT  ·  union-find on same Shabad, identical Pankati, identical Steek", { x: MX, y: 4.05, w: CW, h: 0.3, fontSize: 12, bold: true, charSpacing: 1, color: INK2 });
  const parts = [[0.8, "Train  ·  131,806 pairs", INK], [0.1, "Val  ·  16,457", INK2], [0.1, "Test  ·  16,515", KESRI]];
  let px = MX;
  parts.forEach((p) => {
    const pw = CW * p[0];
    s.addShape(pres.shapes.RECTANGLE, { x: px, y: 4.45, w: pw, h: 0.7, fill: { color: p[2] }, line: { color: WHITE, width: 1 } });
    txt(s, p[1], { x: px, y: 4.45, w: pw, h: 0.7, fontSize: 14, bold: true, color: WHITE, align: "center", valign: "middle" });
    px += pw;
  });
  card(s, MX, 5.45, CW, 1.2, HEX.goodSoft);
  txt(s, "0", { x: MX + 0.35, y: 5.5, w: 1.0, h: 1.1, fontFace: "Cambria", fontSize: 48, bold: true, color: GOOD, valign: "middle" });
  txt(s, "overlap on all 12 cross-partition checks (Pankati, Shabad, group, Steek), so the results measure learning, not leakage.", { x: MX + 1.3, y: 5.55, w: CW - 1.7, h: 1.0, fontSize: 17, valign: "middle" });
}

// 11. Architecture
pres.addSection({ title: "System design" });
{
  const s = content("System design", "System design  ·  Architecture", "Offline preparation, online retrieval",
    "Left: offline pipeline, run once. Right: what happens per query. A query already in the corpus reads its cached vector, so search works on the laptop without loading the 2.3 GB model.");
  const h = 4.95, w = h * 3920 / 2983;
  s.addImage({ path: img("fig_diag_architecture.png"), x: MX, y: 1.85, w, h, altText: "System architecture: offline pipeline and online retrieval path" });
  bullets(s, [
    "Six independent stage scripts; each reads files and writes new ones, so any stage re-runs alone",
    "Base model BGE-M3: 568M parameters, 24 layers, 1024-dimensional vectors, covers Punjabi",
    "A query already in the corpus uses its cached vector; the model is never loaded and search takes under a second",
    "Results exclude the query's own Shabad and keep at most one line per Shabad",
  ], { x: MX + w + 0.45, y: 2.1, w: W - MX - (MX + w + 0.45), h: 4.6, fontSize: 16, gap: 14 });
}

// 13. Model training
{
  const s = content("System design", "Methodology  ·  Model training", "Contrastive fine-tuning with LoRA",
    "The model pulls the anchor toward its positive and pushes it from the hard negative, which shares words but means the opposite. Batch size matters: other batch items act as negatives.");
  const w = 8.6, h = w * 1605 / 4379;
  s.addImage({ path: img("fig_diag_contrastive.png"), x: (W - w) / 2, y: 1.8, w, h, altText: "Anchor pulled toward positive with same meaning and pushed from hard negative with shared words" });
  const st = [["0.4%", "of parameters trained: LoRA on query, key and value, rank 16; the adapter is about 5 MB"], ["63", "negatives per example at batch size 64; the other batch items act as negatives"], ["~10 min", "for 5 epochs on an RTX 3060; corpus encoding takes 3 min there vs 112 min on the laptop"]];
  st.forEach((t, i) => {
    const x = MX + i * 4.14, fw = 3.85;
    card(s, x, 5.1, fw, 1.6);
    txt(s, t[0], { x: x + 0.3, y: 5.2, w: 1.6, h: 1.4, fontFace: "Cambria", fontSize: 28, bold: true, color: KESRI, valign: "middle" });
    txt(s, t[1], { x: x + 1.85, y: 5.25, w: fw - 2.05, h: 1.3, fontSize: 13, color: INK2, valign: "middle" });
  });
}

// 14. Results chart
pres.addSection({ title: "Results" });
{
  const s = content("Results", "Results  ·  Hit@10 on the benchmark", "Verse-pair tuning is the first system to beat keyword search",
    "Pretrained BGE-M3 is below both keyword baselines. Steek-pair tuning did not help. Verse-pair tuning is the only system above the best keyword baseline.");
  const labels = ["BM25, word level", "BM25, character 3–5 gram", "BGE-M3, pretrained", "BGE-M3, Steek-pair tuning", "BGE-M3, verse-pair tuning"];
  const vals = [23.18, 27.09, 19.37, 18.50, 28.29];
  s.addChart(pres.charts.BAR, [{ name: "Hit@10 (%)", labels: labels.slice().reverse(), values: vals.slice().reverse() }], {
    x: MX, y: 1.85, w: 8.4, h: 4.9, barDir: "bar", barGapWidthPct: 45,
    chartColors: [HEX.kesri, HEX.grey, HEX.grey, "D3D8E4", "D3D8E4"],
    showValue: true, dataLabelPosition: "outEnd", dataLabelFontSize: 14, dataLabelColor: HEX.ink, dataLabelFontFace: "+mn-lt", dataLabelFormatCode: "0.00",
    catAxisLabelColor: HEX.ink, catAxisLabelFontSize: 14, catAxisLabelFontFace: "+mn-lt",
    valAxisLabelColor: HEX.ink2, valAxisLabelFontSize: 12, valAxisLabelFontFace: "+mn-lt", valAxisMinVal: 0, valAxisMaxVal: 30, valAxisMajorUnit: 10,
    valGridLine: { color: HEX.rule, size: 0.75 }, catGridLine: { style: "none" }, showLegend: false, showTitle: false,
  });
  const X2 = 9.4, W2 = W - MX - X2;
  stat(s, X2, 2.1, W2, "+8.92", "points of Hit@10 over the pretrained model (19.37 → 28.29)", KESRI, 40);
  stat(s, X2, 4.0, W2, "2.5×", "the pretrained model on pairs sharing no word with the query (1.43 → 3.57)", INK, 30);
  txt(s, "Hit@10: share of queries with at least one gold Pankati in the top 10. Identical queries, same 54,844-line corpus, own Shabad excluded.", { x: X2, y: 5.85, w: W2, h: 0.9, fontSize: 12, color: INK2 });
}

// 16. Findings
{
  const s = content("Results", "Results  ·  Findings", "Three findings that set the direction of the project",
    "The diagnostic in the middle is the key: training worked on the task it was given, but that task was the wrong one. So we changed the objective, not the amount of data.");
  const F = [
    ["19.37 vs 27.09", INK, "The pretrained model loses to keyword search", "A state-of-the-art multilingual model, used as is, scores below character BM25. This justifies Gurbani-specific training rather than assuming it."],
    ["38.0 → 61.6", INK, "Training on the wrong task changes nothing", "Pankati↔Steek tuning lifted Steek→Pankati Hit@10 sharply, yet Pankati→Pankati stayed flat (19.37 → 18.50). The model learned; the skill did not transfer."],
    ["+8.92", KESRI, "The right pairs beat more pairs", "A few thousand mined verse-to-verse pairs did what 131,806 Steek pairs could not. The objective had to change, not the amount of data."],
  ];
  F.forEach((f, i) => {
    const x = MX + i * 4.14, w = 3.85;
    card(s, x, 2.0, w, 4.6);
    txt(s, f[0], { x: x + 0.3, y: 2.25, w: w - 0.6, h: 0.8, fontFace: "Cambria", fontSize: 32, bold: true, color: f[1], valign: "middle" });
    txt(s, f[2], { x: x + 0.3, y: 3.25, w: w - 0.6, h: 0.8, fontSize: 18, bold: true });
    txt(s, f[3], { x: x + 0.3, y: 4.1, w: w - 0.6, h: 2.4, fontSize: 15, color: INK2 });
  });
}

// 19. Testing
pres.addSection({ title: "Testing and conclusion" });
{
  const s = content("Testing and conclusion", "Testing", "20 test cases executed, all pass",
    "A selection of the 20 test cases in the report. Beyond these: retrieval metrics against keyword baselines, ablation of each change, and manual inspection of six known zero-overlap queries.");
  const T = [
    ["T-01", [g("ਪਉੜੀ ॥"), { text: " detected as heading" }]],
    ["T-04", [{ text: "Full Mool Mantar retained as scripture" }]],
    ["T-06", [g("ਰਾਗੁ ਨਾਦੁ ਸਭੁ ਸਚੁ ਹੈ"), { text: " … kept by scripture guard" }]],
    ["T-07–09", [{ text: "Cross-partition Pankati, Shabad, Steek overlap is zero" }]],
    ["T-10", [{ text: "Embedding count equals corpus size, 54,844" }]],
    ["T-11", [{ text: "Query without its danda still matches" }]],
    ["T-13", [{ text: "Whole-Shabad query becomes one averaged vector" }]],
    ["T-15", [{ text: "At most one result per Shabad" }]],
    ["T-17", [{ text: "Saved model reloads identically, cosine 1.000" }]],
    ["T-20", [{ text: "Interrupted encoding resumes from the last shard" }]],
  ];
  const mk = (rows) => [[th("ID"), th("TEST CASE"), th("STATUS", { align: "center" })]].concat(rows.map((r) => [
    { text: r[0], options: { fontFace: MONO, fontSize: 13, color: INK2 } }, { text: r[1] }, { text: "PASS", options: { bold: true, color: GOOD, align: "center", fontSize: 12, fill: { color: HEX.goodSoft } } },
  ]));
  const tw = 5.9, opt = (x) => ({ x, y: 1.95, w: tw, colW: [1.05, 3.85, 1.0], fontSize: 14, color: INK, valign: "middle", margin: [4, 8, 4, 8], border: { type: "solid", pt: 0.75, color: HEX.rule }, rowH: 0.6 });
  s.addTable(mk(T.slice(0, 5)), opt(MX));
  s.addTable(mk(T.slice(5)), opt(W - MX - tw));
  txt(s, "Also: retrieval evaluation (Hit@k, MRR, nDCG) with keyword baselines on identical queries, ablation of each design change, and manual inspection of six queries with known zero-overlap answers.", { x: MX, y: 5.85, w: CW, h: 0.8, fontSize: 14, color: INK2 });
}

// 21. Conclusion and future scope
{
  const s = content("Testing and conclusion", "Conclusion and future scope", "What phase one established, and what comes next",
    "Conclusion: labels without manual work; the pretrained model is not enough; the objective matters more than data volume. Caveat to state if asked: the benchmark is mined from Steeks, so measured recall is a lower bound and the benchmark is not fully independent of training; BM25 is scored on the same set, so comparisons stay fair. Next: hard-negative training, reranker plus LLM explanations, then a web interface.");
  txt(s, "CONCLUSION", { x: MX, y: 1.95, w: 6, h: 0.3, fontSize: 12, bold: true, charSpacing: 2, color: INK2 });
  bullets(s, [
    "Verse-to-verse supervision can be built without manual labels, from agreement between two independent commentators",
    "A pretrained multilingual model is not enough: unadapted BGE-M3 scores below keyword search",
    "The objective matters more than data volume: verse-pair tuning reached 28.29 Hit@10 and 16.77 MRR, best on every measure",
    "Verses sharing no word improved 2.5× but remain the hardest case",
  ], { x: MX, y: 2.4, w: 6.3, h: 4.3, fontSize: 16, gap: 12 });
  const X2 = 7.4, W2 = W - MX - X2;
  txt(s, "FUTURE SCOPE", { x: X2, y: 1.95, w: W2, h: 0.3, fontSize: 12, bold: true, charSpacing: 2, color: INK2 });
  const R = [
    ["NEXT", "Hard-negative training", "Train on the 5,343 mined triplets so that opposite teachings separate"],
    ["THEN", "Reranker and explanations", "Rerank with bge-reranker-v2-m3; an LLM layer explains each match from its Steeks"],
    ["FINALLY", "Web interface", "A FastAPI service and web front end replace the command line"],
  ];
  const y0 = 2.45, step = 1.45;
  s.addShape(pres.shapes.LINE, { x: X2 + 0.17, y: y0 + 0.17, w: 0, h: step * 2, line: { color: HEX.ink, width: 2 } });
  R.forEach((r, i) => {
    const y = y0 + i * step;
    s.addShape(pres.shapes.OVAL, { x: X2, y, w: 0.34, h: 0.34, fill: { color: i === 0 ? KESRI : WHITE }, line: { color: i === 0 ? HEX.kesri : HEX.ink, width: 2 } });
    txt(s, [{ text: r[0] + "   ", options: { fontSize: 11, bold: true, charSpacing: 2, color: i === 0 ? KESRI : INK2 } }, { text: r[1], options: { fontSize: 18, bold: true } }], { x: X2 + 0.6, y: y - 0.05, w: W2 - 0.6, h: 0.45, valign: "middle" });
    txt(s, r[2], { x: X2 + 0.6, y: y + 0.45, w: W2 - 0.6, h: 0.85, fontSize: 14, color: INK2 });
  });
}

// 23. References
{
  const s = content("Testing and conclusion", "References", "References", "IEEE-style list matching the report bibliography.");
  const refs = [
    "D. Kakwani et al., \"IndicNLPSuite: Monolingual corpora, evaluation benchmarks and pre-trained multilingual language models for Indian languages,\" in Findings of EMNLP, 2020.",
    "G. Ramesh et al., \"Samanantar: The largest publicly available parallel corpora collection for 11 Indic languages,\" Trans. ACL, 2022.",
    "F. Feng, Y. Yang, D. Cer, N. Arivazhagan and W. Wang, \"Language-agnostic BERT sentence embedding,\" in Proc. 60th ACL, 2022.",
    "L. Wang, N. Yang, X. Huang, L. Yang, R. Majumder and F. Wei, \"Multilingual E5 text embeddings: A technical report,\" arXiv:2402.05672, 2024.",
    "S. Deode, J. Gadre, A. Kajale, A. Joshi and R. Joshi, \"L3Cube-IndicSBERT: A simple approach for learning cross-lingual sentence representations using multilingual BERT,\" arXiv:2304.11434, 2023.",
    "V. Karpukhin et al., \"Dense passage retrieval for open-domain question answering,\" in Proc. EMNLP, 2020.",
    "M. Douze et al., \"The Faiss library,\" arXiv:2401.08281, 2024.",
    "J. Chen, S. Xiao, P. Zhang, K. Luo, D. Lian and Z. Liu, \"BGE M3-Embedding: Multi-lingual, multi-functionality, multi-granularity text embeddings through self-knowledge distillation,\" in Findings of ACL, 2024.",
    "E. J. Hu et al., \"LoRA: Low-rank adaptation of large language models,\" in Proc. ICLR, 2022.",
    "N. Reimers and I. Gurevych, \"Sentence-BERT: Sentence embeddings using Siamese BERT-networks,\" in Proc. EMNLP, 2019.",
    "S. Robertson and H. Zaragoza, \"The probabilistic relevance framework: BM25 and beyond,\" Foundations and Trends in Information Retrieval, 2009.",
  ];
  const col = (list, start, x) => s.addText(list.map((r, i) => ({ text: `[${start + i}]  ${r}`, options: { breakLine: i < list.length - 1, paraSpaceAfter: 8 } })),
    { isTextBox: true, x, y: 1.9, w: 5.9, h: 4.95, margin: 0, fontSize: 12, color: INK2, valign: "top" });
  col(refs.slice(0, 6), 1, MX);
  col(refs.slice(6), 7, W - MX - 5.9);
}

// 24. Thank you
pres.addSection({ title: "Closing" });
{
  const s = pres.addSlide({ masterName: "DARK", sectionTitle: "Closing" });
  txt(s, "THANK YOU", { x: 0.8, y: 2.0, w: 8, h: 0.4, fontSize: 14, bold: true, charSpacing: 2, color: KESRI });
  txt(s, "Questions", { x: 0.8, y: 2.45, w: 10, h: 1.2, fontFace: "Cambria", fontSize: 54, bold: true, color: WHITE });
  txt(s, "ਜਤ ਕਤ ਦੇਖਉ ਤਤ ਰਹਿਆ ਸਮਾਇ", { x: 0.8, y: 3.8, w: 10, h: 0.8, fontFace: GUR, fontSize: 28, color: "F1A85B" });
  txt(s, "Balkaran Singh  ·  Gurpreet Kaur  ·  Gurveer Singh    |    Guide: Dr. Parminder Singh", { x: 0.8, y: 5.2, w: 11, h: 0.4, fontSize: 16, color: "C9D1E6" });
  s.addNotes("Invite questions. Likely ones: why the benchmark is trustworthy (grouped split, same set for all systems), why the pretrained model lost, what hard negatives will change.");
}

(async () => {
  await pres.writeFile({ fileName: OUT });
  await applyTheme(OUT, THEME);
  console.log("wrote", OUT);
})();
