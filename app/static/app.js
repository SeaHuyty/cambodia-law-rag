const $ = (id) => document.getElementById(id);
const form = $("form"), input = $("question"), submit = $("submit");
const loadingEl = $("loading"), answerEl = $("answer"), bodyEl = $("answer-body"), labelEl = $("answer-label");
const errorEl = $("error"), sourcesEl = $("sources"), listEl = $("source-list"), countEl = $("source-count");

const EXAMPLES = [
  "តើភ្លើងសញ្ញាចរាចរណ៍ពណ៌ក្រហមមានន័យយ៉ាងដូចម្តេច?",
  "តើអ្នកបើកបរត្រូវធ្វើអ្វីនៅពេលជិតដល់ផ្លូវរថភ្លើង?",
];
for (const q of EXAMPLES) {
  const b = document.createElement("button");
  b.type = "button"; b.className = "chip"; b.textContent = q;
  b.onclick = () => { input.value = q; form.requestSubmit(); };
  $("examples").append(b);
}

// Khmer digits -> int, so citations match source article numbers.
const toInt = (s) => parseInt([...s].map((c) => (c >= "០" && c <= "៩" ? c.charCodeAt(0) - 0x17e0 : c)).join(""), 10);
const CITE = /មាត្រា\s*([0-9០-៩]+)/g;

// Inline: **bold** and citation chips. Built with DOM nodes (no innerHTML) so model output can't inject markup.
function inline(parent, text) {
  text.split(/(\*\*[^*]+\*\*)/).forEach((part) => {
    const bold = part.startsWith("**") && part.endsWith("**") && part.length > 4;
    const host = bold ? parent.appendChild(document.createElement("strong")) : parent;
    const t = bold ? part.slice(2, -2) : part;
    let last = 0;
    for (const m of t.matchAll(CITE)) {
      host.append(t.slice(last, m.index));
      const c = document.createElement("button");
      c.type = "button"; c.className = "cite"; c.textContent = m[0];
      c.onclick = () => openSource(toInt(m[1]));
      host.append(c);
      last = m.index + m[0].length;
    }
    host.append(t.slice(last));
  });
}

// Minimal markdown: paragraphs, "-/*" bullets, "1." lists.
function render(text) {
  bodyEl.replaceChildren();
  let list = null, kind = null;
  for (const raw of text.split("\n")) {
    const line = raw.trim();
    if (!line) { list = null; continue; }
    const li = line.match(/^([-*•]|\d+[.)])\s+(.*)$/);
    if (li) {
      const want = /\d/.test(li[1]) ? "ol" : "ul";
      if (!list || kind !== want) { list = bodyEl.appendChild(document.createElement(want)); kind = want; }
      inline(list.appendChild(document.createElement("li")), li[2]);
    } else {
      list = null;
      inline(bodyEl.appendChild(document.createElement("p")), line);
    }
  }
}

function openSource(n) {
  const el = listEl.querySelector(`[data-article="${n}"]`);
  if (!el) return;
  el.open = true;
  el.scrollIntoView({ behavior: "smooth", block: "center" });
  el.classList.remove("flash"); void el.offsetWidth; el.classList.add("flash");
}

function renderSources(sources, answer) {
  listEl.replaceChildren();
  const cited = new Set([...(answer || "").matchAll(CITE)].map((m) => toInt(m[1])));
  // Cited articles first.
  const num = (s) => s.metadata?.article_number ?? s.metadata?.article;
  [...sources].sort((a, b) => cited.has(num(b)) - cited.has(num(a))).forEach((s) => {
    const m = s.metadata || {}, n = num(s);
    const d = document.createElement("details");
    d.className = "source";
    if (n != null) d.dataset.article = n;
    const sum = document.createElement("summary");
    sum.append(n != null ? `មាត្រា ${n}` : s.id);
    if (m.article_title) sum.append(` · ${m.article_title}`);
    if (cited.has(n)) { const b = document.createElement("span"); b.className = "badge"; b.textContent = "ដកស្រង់"; sum.append(b); }
    const sc = document.createElement("span"); sc.className = "score"; sc.textContent = `${Math.round(s.score * 100)}%`;
    sum.append(sc);
    const p = document.createElement("p"); p.textContent = s.text;
    d.append(sum, p);
    listEl.append(d);
  });
  countEl.textContent = `(${sources.length})`;
  sourcesEl.hidden = !sources.length;
}

function reset() { loadingEl.hidden = answerEl.hidden = errorEl.hidden = sourcesEl.hidden = true; }

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const question = input.value.trim();
  if (!question) return;
  submit.disabled = true;
  reset(); loadingEl.hidden = false;
  try {
    const res = await fetch("/api/ask", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ question }),
    });
    if (!res.ok) throw new Error(res.status);
    const data = await res.json();
    loadingEl.hidden = true;
    if (data.answer) {
      render(data.answer);
      answerEl.classList.toggle("refused", !!data.refused);
      labelEl.textContent = data.refused ? "មិនអាចឆ្លើយបាន" : "ចម្លើយ";
      answerEl.hidden = false;
    } else {
      errorEl.textContent = "មិនអាចបង្កើតចម្លើយបានទេ ប៉ុន្តែនេះជាមាត្រាដែលពាក់ព័ន្ធ។";
      errorEl.hidden = false;
    }
    // A refusal means the retrieved articles weren't relevant, so don't present them as sources.
    if (!data.refused) renderSources(data.sources, data.answer);
  } catch {
    loadingEl.hidden = true;
    errorEl.textContent = "មានបញ្ហា សូមព្យាយាមម្តងទៀត។";
    errorEl.hidden = false;
  } finally {
    submit.disabled = false;
  }
});

$("copy").addEventListener("click", async (e) => {
  try { await navigator.clipboard.writeText(bodyEl.innerText); e.target.textContent = "បានចម្លង ✓"; }
  catch { e.target.textContent = "ចម្លងមិនបាន"; }
  setTimeout(() => (e.target.textContent = "ចម្លង"), 1500);
});

input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); form.requestSubmit(); }
});
