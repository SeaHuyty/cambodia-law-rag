const form = document.getElementById("form");
const input = document.getElementById("question");
const submit = document.getElementById("submit");
const statusEl = document.getElementById("status");
const answerEl = document.getElementById("answer");
const sourcesEl = document.getElementById("sources");
const listEl = document.getElementById("source-list");

function setStatus(msg, isError = false) {
  statusEl.textContent = msg;
  statusEl.classList.toggle("error", isError);
}

function renderSources(sources) {
  listEl.replaceChildren();
  for (const s of sources) {
    const m = s.metadata || {};
    const details = document.createElement("details");
    details.className = "source";
    const summary = document.createElement("summary");
    const art = m.article_number ?? m.article;
    summary.textContent = art != null ? `មាត្រា ${art}` : s.id;
    if (m.article_title) summary.textContent += ` - ${m.article_title}`;
    const meta = document.createElement("span");
    meta.className = "meta";
    meta.textContent = `${(s.score * 100).toFixed(0)}%`;
    summary.append(meta);
    const p = document.createElement("p");
    p.textContent = s.text;
    details.append(summary, p);
    listEl.append(details);
  }
  sourcesEl.hidden = sources.length === 0;
}

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const question = input.value.trim();
  if (!question) return;
  submit.disabled = true;
  answerEl.hidden = sourcesEl.hidden = true;
  setStatus("កំពុងស្វែងរក...");
  try {
    const res = await fetch("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
    if (!res.ok) throw new Error(res.status);
    const data = await res.json();
    setStatus(data.sources.length ? "" : "រកមិនឃើញមាត្រាដែលពាក់ព័ន្ធទេ។");
    if (data.answer) {
      answerEl.textContent = data.answer;
      answerEl.hidden = false;
    }
    renderSources(data.sources);
  } catch (err) {
    setStatus("មានបញ្ហា សូមព្យាយាមម្តងទៀត។", true);
  } finally {
    submit.disabled = false;
  }
});

input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); form.requestSubmit(); }
});
