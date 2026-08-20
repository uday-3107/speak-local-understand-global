import { createFeedback, getSession, joinSession, listSegments, listSessions, transcriptUrl } from "./api.js";

const $ = (sel) => document.querySelector(sel);
const viewEls = {
  overview: $("#view-overview"),
  live: $("#view-live"),
  history: $("#view-history"),
  analytics: $("#view-analytics"),
};

let activeSession = null;   // joined live session
let targetLang = "hi";
let pollTimer = null;
let seenSegmentIds = new Set();
let captions = [];
let sessionEnded = false;

/* ---------- navigation ---------- */

function showView(name, { push = true } = {}) {
  Object.entries(viewEls).forEach(([key, el]) => {
    el.classList.toggle("hidden", key !== name);
  });
  document.querySelectorAll(".nav-item").forEach((b) =>
    b.classList.toggle("active", b.dataset.nav === name));
  if (name === "history") renderHistory();
  if (name === "analytics") renderAnalytics();
  if (push) history.replaceState(null, "", `#${name}`);
}

document.querySelectorAll(".nav-item, [data-nav]").forEach((b) =>
  b.addEventListener("click", () => showView(b.dataset.nav)));
document.querySelectorAll("[data-goto]").forEach((b) =>
  b.addEventListener("click", () => showView(b.dataset.goto)));
$("#menu-toggle").addEventListener("click", () => {
  $("#sidebar").classList.remove("-translate-x-full");
  $("#overlay").classList.remove("hidden");
});
$("#overlay").addEventListener("click", () => {
  $("#sidebar").classList.add("-translate-x-full");
  $("#overlay").classList.add("hidden");
});

window.addEventListener("hashchange", () => {
  const name = ["overview", "live", "history", "analytics"].includes(
    location.hash.slice(1)) ? location.hash.slice(1) : "overview";
  showView(name, { push: false });
});

/* ---------- overview ---------- */

async function renderOverview() {
  try {
    const [analytics, sessions] = await Promise.all([
      fetch("/api/v1/analytics").then((r) => r.json()),
      listSessions(),
    ]);
    const live = sessions.filter((s) => !s.ended_at);
    $("#stat-live").textContent = live.length;
    $("#stat-total").textContent = sessions.length;
    const segments = await Promise.all(
      sessions.map((s) => listSegments(s.id).catch(() => [])));
    const segCount = segments.reduce((n, list) => n + list.length, 0);
    $("#stat-segments").textContent = segCount;
    const sat = analytics.satisfaction;
    $("#stat-satisfaction").textContent =
      Math.round(sat.positive_ratio * 100) + "%";
    const best = analytics.ml.rows[0];
    $("#stat-best").textContent = best ? best.model : "—";
  } catch {
    /* analytics may be unavailable; stats stay "–" */
  }
}

/* ---------- live lecture ---------- */

const LANGS = [
  ["en", "English"],
  ["hi", "Hindi (हिन्दी)"],
  ["te", "Telugu (తెలుగు)"],
];

function renderCaptions() {
  const list = $("#live-captions");
  if (captions.length === 0) {
    list.innerHTML = `<li class="text-sm text-on-surface-variant flex items-center gap-2">
        <span class="pulse-dot h-2 w-2 rounded-full bg-accent-teal"></span>
        You're in the class. Waiting for the lecturer to start speaking — live captions will appear here.
      </li>`;
    return;
  }
  list.innerHTML = captions.map((c) => `
    <li class="fade-in bg-surface-container-low rounded-xl p-4" data-seg="${c.id}">
      <p class="text-[15px] leading-relaxed font-medium">${escapeHtml(c.translated_text)}</p>
      <p class="text-xs text-on-surface-variant mt-1">${escapeHtml(c.source_text)}</p>
      <div class="flex items-center gap-3 mt-2.5">
        <span class="text-[11px] text-on-surface-variant shrink-0">${time(c.timestamp)} · ${c.model_used || "engine"} · ${latency(c.latency_ms)}</span>
        <button class="fb ml-auto grid place-items-center h-7 w-7 rounded-lg bg-white border border-outline-variant/60 text-on-surface-variant hover:text-accent-teal" data-rating="true" title="Good caption"><span class="material-symbols-outlined text-[18px]">thumb_up</span></button>
        <button class="fb grid place-items-center h-7 w-7 rounded-lg bg-white border border-outline-variant/60 text-on-surface-variant hover:text-error" data-rating="false" title="Bad caption"><span class="material-symbols-outlined text-[18px]">thumb_down</span></button>
      </div>
    </li>`).join("");

  list.querySelectorAll(".fb").forEach((b) =>
    b.addEventListener("click", () => rate(b.dataset.rating === "true", b)));
  list.scrollTop = list.scrollHeight;
}

async function joinLecture() {
  const code = $("#join-code").value.trim().toUpperCase();
  const hint = $("#join-hint");
  if (!code) {
    hint.textContent = "Enter the code your lecturer shows before joining.";
    hint.classList.add("text-error");
    return;
  }
  hint.textContent = "Joining…";
  targetLang = $("#join-lang").value;
  try {
    const session = await joinSession(code, targetLang);
    activeSession = session;
    sessionEnded = false;
    $("#live-title").textContent = `${session.subject || "Lecture"} · #${shortId(session.id)}`;
    // pre-mark everything recorded BEFORE joining so only new captions stream
    seenSegmentIds = new Set();
    try {
      for (const seg of await listSegments(session.id)) seenSegmentIds.add(seg.id);
    } catch {
      /* no historical segments */
    }
    captions = [];
    renderCaptions();
    $("#join-card").classList.add("hidden");
    $("#join-hint").textContent = "";
    $("#live-card").classList.remove("hidden");
    $("#live-card").classList.add("grid");
    $("#live-chip").classList.remove("hidden");
    $("#m-live-chip").classList.remove("hidden");
    $("#m-live-chip").classList.add("flex");
    pollSegments();
    pollTimer = setInterval(pollSegments, 1500);
  } catch (err) {
    hint.textContent = `Could not join: ${err.message}. Check the code and try again.`;
    hint.classList.add("text-error");
  }
}

async function pollSegments() {
  if (!activeSession) return;
  try {
    const segs = await listSegments(activeSession.id);
    for (const s of segs) {
      if (seenSegmentIds.has(s.id)) continue;
      seenSegmentIds.add(s.id);
      captions.push(s);
    }
    if (captions.length) renderCaptions();
  } catch {
    /* transient */
  }
  try {
    const fresh = await getSession(activeSession.id);
    if (fresh.ended_at && !sessionEnded) {
      sessionEnded = true;
      clearInterval(pollTimer);
      showEndedModal();
    }
  } catch {
    /* transient */
  }
}

function showEndedModal() {
  const modal = $("#end-modal");
  const chip = $("#live-chip");
  const mChip = $("#m-live-chip");
  chip.classList.remove("text-accent-teal", "flex");
  chip.classList.add("text-on-surface-variant", "inline-flex");
  chip.innerHTML = `<span class="material-symbols-outlined text-[14px]">stop_circle</span> Ended`;
  mChip.classList.remove("text-accent-teal", "flex");
  mChip.classList.add("text-on-surface-variant");
  mChip.innerHTML = `<span class="material-symbols-outlined text-[14px]">stop_circle</span> Ended`;
  $("#end-download").href = transcriptUrl(activeSession.id);
  modal.classList.remove("hidden");
  modal.classList.add("flex");
}

$("#end-dismiss").addEventListener("click", () => {
  const modal = $("#end-modal");
  modal.classList.add("hidden");
  modal.classList.remove("flex");
});

function shortId(uuid) {
  return uuid ? uuid.slice(0, 8) : "";
}

async function rate(isPositive, btn) {
  const segId = btn.closest("[data-seg]").dataset.seg;
  btn.disabled = true;
  btn.classList.add("bg-accent-teal", "text-white", "border-accent-teal");
  try {
    await createFeedback(segId, isPositive);
  } catch {
    btn.disabled = false;
    btn.classList.remove("bg-accent-teal", "text-white", "border-accent-teal");
  }
}

$("#join-btn").addEventListener("click", joinLecture);

/* ---------- AI study assistant (live Ollama via /api/v1/assistant; template fallback) ---------- */

async function assistantBot(text) {
  try {
    const res = await fetch("/api/v1/assistant", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question: text,
        role: "student",
        context: captions.slice(-6).map((c) => ({
          source_text: c.source_text,
          source_lang: c.source_lang,
          translated_text: c.translated_text,
          target_lang: c.target_lang,
        })),
      }),
    });
    if (!res.ok) throw new Error("assistant offline");
    const data = await res.json();
    if (data.answer) return data.answer;
  } catch {
    /* fall through to template answers */
  }
  const lower = text.toLowerCase();
  if (lower.includes("summar")) {
    if (captions.length === 0) {
      return "I can't summarize yet — no captions have arrived. Join a live lecture and I'll help you recap it.";
    }
    const points = captions.slice(-4).map((c) => `• ${c.translated_text}`).join("\n");
    return `Here's a quick recap of the last few captions:\n${points}`;
  }
  if (lower.includes("last") || lower.includes("latest")) {
    const last = captions[captions.length - 1];
    if (!last) return "No captions yet — start a live lecture first.";
    return `The latest caption was: "${last.translated_text}" (in ${last.target_lang}, via ${last.model_used || "engine"} in ${latency(last.latency_ms)}).`;
  }
  return "The AI assistant is offline right now (Ollama not running). Start it with `ollama serve` for live answers; meanwhile I'll answer from templates.";
}

function chat(message, fromUser = true) {
  const box = $("#assist-msgs");
  const bubble = document.createElement("div");
  bubble.className =
    `max-w-[90%] p-3 rounded-2xl whitespace-pre-line ` +
    (fromUser ? "chat-bubble-out ml-auto rounded-br-sm" : "chat-bubble-in mr-auto rounded-bl-sm");
  bubble.textContent = message;
  box.appendChild(bubble);
  box.scrollTop = box.scrollHeight;
  return bubble;
}

function sendAssistant() {
  const input = $("#assist-input");
  const text = input.value.trim();
  if (!text) return;
  chat(text, true);
  input.value = "";
  const pending = chat("…", false);
  setTimeout(async () => {
    pending.textContent = await assistantBot(text);
    pending.scrollIntoView({ block: "nearest" });
  }, 50);
}

$("#assist-send").addEventListener("click", sendAssistant);
$("#assist-input").addEventListener("keydown", (e) => {
  if (e.key === "Enter") sendAssistant();
});
document.querySelectorAll("[data-suggest]").forEach((b) =>
  b.addEventListener("click", () => {
    chat(b.dataset.suggest, true);
    const pending = chat("…", false);
    setTimeout(async () => {
      pending.textContent = await assistantBot(b.dataset.suggest);
      pending.scrollIntoView({ block: "nearest" });
    }, 50);
  }));

/* ---------- history ---------- */

async function renderHistory() {
  const box = $("#history-list");
  let sessions = [];
  try {
    sessions = await listSessions();
  } catch {
    box.innerHTML = `<div class="text-sm p-10 text-center">Could not load lectures.</div>`;
    return;
  }
  if (!sessions.length) {
    box.innerHTML = `<div class="text-sm text-on-surface-variant p-10 text-center">No lectures yet — they'll appear here once a lecturer starts one.</div>`;
    return;
  }
  box.innerHTML = `
    <table class="w-full text-sm">
      <thead class="bg-surface-container-low/60 text-left text-xs uppercase tracking-wide text-on-surface-variant">
        <tr><th class="px-5 py-3">Lecture</th><th class="px-5 py-3 hidden sm:table-cell">Languages</th><th class="px-5 py-3 hidden md:table-cell">Started</th><th class="px-5 py-3">Status</th><th class="px-5 py-3"></th></tr>
      </thead>
      <tbody>${sessions
        .slice()
        .sort((a, b) => new Date(b.started_at) - new Date(a.started_at))
        .map((s) => `
          <tr class="border-t border-outline-variant/40 hover:bg-surface-container-low/50 cursor-pointer" data-sid="${s.id}">
            <td class="px-5 py-3 font-medium">${escapeHtml(s.subject || "Untitled lecture")}</td>
            <td class="px-5 py-3 hidden sm:table-cell text-on-surface-variant">${s.source_lang} → ${s.target_lang}</td>
            <td class="px-5 py-3 hidden md:table-cell text-on-surface-variant">${time(s.started_at)}</td>
            <td class="px-5 py-3">${s.ended_at
              ? `<span class="text-[11px] px-2 py-0.5 rounded-full bg-surface-variant text-on-surface-variant">ended</span>`
              : `<span class="text-[11px] px-2 py-0.5 rounded-full bg-[#dcfce7] text-[#166534]">live</span>`}</td>
            <td class="px-5 py-3 text-right text-on-surface-variant"><span class="material-symbols-outlined text-[18px]">chevron_right</span></td>
          </tr>`).join("")}
      </tbody>
    </table>
    <div id="history-detail"></div>
  `;
  box.querySelectorAll("tr[data-sid]").forEach((row) =>
    row.addEventListener("click", async () => {
      const detail = $("#history-detail");
      if (detail.dataset.sid === row.dataset.sid && detail.innerHTML) {
        detail.innerHTML = "";
        delete detail.dataset.sid;
        return;
      }
      detail.dataset.sid = row.dataset.sid;
      detail.innerHTML = `<div class="text-sm p-8 text-center text-on-surface-variant">Loading captions…</div>`;
      try {
        const segs = await listSegments(row.dataset.sid);
        detail.innerHTML = segs.length
          ? segs.map((s) => `
              <div class="px-10 py-3 border-t border-outline-variant/40 flex gap-4">
                <p class="text-sm flex-1"><span class="font-medium">${escapeHtml(s.translated_text)}</span>
                  <span class="block text-xs text-on-surface-variant mt-0.5">${escapeHtml(s.source_text)}</span></p>
                <span class="text-[11px] text-on-surface-variant shrink-0 pt-1">${time(s.timestamp)}</span>
              </div>`).join("")
          : `<div class="text-sm p-8 text-center text-on-surface-variant">No captions recorded for this lecture.</div>`;
      } catch {
        detail.innerHTML = `<div class="text-sm p-8 text-center text-on-surface-variant">Captions unavailable.</div>`;
      }
    }));
}

/* ---------- analytics (Plotly) ---------- */

let analyticsRendered = false;

const PLOT = {
  base: {
    font: { family: "Inter, sans-serif", color: "#434655", size: 11 },
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    margin: { l: 46, r: 12, t: 28, b: 40 },
  },
  accent: "#004ac6",
  teal: "#0D9488",
  grey: "#c3c6d7",
  xaxis: { gridcolor: "#e5eeff", zeroline: false },
  yaxis: { gridcolor: "#e5eeff", zeroline: false },
};

async function renderAnalytics() {
  if (analyticsRendered) return;
  let data;
  try {
    const res = await fetch("/api/v1/analytics");
    if (!res.ok) throw new Error("analytics unavailable");
    data = await res.json();
  } catch (err) {
    $("#chart-models").innerHTML =
      `<div class="text-sm text-on-surface-variant p-10 text-center h-full grid place-items-center">${err.message} — run the module scripts to generate artifacts.</div>`;
    return;
  }
  const ml = data.ml.rows;
  const famColor = (f) => (f === "deep" ? PLOT.accent : PLOT.teal);

  Plotly.newPlot("chart-models", [
    {
      y: ml.map((r) => r.accuracy), name: "Accuracy",
      marker: { color: ml.map((r) => famColor(r.family)), opacity: 0.25 },
    },
    { y: ml.map((r) => r.f1), name: "F1 (macro)", marker: { color: ml.map((r) => famColor(r.family)) } },
  ].map((trace) => ({ ...trace, x: ml.map((r) => r.model), type: "bar", hovertemplate: "%{x}: %{y:.3f}<extra>" })), {
    ...PLOT.base,
    barmode: "group",
    showlegend: true,
    legend: { orientation: "h", y: 1.12 },
    xaxis: { ...PLOT.xaxis, tickangle: -30 },
    yaxis: { ...PLOT.yaxis, range: [0, 1] },
  });

  const trans = data.translation;
  if (trans.length) {
    Plotly.newPlot("chart-bleu", [
      { x: trans.map((t) => t.direction), y: trans.map((t) => t.rule_baseline.bleu4), name: "Rule baseline", type: "bar", marker: { color: PLOT.grey } },
      { x: trans.map((t) => t.direction), y: trans.map((t) => t.ai.bleu4), name: "NLLB (AI)", type: "bar", marker: { color: PLOT.accent } },
    ], {
      ...PLOT.base,
      barmode: "group",
      legend: { orientation: "h", y: 1.12 },
      xaxis: PLOT.xaxis,
      yaxis: { ...PLOT.yaxis, title: "BLEU-4" },
    });

    Plotly.newPlot("chart-latency", [
      { x: trans.map((t) => t.direction), y: trans.map((t) => t.ai.latency_ms.p50), name: "p50", type: "bar", marker: { color: PLOT.teal } },
      { x: trans.map((t) => t.direction), y: trans.map((t) => t.ai.latency_ms.p95), name: "p95", type: "bar", marker: { color: PLOT.accent } },
    ], {
      ...PLOT.base,
      barmode: "group",
      legend: { orientation: "h", y: 1.12 },
      xaxis: PLOT.xaxis,
      yaxis: { ...PLOT.yaxis, title: "ms" },
    });
  } else {
    $("#chart-bleu").innerHTML =
      `<div class="text-sm text-on-surface-variant p-10 text-center h-full grid place-items-center">No translation evaluation artifacts.</div>`;
    $("#chart-latency").innerHTML = "";
  }

  const sat = data.satisfaction;
  const total = sat.count || 1;
  Plotly.newPlot("chart-satisfaction", [
    { labels: ["Positive", "Neutral", "Negative"],
      values: [sat.positive, total - sat.positive - sat.negative, sat.negative],
      type: "pie", hole: 0.62,
      marker: { colors: ["#0D9488", "#e5eeff", "#ba1a1a"] },
      textinfo: "label+percent", hoverinfo: "label+value" },
  ], { ...PLOT.base, showlegend: false });

  $("#analytics-note").textContent =
    `Artifacts loaded: ${data.sources.join(", ")}` +
    (data.generated_at ? ` · generated ${new Date(data.generated_at).toLocaleString()}` : "");
  analyticsRendered = true;
}

/* ---------- helpers ---------- */

function escapeHtml(text) {
  return String(text ?? "")
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;").replace(/'/g, "&#39;");
}

function time(iso) {
  if (!iso) return "";
  const d = new Date(iso);
  return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

function latency(ms) {
  if (ms == null) return "n/a";
  return ms < 1000 ? `${ms}ms` : `${(ms / 1000).toFixed(1)}s`;
}

/* ---------- boot ---------- */

$("#join-lang").innerHTML = LANGS.map(
  ([v, l]) => `<option value="${v}" ${v === "hi" ? "selected" : ""}>${l}</option>`).join("");
$("#pref-lang").addEventListener("change", () => {
  targetLang = $("#pref-lang").value;
  $("#join-lang").value = targetLang;
});

renderOverview();
const tab = location.hash.slice(1);
showView(["overview", "live", "history", "analytics"].includes(tab) ? tab : "overview", { push: false });