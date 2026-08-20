import { initLectureView } from "./lecture.js";
import { listSegments, listSessions } from "./api.js";

const $ = (sel) => document.querySelector(sel);

initLectureView($("#studio"));

/* ---------- live stats (read-only observer of the session we created) ---------- */

let statsTimer = null;
let codeBannerShown = false;
let sessionCode = null;

function showCodeBanner(code) {
  if (!code || codeBannerShown) return;
  codeBannerShown = true;
  sessionCode = code;
  $("#code-banner").classList.remove("hidden");
  $("#code-banner").classList.add("flex");
  $("#banner-code").textContent = code;
}

window.addEventListener("session-created", (e) => showCodeBanner(e.detail));

async function refreshStats() {
  let sessions = [];
  try {
    sessions = (await listSessions())
      .filter((s) => !s.ended_at)
      .sort((a, b) => new Date(b.started_at) - new Date(a.started_at));
  } catch {
    return;
  }
  // only follow the session created on THIS page; ignore leftover sessions
  const current = sessionCode
    ? sessions.find((s) => s.join_code === sessionCode)
    : null;
  if (!current) return;

  let segments;
  try {
    segments = await listSegments(current.id);
  } catch {
    segments = [];
  }
  $("#stat-segments").textContent = segments.length;

  const start = new Date(current.started_at);
  const stop = current.ended_at ? new Date(current.ended_at) : Date.now();
  const secs = Math.max(0, Math.floor((stop - start.getTime()) / 1000));
  const m = Math.floor(secs / 60);
  const s = secs % 60;
  $("#stat-duration").textContent = `${m}:${String(s).padStart(2, "0")}`;

  const latencies = segments.filter((sg) => sg.latency_ms > 0).map((sg) => sg.latency_ms);
  if (latencies.length) {
    const avg = Math.round(latencies.reduce((n, v) => n + v, 0) / latencies.length);
    $("#stat-latency").textContent = avg < 1000 ? `${avg}ms` : `${(avg / 1000).toFixed(1)}s`;
  } else {
    $("#stat-latency").textContent = "–";
  }
}

/* ---------- AI lecturer assistant (demo wiring; live Ollama at test time) ---------- */

setInterval(refreshStats, 3000);
refreshStats();

$("#copy-code").addEventListener("click", async () => {
  const code = $("#banner-code").textContent;
  try {
    await navigator.clipboard.writeText(code);
    $("#copy-code").innerHTML = `<span class="material-symbols-outlined text-[18px]">check</span> Copied`;
    setTimeout(() => {
      $("#copy-code").innerHTML = `<span class="material-symbols-outlined text-[18px]">content_copy</span> Copy`;
    }, 1500);
  } catch {
    window.prompt("Copy the join code:", code);
  }
});

async function assistantBot(text) {
  let latest = null;
  let context = [];
  try {
    const sessions = (await listSessions())
      .filter((s) => !s.ended_at)
      .sort((a, b) => new Date(b.started_at) - new Date(a.started_at));
    if (sessions[0]) {
      const segs = await listSegments(sessions[0].id);
      latest = segs[segs.length - 1] ?? null;
      context = segs.slice(-6).map((s) => ({
        source_text: s.source_text,
        source_lang: s.source_lang,
        translated_text: s.translated_text,
        target_lang: s.target_lang,
      }));
    }
  } catch {
    /* offline */
  }
  try {
    const res = await fetch("/api/v1/assistant", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: text, role: "lecturer", context }),
    });
    if (res.ok) {
      const data = await res.json();
      if (data.answer) return data.answer;
    }
  } catch {
    /* fall through to template answers */
  }
  const lower = text.toLowerCase();
  if (lower.includes("last") || lower.includes("explain") || lower.includes("latest")) {
    if (!latest) return "No captions yet — start the session and speak, or check your microphone.";
    return `Your last caption was: "${latest.translated_text}"\nSource: "${latest.source_text}"\n(${latest.target_lang}, ${latest.model_used || "engine"}, ${latest.latency_ms}ms)`;
  }
  if (lower.includes("simpler") || lower.includes("rephrase")) {
    if (!latest) return "Start speaking first — I'll offer a simpler phrasing of your last caption.";
    return `A simpler version of "${latest.translated_text}" could be: keep sentences short, use everyday words, and repeat key terms. (Live rephrasing needs Ollama: \`ollama serve\`.)`;
  }
  if (lower.includes("summary") || lower.includes("summar")) {
    return "The lecturer-side summary will list the key points spoken so far — the offline LLM (Ollama) does this once running.";
  }
  return "The AI assistant is offline right now (Ollama not running). Start it with `ollama serve` for live answers; meanwhile I'll answer from templates.";
}

function chat(message, fromUser = true) {
  const box = $("#assist-msgs");
  const bubble = document.createElement("div");
  bubble.className =
    `max-w-[90%] p-3 rounded-2xl whitespace-pre-line ` +
    (fromUser ? "ml-auto rounded-br-sm bg-primary text-white" : "chat-bubble-in mr-auto rounded-bl-sm");
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