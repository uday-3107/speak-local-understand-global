import { listSessions } from "./api.js";

export function initDashboardView(root) {
  const stats = [
    { key: "sessions", label: "Sessions" },
    { key: "segments", label: "Segments translated" },
    { key: "avg_latency", label: "Avg latency" },
    { key: "bleu", label: "BLEU score" },
  ];

  root.innerHTML = `
    <h2 style="font-size:1.25rem;margin-bottom:1rem">Translation analytics</h2>
    <div class="stats-grid">
      ${stats
        .map(
          (s) => `
        <div class="stat-card">
          <p class="stat-label">${s.label}</p>
          <p class="stat-value" data-stat="${s.key}">—</p>
        </div>`,
        )
        .join("")}
    </div>
    <p class="empty-note">Module 8 metrics (BLEU, model comparison, latency) will populate here.</p>
  `;

  (async () => {
    try {
      const sessions = await listSessions();
      const totalSegments = await Promise.all(
        sessions.map(async (s) => (await segmentsFor(s.id)).length),
      ).then((counts) => counts.reduce((a, b) => a + b, 0));
      root.querySelector('[data-stat="sessions"]').textContent = sessions.length;
      root.querySelector('[data-stat="segments"]').textContent = totalSegments;
    } catch {
      /* leave placeholders */
    }
  })();
}

async function segmentsFor(sessionId) {
  const res = await fetch(`/api/v1/sessions/${sessionId}/segments`);
  return res.ok ? res.json() : [];
}