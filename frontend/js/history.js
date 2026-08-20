import { listSessions } from "./api.js";
import { languageLabel } from "./utils.js";

export function initHistoryView(root) {
  root.innerHTML = `<div class="panel" id="history-panel">
    <p class="empty-note">Loading sessions…</p>
  </div>`;
  load(root.querySelector("#history-panel"));
}

async function load(panel) {
  try {
    const sessions = await listSessions();
    if (sessions.length === 0) {
      panel.innerHTML = '<p class="empty-note">No sessions yet.</p>';
      return;
    }
    panel.innerHTML = `
      <table>
        <thead>
          <tr><th>Subject</th><th>Language</th><th>Started</th><th>Status</th></tr>
        </thead>
        <tbody>
          ${sessions
            .map(
              (s) => `<tr>
                <td>${escapeHtml(s.subject || "Untitled")}</td>
                <td>${languageLabel(s.source_lang)}</td>
                <td>${new Date(s.started_at).toLocaleString()}</td>
                <td>${s.ended_at ? "ended" : "live"}</td>
              </tr>`,
            )
            .join("")}
        </tbody>
      </table>`;
  } catch (err) {
    panel.innerHTML = `<p class="empty-note">Failed to load sessions: ${escapeHtml(err.message)}</p>`;
  }
}

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, (ch) => {
    const map = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" };
    return map[ch];
  });
}