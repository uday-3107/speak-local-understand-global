const BASE = "/api/v1";

async function request(path, options = {}) {
  const res = await fetch(`${BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) {
    let detail = `HTTP ${res.status}`;
    try {
      const body = await res.json();
      detail = body.error?.message ?? detail;
    } catch {
      /* non-JSON error body */
    }
    throw new Error(detail);
  }
  return res.json();
}

export function createSession(subject, sourceLang, targetLang) {
  return request("/sessions", {
    method: "POST",
    body: JSON.stringify({ subject, source_lang: sourceLang, target_lang: targetLang }),
  });
}

export function endSession(sessionId) {
  return request(`/sessions/${sessionId}/end`, { method: "POST" });
}

export function startSession(sessionId) {
  return request(`/sessions/${sessionId}/start`, { method: "POST" });
}

export function listRecordings(sessionId) {
  return request(`/sessions/${sessionId}/recordings`);
}

export function downloadRecordingUrl(sessionId) {
  return `/api/v1/sessions/${sessionId}/recording/download`;
}

export function listSessions() {
  return request("/sessions");
}

export function getSession(sessionId) {
  return request(`/sessions/${sessionId}`);
}

export function listSegments(sessionId) {
  return request(`/sessions/${sessionId}/segments`);
}

export function transcriptUrl(sessionId) {
  return `/api/v1/sessions/${sessionId}/transcript`;
}

export function joinSession(code, targetLang) {
  return request("/sessions/join", {
    method: "POST",
    body: JSON.stringify({ code, target_lang: targetLang }),
  });
}

export function createFeedback(segmentId, rating, comment = "") {
  return request("/sessions/feedback", {
    method: "POST",
    body: JSON.stringify({ segment_id: segmentId, rating, comment }),
  });
}
