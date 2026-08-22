import { createSession, downloadRecordingUrl, endSession, listRecordings, startSession } from "./api.js";
import { connectWebSocket } from "./ws.js";
import { formatLatency, formatTime } from "./utils.js";

let ws = null;
let wsStatus = "idle";
let mediaRecorder = null;
let mediaStream = null;
let captions = [];
const interims = new Map(); // interim_id -> preview caption (replaced by final)
let activeSessionId = null;
let stopped = false;

export function initLectureView(root) {
  root.innerHTML = `
    <div class="panel controls">
      <button class="btn" id="generate-code">Generate code</button>
      <div id="code-form" hidden>
        <label class="field">
          <span>Lecture language</span>
          <select id="src-lang">
            <option value="en">English</option>
            <option value="hi">Hindi (हिन्दी)</option>
            <option value="te">Telugu (తెలుగు)</option>
          </select>
        </label>
        <label class="field">
          <span>Translate to</span>
          <select id="tgt-lang">
            <option value="hi">Hindi (हिन्दी)</option>
            <option value="en">English</option>
            <option value="te">Telugu (తెలుగు)</option>
          </select>
        </label>
        <label class="field">
          <span>Subject</span>
          <input type="text" id="subject" placeholder="e.g. Biology" />
        </label>
        <button class="btn" id="create-session">Create session</button>
      </div>
      <button class="btn" id="start-session" hidden>Start session</button>
      <button class="btn btn-danger" id="stop-session" hidden>Stop session</button>
      <button class="btn" id="download-recording" hidden>Download recording</button>
      <span id="ws-status" class="status-dot idle"></span>
    </div>

    <div class="mic-area" id="mic-area" hidden>
      <button class="mic-btn" id="mic" title="Toggle microphone" aria-label="Toggle microphone">
        <svg viewBox="0 0 24 24">
          <rect x="9" y="2" width="6" height="12" rx="3" />
          <path d="M5 10a7 7 0 0 0 14 0" />
          <line x1="12" y1="19" x2="12" y2="22" />
        </svg>
      </button>
      <span id="mic-hint" class="mic-hint"></span>
      <span id="mic-error" class="mic-error"></span>
    </div>

    <div class="panel caption-box empty" id="captions">
      <p>Generate a code, start the session, then speak — captions will appear here.</p>
    </div>
  `;

  const srcLang = root.querySelector("#src-lang");
  const tgtLang = root.querySelector("#tgt-lang");
  const subject = root.querySelector("#subject");
  const generateBtn = root.querySelector("#generate-code");
  const codeForm = root.querySelector("#code-form");
  const createBtn = root.querySelector("#create-session");
  const startBtn = root.querySelector("#start-session");
  const stopBtn = root.querySelector("#stop-session");
  const downloadBtn = root.querySelector("#download-recording");
  const statusEl = root.querySelector("#ws-status");
  const micBtn = root.querySelector("#mic");
  const micArea = root.querySelector("#mic-area");
  const hintEl = root.querySelector("#mic-hint");
  const errorEl = root.querySelector("#mic-error");
  const captionsEl = root.querySelector("#captions");

  const renderStatus = () => {
    statusEl.className = `status-dot ${wsStatus}`;
    statusEl.textContent =
      wsStatus === "connected" ? "● connected" : `● ${wsStatus}`;
  };

  const renderCaptions = () => {
    if (captions.length === 0 && interims.size === 0) {
      captionsEl.className = "panel caption-box empty";
      captionsEl.innerHTML = "<p>Generate a code, start the session, then speak — captions will appear here.</p>";
      return;
    }
    const interimItems = [...interims.values()]
      .map(
        (c) => `
        <li class="caption-item" data-interim="${c.interimId}" style="opacity:0.55">
          <p class="caption-translated"></p>
          <p class="caption-source"></p>
          <p class="caption-meta"></p>
        </li>`,
      )
      .join("");
    captionsEl.className = "panel caption-box";
    captionsEl.innerHTML = `<ul class="caption-list">${captions
      .map(
        (c, i) => `
        <li class="caption-item" data-idx="${i}">
          <p class="caption-translated"></p>
          <p class="caption-source"></p>
          <p class="caption-meta"></p>
        </li>`,
      )
      .join("")}${interimItems}</ul>`;

    captions.forEach((c, i) => {
      const item = captionsEl.querySelector(`[data-idx="${i}"]`);
      item.querySelector(".caption-translated").textContent = c.translated_text;
      item.querySelector(".caption-source").textContent = c.source_text;
      item.querySelector(".caption-meta").textContent =
        `${formatTime(c.timestamp)} · ${c.model_used} · ${formatLatency(c.latency_ms)}`;
    });
    interims.forEach((c) => {
      const item = captionsEl.querySelector(`[data-interim="${c.interimId}"]`);
      if (!item) return;
      item.querySelector(".caption-translated").textContent = c.translated_text;
      item.querySelector(".caption-source").textContent = c.source_text;
      item.querySelector(".caption-meta").textContent = "listening…";
    });
    captionsEl.scrollTop = captionsEl.scrollHeight;
  };

  generateBtn.addEventListener("click", () => {
    generateBtn.hidden = true;
    codeForm.hidden = false;
  });

  createBtn.addEventListener("click", async () => {
    try {
      const session = await createSession(
        subject.value.trim(),
        srcLang.value,
        tgtLang.value,
      );
      activeSessionId = session.id;
      srcLang.disabled = true;
      tgtLang.disabled = true;
      subject.disabled = true;
      codeForm.hidden = true;
      micArea.hidden = false;
      startBtn.hidden = false;
      window.dispatchEvent(
        new CustomEvent("session-created", { detail: session.join_code }),
      );
    } catch (err) {
      statusEl.className = "status-dot disconnected";
      statusEl.textContent = `● error: ${err.message}`;
    }
  });

  startBtn.addEventListener("click", async () => {
    try {
      await startSession(activeSessionId);
    } catch {
      /* session may already be started — treat as idempotent */
    }
    stopped = false;
    pendingPcm = [];
    pendingHasSpeech = false;
    startBtn.hidden = true;
    stopBtn.hidden = false;
    openSocket(activeSessionId, tgtLang.value);
  });

  stopBtn.addEventListener("click", async () => {
    if (mediaRecorder) stopCapture();
    stopped = true;
    closeSocket();
    const sid = activeSessionId;
    if (sid) {
      try {
        await endSession(sid);
      } catch {
        /* already ended / offline: nothing to do */
      }
    }
    srcLang.disabled = false;
    tgtLang.disabled = false;
    subject.disabled = false;
    stopBtn.hidden = true;
    startBtn.hidden = false;
    wsStatus = "idle";
    renderStatus();
    if (sid) {
      const ready = await waitForRecording(sid);
      if (ready) downloadBtn.hidden = false;
      else errorEl.textContent = "No recording was captured for this session.";
    }
  });

  async function waitForRecording(sessionId, timeoutMs = 10000) {
    const deadline = Date.now() + timeoutMs;
    while (Date.now() < deadline) {
      try {
        const recs = await listRecordings(sessionId);
        if (recs.length > 0) return true;
      } catch {
        /* retry */
      }
      await new Promise((resolve) => setTimeout(resolve, 400));
    }
    return false;
  }

  downloadBtn.addEventListener("click", () => {
    if (!activeSessionId) return;
    const a = document.createElement("a");
    a.href = downloadRecordingUrl(activeSessionId);
    a.download = "recording.wav";
    document.body.appendChild(a);
    a.click();
    a.remove();
  });

  function openSocket(sessionId, target) {
    stopped = false;
    closeSocket();
    wsStatus = "connecting";
    renderStatus();
    ws = connectWebSocket(sessionId, {
      onOpen: () => {
        wsStatus = "connected";
        renderStatus();
      },
      onMessage: (msg) => {
        if (msg.type === "segment") {
          if (msg.interim && msg.interim_id) {
            // Live preview while the lecturer is still speaking.
            interims.set(msg.interim_id, {
              interimId: msg.interim_id,
              ...msg.payload,
              timestamp: new Date().toISOString(),
            });
            renderCaptions();
            return;
          }
          // Final segment: drop the preview it finalizes.
          if (msg.interim_id && interims.has(msg.interim_id)) {
            interims.delete(msg.interim_id);
          }
          const caption = {
            id: msg.payload.id ?? crypto.randomUUID(),
            ...msg.payload,
            timestamp: new Date().toISOString(),
          };
          captions = [...captions, caption];
          renderCaptions();
        } else if (msg.type === "error") {
          errorEl.textContent = msg.message;
        }
      },
      onClose: () => {
        if (!stopped) {
          wsStatus = "disconnected";
          renderStatus();
        }
      },
    });
  }

  function closeSocket() {
    if (ws) {
      ws.close();
      ws = null;
    }
  }

  micBtn.addEventListener("click", async () => {
    if (mediaRecorder) {
      stopCapture();
      return;
    }

    errorEl.textContent = "";
    try {
      mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
      startCapture();
      micBtn.classList.add("recording");
      hintEl.textContent = "Listening…";
    } catch (err) {
      errorEl.textContent = err.message || "microphone unavailable";
    }
  });

  let audioCtx = null;
  let sourceNode = null;
  let processor = null;
  let silentGain = null;
  let pendingPcm = [];
  let pendingHasSpeech = false;
  let checkTimer = null;
  let inSilence = true;
  let silenceStart = Date.now();

  function startCapture() {
    audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    sourceNode = audioCtx.createMediaStreamSource(mediaStream);
    silentGain = audioCtx.createGain();
    silentGain.gain.value = 0;
    processor = audioCtx.createScriptProcessor(4096, 1, 1);
    processor.onaudioprocess = (event) => {
      const input = event.inputBuffer.getChannelData(0);
      let energy = 0;
      for (let i = 0; i < input.length; i++) energy += input[i] * input[i];
      const rms = Math.sqrt(energy / input.length);
      if (rms >= 0.02) pendingHasSpeech = true;
      if (rms < 0.008) {
        if (!inSilence) {
          inSilence = true;
          silenceStart = Date.now();
        }
      } else {
        inSilence = false;
      }
      pendingPcm.push(resampleTo16k(input, audioCtx.sampleRate));
    };
    sourceNode.connect(processor);
    processor.connect(silentGain);
    silentGain.connect(audioCtx.destination);
    mediaRecorder = {};
    checkTimer = setInterval(checkFlush, 250);
  }

  function stopCapture() {
    clearInterval(checkTimer);
    checkTimer = null;
    flushPcm();
    if (processor) {
      processor.disconnect();
      processor.onaudioprocess = null;
      processor = null;
    }
    if (sourceNode) {
      sourceNode.disconnect();
      sourceNode = null;
    }
    if (silentGain) {
      silentGain.disconnect();
      silentGain = null;
    }
    if (audioCtx) {
      audioCtx.close();
      audioCtx = null;
    }
    mediaStream.getTracks().forEach((t) => t.stop());
    mediaStream = null;
    mediaRecorder = null;
    micBtn.classList.remove("recording");
    hintEl.textContent = "";
  }

  function pendingSeconds() {
    return pendingPcm.reduce((n, p) => n + p.length, 0) / 16000;
  }

  function checkFlush() {
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    const secs = pendingSeconds();
    if (secs < 0.3) return;
    if (!pendingHasSpeech) {
      if (secs > 4) pendingPcm = []; // drop long pure-silence accumulation
      return;
    }
    if (inSilence && Date.now() - silenceStart >= 500) {
      if (secs >= 2.0) flushPcm();
      else if (Date.now() - silenceStart >= 2000) flushPcm();
    } else if (!inSilence && secs >= 4.0) {
      flushPcm();
    }
  }

  function resampleTo16k(input, fromRate) {
    const ratio = fromRate / 16000;
    const outLen = Math.floor(input.length / ratio);
    const out = new Int16Array(outLen);
    for (let i = 0; i < outLen; i++) {
      const idx = i * ratio;
      const i0 = Math.floor(idx);
      const frac = idx - i0;
      const i1 = Math.min(i0 + 1, input.length - 1);
      const sample = input[i0] * (1 - frac) + input[i1] * frac;
      out[i] = Math.max(-1, Math.min(1, sample)) * 32767;
    }
    return out;
  }

  function buildWav(pcm) {
    const buffer = new ArrayBuffer(44 + pcm.length * 2);
    const view = new DataView(buffer);
    const writeStr = (offset, str) => {
      for (let i = 0; i < str.length; i++) view.setUint8(offset + i, str.charCodeAt(i));
    };
    writeStr(0, "RIFF");
    view.setUint32(4, 36 + pcm.length * 2, true);
    writeStr(8, "WAVE");
    writeStr(12, "fmt ");
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);
    view.setUint16(22, 1, true);
    view.setUint32(24, 16000, true);
    view.setUint32(28, 32000, true);
    view.setUint16(32, 2, true);
    view.setUint16(34, 16, true);
    writeStr(36, "data");
    view.setUint32(40, pcm.length * 2, true);
    new Int16Array(buffer, 44).set(pcm);
    return new Blob([buffer], { type: "audio/wav" });
  }

  function flushPcm() {
    if (pendingPcm.length === 0 || !ws || ws.readyState !== WebSocket.OPEN) return;
    if (!pendingHasSpeech) return;
    pendingHasSpeech = false;
    const total = pendingPcm.reduce((n, p) => n + p.length, 0);
    const merged = new Int16Array(total);
    let offset = 0;
    for (const part of pendingPcm) {
      merged.set(part, offset);
      offset += part.length;
    }
    pendingPcm = [];
    const reader = new FileReader();
    reader.onload = () => {
      const base64 = reader.result.split(",")[1];
      ws.send(
        JSON.stringify({ type: "audio", data: base64, language: srcLang.value, target: tgtLang.value }),
      );
    };
    reader.readAsDataURL(buildWav(merged));
  }

  renderStatus();
  renderCaptions();
  return {
    cleanup() {
      closeSocket();
      if (mediaRecorder) stopCapture();
      captions = [];
      interims.clear();
    },
  };
}
