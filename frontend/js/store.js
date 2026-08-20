const state = {
  sourceLang: "en",
  targetLang: "hi",
  sessionId: null,
  status: "idle",
  captions: [],
  isRecording: false,
};

const listeners = new Set();

export function getState() {
  return state;
}

export function subscribe(fn) {
  listeners.add(fn);
  return () => listeners.delete(fn);
}

export function update(patch) {
  Object.assign(state, patch);
  listeners.forEach((fn) => fn());
}
