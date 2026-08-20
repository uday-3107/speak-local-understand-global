export function connectWebSocket(sessionId, handlers = {}) {
  const protocol = window.location.protocol === "https:" ? "wss" : "ws";
  const url = `${protocol}://${window.location.host}/ws/${sessionId}`;
  const ws = new WebSocket(url);

  ws.addEventListener("open", () => handlers.onOpen?.());
  ws.addEventListener("message", (event) => {
    let msg;
    try {
      msg = JSON.parse(event.data);
    } catch {
      return;
    }
    handlers.onMessage?.(msg);
  });
  ws.addEventListener("close", () => handlers.onClose?.());

  return ws;
}