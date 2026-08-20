import asyncio
from typing import Any

from fastapi import WebSocket


class ConnectionManager:
    """Registry of live WebSocket connections for broadcast to clients."""

    def __init__(self) -> None:
        self.active: list[WebSocket] = []

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self.active.append(websocket)

    def disconnect(self, websocket: WebSocket) -> None:
        if websocket in self.active:
            self.active.remove(websocket)

    async def broadcast(self, message: dict[str, Any]) -> None:
        disconnected = []
        for ws in self.active:
            try:
                await ws.send_json(message)
            except Exception:
                disconnected.append(ws)
        for ws in disconnected:
            self.disconnect(ws)

    async def send(self, websocket: WebSocket, message: dict[str, Any]) -> None:
        try:
            await websocket.send_json(message)
        except Exception:
            self.disconnect(websocket)

    async def pulse(self, interval: float = 10.0) -> None:
        while True:
            await asyncio.sleep(interval)
            await self.broadcast({"type": "ping", "ts": asyncio.get_event_loop().time()})


manager = ConnectionManager()