from __future__ import annotations

import asyncio
import contextlib
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from web_runtime import WebSimulationSession


ROOT = Path(__file__).resolve().parent
STATIC = ROOT / "web_static"
app = FastAPI(title="Magnetic Rescue Robot Simulator")
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


async def receive_commands(websocket: WebSocket, queue: asyncio.Queue) -> None:
    while True:
        await queue.put(await websocket.receive_json())


@app.websocket("/ws")
async def simulation_socket(websocket: WebSocket) -> None:
    await websocket.accept()
    queue: asyncio.Queue = asyncio.Queue()
    receiver = asyncio.create_task(receive_commands(websocket, queue))
    try:
        session = await asyncio.to_thread(WebSimulationSession)
        await websocket.send_json(await asyncio.to_thread(session.snapshot, True))
        while True:
            while not queue.empty():
                session.apply_command(queue.get_nowait())
            if receiver.done():
                receiver.result()
            if session.running:
                await asyncio.to_thread(session.step)
            include_map = session.frame % 4 == 0 or not session.running
            state = await asyncio.to_thread(session.snapshot, include_map)
            await websocket.send_json(state)
            await asyncio.sleep(session.tick_delay() if session.running else 0.25)
    except (WebSocketDisconnect, RuntimeError, asyncio.CancelledError):
        pass
    finally:
        receiver.cancel()
        with contextlib.suppress(asyncio.CancelledError, WebSocketDisconnect):
            await receiver

