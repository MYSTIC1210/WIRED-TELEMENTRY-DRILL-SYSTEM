"""
api.py — FastAPI backend: REST endpoints + WebSocket push for the dashboard.
Maintains an in-memory ring buffer of the last 200 telemetry frames and
50 alerts; broadcasts new data to all connected WebSocket clients.
"""

import asyncio
import collections
import time
from typing import List

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Drill Telemetry API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── In-memory stores ─────────────────────────────────────────────────────────
telemetry_buf: collections.deque = collections.deque(maxlen=200)
alerts_buf:    collections.deque = collections.deque(maxlen=50)
ws_clients:    List[WebSocket]   = []


# ─── Models ───────────────────────────────────────────────────────────────────

class TelemetryPayload(BaseModel):
    sensor_id:     str
    timestamp:     float
    depth_m:       float
    rpm:           float
    torque_nm:     float
    temperature_c: float
    vibration_g:   float
    wob:           float
    flow_rate:     float
    is_anomaly:    bool = False
    confidence:    float = 0.0
    detail:        str = "NOMINAL"


class AlertPayload(BaseModel):
    sensor_id:  str
    timestamp:  float
    detail:     str
    confidence: float
    scores:     dict


# ─── REST endpoints ───────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status": "ok",
        "telemetry_buffered": len(telemetry_buf),
        "alerts_buffered":    len(alerts_buf),
        "ws_clients":         len(ws_clients),
    }


@app.get("/telemetry")
def get_telemetry(n: int = 100):
    return list(telemetry_buf)[-n:]


@app.get("/alerts")
def get_alerts(n: int = 20):
    return list(alerts_buf)[-n:]


@app.post("/ingest/telemetry", status_code=204)
async def ingest_telemetry(payload: TelemetryPayload):
    record = payload.dict()
    telemetry_buf.append(record)
    await _broadcast({"type": "telemetry", "data": record})


@app.post("/ingest/alert", status_code=204)
async def ingest_alert(payload: AlertPayload):
    record = payload.dict()
    alerts_buf.append(record)
    await _broadcast({"type": "alert", "data": record})


# ─── WebSocket ────────────────────────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    ws_clients.append(ws)
    try:
        # Send current snapshot on connect
        await ws.send_json({
            "type":      "snapshot",
            "telemetry": list(telemetry_buf)[-50:],
            "alerts":    list(alerts_buf)[-10:],
        })
        while True:
            await asyncio.sleep(30)   # keep-alive
            await ws.send_json({"type": "ping"})
    except WebSocketDisconnect:
        ws_clients.remove(ws)


async def _broadcast(message: dict) -> None:
    dead = []
    for ws in ws_clients:
        try:
            await ws.send_json(message)
        except Exception:
            dead.append(ws)
    for ws in dead:
        ws_clients.remove(ws)
