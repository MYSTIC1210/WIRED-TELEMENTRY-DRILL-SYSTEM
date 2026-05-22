"""
dashboard.py — Real-time Plotly Dash dashboard.
Polls REST API every 2s and renders live KPI tiles + time-series charts.
"""

import time
from datetime import datetime

import plotly.graph_objs as go
import requests
from dash import Dash, Input, Output, dcc, html

API_BASE   = "http://localhost:8000"
POLL_MS    = 2000

app = Dash(__name__, title="DrillWatch — Live Telemetry")

# ─── Layout ───────────────────────────────────────────────────────────────────

app.layout = html.Div(
    style={"backgroundColor": "#0a0e14", "minHeight": "100vh", "fontFamily": "monospace", "color": "#c5ddf0"},
    children=[
        html.H2("⛏ DRILL TELEMETRY — LIVE MONITORING",
                style={"textAlign": "center", "color": "#f59e0b", "padding": "16px 0 8px"}),

        # KPI row
        html.Div(id="kpi-row", style={"display": "flex", "gap": "12px", "padding": "0 24px 16px"}),

        # Charts
        html.Div(style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "12px", "padding": "0 24px"}, children=[
            dcc.Graph(id="chart-rpm",   style={"backgroundColor": "#101520"}),
            dcc.Graph(id="chart-vib",   style={"backgroundColor": "#101520"}),
            dcc.Graph(id="chart-temp",  style={"backgroundColor": "#101520"}),
            dcc.Graph(id="chart-torq",  style={"backgroundColor": "#101520"}),
        ]),

        # Alert feed
        html.H4("⚠ ALERT FEED", style={"padding": "16px 24px 4px", "color": "#ef4444"}),
        html.Div(id="alert-feed", style={"padding": "0 24px 24px"}),

        dcc.Interval(id="interval", interval=POLL_MS, n_intervals=0),
    ],
)

# ─── Helpers ──────────────────────────────────────────────────────────────────

CHART_LAYOUT = dict(
    paper_bgcolor="#101520", plot_bgcolor="#0d1118",
    font=dict(color="#8bb8d4", size=11),
    margin=dict(l=50, r=20, t=36, b=36),
    xaxis=dict(gridcolor="#1c2d44"),
    yaxis=dict(gridcolor="#1c2d44"),
)

def _kpi_tile(label: str, value: str, color: str = "#00c8b4") -> html.Div:
    return html.Div(style={
        "flex": "1", "backgroundColor": "#0f1520", "border": f"1px solid {color}33",
        "borderRadius": "6px", "padding": "12px", "textAlign": "center",
    }, children=[
        html.Div(label, style={"fontSize": "11px", "color": "#8bb8d4", "marginBottom": "4px"}),
        html.Div(value, style={"fontSize": "22px", "fontWeight": "bold", "color": color}),
    ])


def _fetch(endpoint: str, params: dict = None):
    try:
        r = requests.get(f"{API_BASE}{endpoint}", params=params, timeout=1)
        return r.json()
    except Exception:
        return []


def _ts_chart(records, y_field, title, color):
    if not records:
        return go.Figure(layout={**CHART_LAYOUT, "title": title})
    xs = [datetime.fromtimestamp(r["timestamp"]) for r in records]
    ys = [r.get(y_field, 0) for r in records]
    anomaly_x = [x for x, r in zip(xs, records) if r.get("is_anomaly")]
    anomaly_y = [y for y, r in zip(ys, records) if r.get("is_anomaly")]
    fig = go.Figure(layout={**CHART_LAYOUT, "title": {"text": title, "font": {"color": "#c5ddf0"}}})
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=color, width=1.5), name=y_field))
    if anomaly_x:
        fig.add_trace(go.Scatter(x=anomaly_x, y=anomaly_y, mode="markers",
                                  marker=dict(color="#ef4444", size=8, symbol="x"), name="Anomaly"))
    return fig


# ─── Callbacks ────────────────────────────────────────────────────────────────

@app.callback(
    Output("kpi-row",    "children"),
    Output("chart-rpm",  "figure"),
    Output("chart-vib",  "figure"),
    Output("chart-temp", "figure"),
    Output("chart-torq", "figure"),
    Output("alert-feed", "children"),
    Input("interval",    "n_intervals"),
)
def update(_):
    records = _fetch("/telemetry", {"n": 120})
    alerts  = _fetch("/alerts",    {"n": 8})
    health  = _fetch("/health") or {}

    latest = records[-1] if records else {}

    kpis = [
        _kpi_tile("RPM",         f"{latest.get('rpm', '--'):.1f}" if latest else "--", "#00c8b4"),
        _kpi_tile("TORQUE (Nm)", f"{latest.get('torque_nm', '--'):.0f}" if latest else "--", "#0ea5e9"),
        _kpi_tile("TEMP (°C)",   f"{latest.get('temperature_c', '--'):.1f}" if latest else "--", "#f59e0b"),
        _kpi_tile("VIB (g)",     f"{latest.get('vibration_g', '--'):.3f}" if latest else "--",
                  "#ef4444" if latest.get("is_anomaly") else "#22c55e"),
        _kpi_tile("WS CLIENTS",  str(health.get("ws_clients", 0)), "#a855f7"),
        _kpi_tile("ALERTS",      str(len(alerts)), "#ef4444"),
    ]

    alert_rows = []
    for a in reversed(alerts):
        ts  = datetime.fromtimestamp(a["timestamp"]).strftime("%H:%M:%S")
        alert_rows.append(html.Div(
            f"[{ts}] {a['sensor_id']} — {a['detail']} (conf={a['confidence']:.0%})",
            style={"padding": "4px 0", "borderBottom": "1px solid #1c2d44", "color": "#ef4444", "fontSize": "13px"},
        ))
    if not alert_rows:
        alert_rows = [html.Div("No alerts.", style={"color": "#22c55e"})]

    return (
        kpis,
        _ts_chart(records, "rpm",           "RPM",            "#00c8b4"),
        _ts_chart(records, "vibration_g",   "Vibration (g)",  "#ef4444"),
        _ts_chart(records, "temperature_c", "Temperature °C", "#f59e0b"),
        _ts_chart(records, "torque_nm",     "Torque (Nm)",    "#0ea5e9"),
        alert_rows,
    )


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=8050)
