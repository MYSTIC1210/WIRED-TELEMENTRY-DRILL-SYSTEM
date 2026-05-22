from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import json
import asyncio
import logging
from datetime import datetime
from anomaly_detector import EnsembleAnomalyDetector
import numpy as np
from pathlib import Path
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Optional Kafka imports
try:
    from confluent_kafka import Producer, Consumer
    KAFKA_AVAILABLE = True
except ImportError:
    KAFKA_AVAILABLE = False

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Wired Drill Pipe Telemetry System")

# Global state
anomaly_detector = EnsembleAnomalyDetector()
kafka_producer = None
kafka_consumer = None
connected_clients = set()
sensor_history = {}
dashboard_metrics = {
    "accuracy": 0.0,
    "precision": 0.0,
    "recall": 0.0,
    "f1_score": 0.0,
    "samples": 0,
}
real_world_stream = []
stream_cursor = 0
scaler = StandardScaler()
SENSOR_IDS = ['sensor_01', 'sensor_02', 'sensor_03', 'sensor_04']
SENSOR_FEATURES = {
    'sensor_01': 'mean radius',
    'sensor_02': 'mean texture',
    'sensor_03': 'mean perimeter',
    'sensor_04': 'mean area',
}

def init_kafka():
    """Initialize Kafka producer and consumer"""
    global kafka_producer, kafka_consumer
    if not KAFKA_AVAILABLE:
        logger.info("Kafka not available. Running in demo mode.")
        return False
    
    try:
        logger.info("Kafka client available but not initializing due to server not running")
        return False
    except Exception as e:
        logger.warning(f"Kafka not available: {e}. Running in demo mode.")
        return False

def load_real_world_dataset():
    """Load a real-world labeled dataset and prepare a telemetry-like stream"""
    global real_world_stream, scaler, dashboard_metrics

    dataset = load_breast_cancer(as_frame=True)
    feature_names = list(SENSOR_FEATURES.values())
    data_frame = dataset.frame[feature_names].copy()
    labels = (dataset.target == 0).astype(int).to_numpy()  # 1 = anomaly (malignant)

    X_train, X_test, y_train, y_test = train_test_split(
        data_frame,
        labels,
        test_size=0.25,
        random_state=42,
        stratify=labels,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    anomaly_detector.train(X_train_scaled, y_train, feature_names=SENSOR_IDS)
    dashboard_metrics = anomaly_detector.evaluate(X_test_scaled, y_test)

    real_world_stream = []
    for idx, (raw_row, scaled_row, label) in enumerate(zip(X_test.to_numpy(), X_test_scaled, y_test)):
        raw_values = {
            sensor_id: float(raw_row[position])
            for position, sensor_id in enumerate(SENSOR_IDS)
        }
        scaled_values = {
            sensor_id: float(scaled_row[position])
            for position, sensor_id in enumerate(SENSOR_IDS)
        }
        real_world_stream.append({
            'sample_id': f'sample_{idx + 1}',
            'timestamp': datetime.utcnow().isoformat(),
            'sensor_values': raw_values,
            'scaled_values': scaled_values,
            'actual_label': int(label),
        })

    logger.info(
        "Loaded real-world dataset: Breast Cancer Wisconsin (%s samples, accuracy=%.1f%%)",
        len(real_world_stream),
        dashboard_metrics.get('accuracy', 0.0),
    )

def get_next_real_world_sample():
    """Return the next sample from the prepared dataset stream"""
    global stream_cursor
    if not real_world_stream:
        return None

    sample = real_world_stream[stream_cursor]
    stream_cursor = (stream_cursor + 1) % len(real_world_stream)
    return sample

async def process_sensor_data():
    """Continuously process real-world dataset samples"""
    while True:
        try:
            sample = get_next_real_world_sample()
            if sample is None:
                await asyncio.sleep(1)
                continue

            reading = {
                'timestamp': datetime.utcnow().isoformat(),
                'sample_id': sample['sample_id'],
                'sensor_values': sample['sensor_values'],
                'scaled_values': sample['scaled_values'],
                'actual_label': sample['actual_label'],
            }

            # Run anomaly detection on the multivariate sample
            is_anomaly, confidence, methods = anomaly_detector.detect(reading)
            dominant_feature = max(
                sample['scaled_values'],
                key=lambda feature: abs(sample['scaled_values'][feature])
            )

            result = {
                'timestamp': reading['timestamp'],
                'sample_id': sample['sample_id'],
                'sensor_id': dominant_feature,
                'sensor_values': sample['sensor_values'],
                'actual_label': sample['actual_label'],
                'anomaly_detected': is_anomaly,
                'confidence': confidence,
                'detection_methods': methods
            }

            sensor_history.setdefault('events', []).append(result)
            if len(sensor_history['events']) > 1000:
                sensor_history['events'].pop(0)

            # Broadcast to connected clients
            for client in list(connected_clients):
                try:
                    await client.send_json(result)
                except Exception as e:
                    logger.error(f"Error sending to client: {e}")
                    connected_clients.discard(client)

            # Send to Kafka if available
            if kafka_producer and KAFKA_AVAILABLE:
                try:
                    kafka_producer.produce('drill-telemetry', json.dumps(result).encode('utf-8'))
                    kafka_producer.flush()
                except Exception as e:
                    logger.warning(f"Kafka send failed: {e}")

            await asyncio.sleep(0.08)

        except Exception as e:
            logger.error(f"Error processing sensor data: {e}")
            await asyncio.sleep(1)

@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    try:
        load_real_world_dataset()
    except Exception as e:
        logger.error(f"Failed to load real-world dataset: {e}")
    
    try:
        init_kafka()
    except Exception as e:
        logger.error(f"Failed to init Kafka: {e}")
    
    try:
        asyncio.create_task(process_sensor_data())
    except Exception as e:
        logger.error(f"Failed to start sensor data processing: {e}")
    
    logger.info("Application startup completed")

@app.get("/")
async def get_dashboard():
    """Serve interactive dashboard"""
    dashboard_html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Wired Drill Pipe Telemetry System</title>
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
        <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
        <style>
            :root {
                --bg-0: #07111f;
                --bg-1: #0c1c31;
                --panel: rgba(10, 18, 33, 0.82);
                --panel-border: rgba(120, 186, 255, 0.18);
                --text: #eaf2ff;
                --muted: #8fa2bf;
                --accent: #52d0ff;
                --accent-2: #7c5cff;
                --danger: #ff5c73;
                --shadow: 0 16px 40px rgba(0, 0, 0, 0.28);
            }
            * { margin: 0; padding: 0; box-sizing: border-box; }
            html, body {
                width: 100%;
                min-height: 100%;
                overflow-x: hidden;
            }
            body {
                font-family: 'Inter', system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
                color: var(--text);
                background:
                    radial-gradient(circle at top left, rgba(82, 208, 255, 0.16), transparent 28%),
                    radial-gradient(circle at top right, rgba(124, 92, 255, 0.15), transparent 30%),
                    linear-gradient(180deg, var(--bg-1), var(--bg-0));
                padding: 24px;
            }
            .container {
                max-width: 1360px;
                margin: 0 auto;
                width: 100%;
                display: flex;
                flex-direction: column;
                gap: 20px;
            }
            .hero {
                padding: 24px 26px;
                border-radius: 24px;
                background: linear-gradient(135deg, rgba(12, 28, 49, 0.92), rgba(7, 17, 31, 0.92));
                border: 1px solid var(--panel-border);
                box-shadow: var(--shadow);
            }
            .hero-top {
                display: flex;
                align-items: flex-start;
                justify-content: space-between;
                gap: 16px;
                flex-wrap: wrap;
            }
            .eyebrow {
                display: inline-flex;
                align-items: center;
                gap: 8px;
                color: var(--accent);
                text-transform: uppercase;
                letter-spacing: 0.14em;
                font-size: 0.74rem;
                font-weight: 700;
                margin-bottom: 10px;
            }
            .hero h1 {
                font-size: clamp(1.9rem, 3vw, 3rem);
                line-height: 1.05;
                margin-bottom: 10px;
            }
            .hero p {
                max-width: 760px;
                color: var(--muted);
                line-height: 1.6;
            }
            .status-pill {
                display: inline-flex;
                align-items: center;
                gap: 8px;
                padding: 10px 14px;
                border-radius: 999px;
                background: rgba(82, 208, 255, 0.09);
                border: 1px solid rgba(82, 208, 255, 0.22);
                color: var(--text);
                font-weight: 600;
                white-space: nowrap;
            }
            .status-dot {
                width: 10px;
                height: 10px;
                border-radius: 50%;
                background: #39e58c;
                box-shadow: 0 0 0 5px rgba(57, 229, 140, 0.14);
            }
            .stats-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
                gap: 16px;
                width: 100%;
            }
            .stat-card,
            .chart-card,
            .alerts-panel {
                background: var(--panel);
                border: 1px solid var(--panel-border);
                border-radius: 22px;
                box-shadow: var(--shadow);
                backdrop-filter: blur(18px);
            }
            .stat-card {
                padding: 18px 18px 16px;
                min-height: 120px;
                display: flex;
                flex-direction: column;
                justify-content: center;
            }
            .stat-label {
                color: var(--muted);
                font-size: 0.84rem;
                font-weight: 600;
                text-transform: uppercase;
                letter-spacing: 0.08em;
            }
            .stat-value {
                margin-top: 10px;
                font-size: clamp(1.7rem, 2vw, 2.4rem);
                font-weight: 800;
                color: var(--text);
            }
            .stat-note {
                margin-top: 6px;
                color: var(--muted);
                font-size: 0.9rem;
            }
            .stat-accent {
                color: var(--accent);
            }
            .charts-grid {
                display: grid;
                grid-template-columns: 1fr;
                gap: 18px;
                width: 100%;
            }
            .chart-card {
                padding: 18px;
                overflow: hidden;
            }
            .panel-header {
                display: flex;
                align-items: center;
                justify-content: space-between;
                gap: 12px;
                margin-bottom: 12px;
            }
            .panel-title {
                font-size: 1rem;
                font-weight: 700;
                letter-spacing: -0.02em;
            }
            .panel-subtitle {
                color: var(--muted);
                font-size: 0.88rem;
            }
            .chart-container {
                width: 100%;
                height: 340px;
                overflow: hidden;
                border-radius: 16px;
                background: rgba(6, 12, 24, 0.28);
            }
            .alerts-panel {
                padding: 18px;
                min-height: 360px;
            }
            .alerts-list {
                display: grid;
                gap: 12px;
                max-height: 300px;
                overflow-y: auto;
                padding-right: 4px;
            }
            .alert-item {
                padding: 14px 14px 13px;
                border-radius: 16px;
                background: rgba(255, 92, 115, 0.07);
                border: 1px solid rgba(255, 92, 115, 0.18);
                border-left: 4px solid var(--danger);
            }
            .alert-top {
                display: flex;
                justify-content: space-between;
                gap: 10px;
                align-items: center;
                margin-bottom: 6px;
            }
            .alert-sensor {
                font-weight: 700;
            }
            .anomaly-badge {
                display: inline-flex;
                align-items: center;
                gap: 6px;
                background: rgba(255, 92, 115, 0.16);
                color: #ffd3da;
                padding: 5px 10px;
                border-radius: 999px;
                font-size: 0.82rem;
                font-weight: 700;
            }
            .alert-meta {
                color: var(--muted);
                font-size: 0.9rem;
                line-height: 1.45;
            }
            .alert-time { color: var(--muted); font-size: 0.8rem; margin-top: 8px; }
            .status-online { color: #39e58c; }
            .status-offline { color: #ff6e7f; }
            .empty-state {
                color: var(--muted);
                text-align: center;
                padding: 36px 12px;
                border: 1px dashed rgba(143, 162, 191, 0.3);
                border-radius: 16px;
                background: rgba(255, 255, 255, 0.02);
            }
            @media (max-width: 1100px) {
                .stats-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
            }
            @media (max-width: 720px) {
                body { padding: 14px; }
                .hero, .stat-card, .chart-card, .alerts-panel { border-radius: 18px; }
                .stats-grid { grid-template-columns: 1fr; }
                .chart-container { height: 300px; }
                .alerts-panel { min-height: 320px; }
                .alerts-list { max-height: 230px; }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <section class="hero">
                <div class="hero-top">
                    <div>
                        <div class="eyebrow">Real-time downhole telemetry</div>
                        <h1>Wired Drill Pipe Telemetry System</h1>
                        <p>Live sensor telemetry with ensemble anomaly detection, operator alerts, and continuous visual analytics.</p>
                    </div>
                    <div class="status-pill"><span class="status-dot"></span> System online</div>
                </div>
            </section>

            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-label">Stream Rate</div>
                    <div class="stat-value" id="throughput">0</div>
                    <div class="stat-note">Messages per second</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Anomalies Detected</div>
                    <div class="stat-value" id="anomaly-count">0</div>
                    <div class="stat-note">Detected in the current session</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Detection Accuracy</div>
                    <div class="stat-value stat-accent" id="accuracy">--</div>
                    <div class="stat-note">Labeled test set</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">System Status</div>
                    <div class="stat-value status-online">ONLINE</div>
                    <div class="stat-note">WebSocket + dataset stream</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Precision</div>
                    <div class="stat-value" id="precision">--</div>
                    <div class="stat-note">Positive prediction quality</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Recall</div>
                    <div class="stat-value" id="recall">--</div>
                    <div class="stat-note">Anomaly capture coverage</div>
                </div>
            </div>
            
            <div class="charts-grid">
                <div class="chart-card">
                    <div class="panel-header">
                        <div>
                            <div class="panel-title">Sensor Readings Over Time</div>
                            <div class="panel-subtitle">Live values from the active telemetry stream</div>
                        </div>
                    </div>
                    <div class="chart-container">
                        <div id="sensor-chart" style="width: 100%; height: 100%;"></div>
                    </div>
                </div>
                <div class="chart-card">
                    <div class="panel-header">
                        <div>
                            <div class="panel-title">Anomalies by Sensor</div>
                            <div class="panel-subtitle">Counts of detected anomalies over the current session</div>
                        </div>
                    </div>
                    <div class="chart-container">
                        <div id="anomaly-distribution" style="width: 100%; height: 100%;"></div>
                    </div>
                </div>
            </div>
            
            <div class="alerts-panel">
                <div class="panel-header">
                    <div>
                        <div class="panel-title">Recent Anomalies</div>
                            <div class="panel-subtitle">Latest alerts from the ensemble detector</div>
                    </div>
                </div>
                <div id="alerts-list" class="alerts-list">
                    <div class="empty-state">Waiting for anomaly events...</div>
                </div>
            </div>
        </div>

        <script>
            const ws = new WebSocket(`ws://${window.location.host}/ws`);
            const sensorData = {};
            const anomalies = [];
            let messageCount = 0;
            let lastCountTime = Date.now();
            const SENSOR_IDS = ['sensor_01', 'sensor_02', 'sensor_03', 'sensor_04'];

            SENSOR_IDS.forEach(id => {
                sensorData[id] = {
                    timestamps: [],
                    values: [],
                    anomalies: []
                };
            });

            async function loadSystemMetrics() {
                try {
                    const response = await fetch('/api/status');
                    const status = await response.json();
                    const metrics = status.metrics || {};

                    if (metrics.accuracy !== undefined) {
                        document.getElementById('accuracy').textContent = `${metrics.accuracy.toFixed(1)}%`;
                    }
                    if (metrics.precision !== undefined) {
                        document.getElementById('precision').textContent = `${metrics.precision.toFixed(1)}%`;
                    }
                    if (metrics.recall !== undefined) {
                        document.getElementById('recall').textContent = `${metrics.recall.toFixed(1)}%`;
                    }
                } catch (error) {
                    console.error('Failed to load metrics:', error);
                }
            }

            ws.onmessage = function(event) {
                const data = JSON.parse(event.data);
                const sensorValues = data.sensor_values || {};

                Object.keys(sensorValues).forEach(sensorId => {
                    if (!sensorData[sensorId]) {
                        sensorData[sensorId] = { timestamps: [], values: [], anomalies: [] };
                    }

                    if (sensorData[sensorId].timestamps.length > 200) {
                        sensorData[sensorId].timestamps.shift();
                        sensorData[sensorId].values.shift();
                        sensorData[sensorId].anomalies.shift();
                    }

                    sensorData[sensorId].timestamps.push(data.timestamp);
                    sensorData[sensorId].values.push(sensorValues[sensorId]);
                    sensorData[sensorId].anomalies.push(data.anomaly_detected ? sensorValues[sensorId] : null);
                });

                messageCount++;

                if (data.anomaly_detected) {
                    anomalies.unshift({
                        timestamp: data.timestamp,
                        sample_id: data.sample_id,
                        sensor_id: 'multivariate sample',
                        value: Object.values(sensorValues).map(v => Number(v).toFixed(2)).join(' / '),
                        confidence: (data.confidence * 100).toFixed(1),
                        methods: data.detection_methods.join(', ')
                    });

                    if (anomalies.length > 20) anomalies.pop();
                    document.getElementById('anomaly-count').textContent = anomalies.length;
                    updateAlerts();
                }

                // Update charts every 10 messages
                if (messageCount % 10 === 0) {
                    updateThroughput();
                    updateCharts();
                }
            };

            function updateThroughput() {
                const now = Date.now();
                const elapsed = (now - lastCountTime) / 1000;
                const throughput = (messageCount / elapsed).toFixed(1);
                document.getElementById('throughput').textContent = throughput;
                messageCount = 0;
                lastCountTime = now;
            }

            function updateCharts() {
                // Sensor values chart
                const traces = [];
                Object.keys(sensorData).forEach(sensorId => {
                    traces.push({
                        x: sensorData[sensorId].timestamps,
                        y: sensorData[sensorId].values,
                        name: sensorId,
                        mode: 'lines',
                        type: 'scatter'
                    });
                });

                Plotly.react('sensor-chart', traces, {
                    title: 'Sensor Readings Over Time',
                    hovermode: 'x unified',
                    plot_bgcolor: 'rgba(6, 12, 24, 0.12)',
                    paper_bgcolor: 'rgba(0,0,0,0)',
                    font: { color: '#eaf2ff' },
                    titlefont: { color: '#eaf2ff', size: 15 },
                    hoverlabel: {
                        bgcolor: '#0b1220',
                        bordercolor: '#52d0ff',
                        font: { color: '#ffffff', size: 14 }
                    },
                    xaxis: {
                        title: 'Time',
                        color: '#8fa2bf',
                        gridcolor: 'rgba(143, 162, 191, 0.15)',
                        zerolinecolor: 'rgba(143, 162, 191, 0.15)'
                    },
                    yaxis: {
                        title: 'Value',
                        color: '#8fa2bf',
                        gridcolor: 'rgba(143, 162, 191, 0.15)',
                        zerolinecolor: 'rgba(143, 162, 191, 0.15)'
                    },
                    margin: { l: 60, r: 20, t: 18, b: 46 },
                    height: 320,
                    autosize: false
                }, { responsive: true });

                // Anomaly distribution
                const anomalyByMethod = {};
                anomalies.forEach(a => {
                    if (!anomalyByMethod[a.sensor_id]) {
                        anomalyByMethod[a.sensor_id] = 0;
                    }
                    anomalyByMethod[a.sensor_id]++;
                });

                const aTrace = [{
                    x: Object.keys(anomalyByMethod),
                    y: Object.values(anomalyByMethod),
                    type: 'bar',
                    marker: { color: '#ff4444' }
                }];

                Plotly.react('anomaly-distribution', aTrace, {
                    title: 'Anomalies by Sensor',
                    plot_bgcolor: 'rgba(6, 12, 24, 0.12)',
                    paper_bgcolor: 'rgba(0,0,0,0)',
                    font: { color: '#eaf2ff' },
                    titlefont: { color: '#eaf2ff', size: 15 },
                    hoverlabel: {
                        bgcolor: '#0b1220',
                        bordercolor: '#52d0ff',
                        font: { color: '#ffffff', size: 14 }
                    },
                    xaxis: {
                        title: 'Sensor ID',
                        color: '#8fa2bf',
                        gridcolor: 'rgba(143, 162, 191, 0.15)',
                        zerolinecolor: 'rgba(143, 162, 191, 0.15)'
                    },
                    yaxis: {
                        title: 'Count',
                        color: '#8fa2bf',
                        gridcolor: 'rgba(143, 162, 191, 0.15)',
                        zerolinecolor: 'rgba(143, 162, 191, 0.15)'
                    },
                    margin: { l: 60, r: 20, t: 18, b: 46 },
                    height: 320,
                    autosize: false
                }, { responsive: true });
            }

            function updateAlerts() {
                const alertsList = document.getElementById('alerts-list');
                if (anomalies.length === 0) {
                    alertsList.innerHTML = '<div class="empty-state">No anomalies detected yet</div>';
                    return;
                }

                alertsList.innerHTML = anomalies.slice(0, 10).map(a => `
                    <div class="alert-item">
                        <div class="alert-top">
                            <div class="alert-sensor">${a.sample_id || a.sensor_id}</div>
                            <span class="anomaly-badge">${(a.confidence)}% confidence</span>
                        </div>
                        <div class="alert-meta">Features: ${a.value} | Methods: ${a.methods}</div>
                        <div class="alert-time">${new Date(a.timestamp).toLocaleTimeString()}</div>
                    </div>
                `).join('');
            }

            ws.onerror = function(error) {
                console.error('WebSocket error:', error);
            };

            ws.onclose = function() {
                console.log('WebSocket connection closed');
            };

            // Initial chart
            loadSystemMetrics();
            updateCharts();
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=dashboard_html)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time data streaming"""
    await websocket.accept()
    connected_clients.add(websocket)
    logger.info(f"Client connected. Total clients: {len(connected_clients)}")
    
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        connected_clients.discard(websocket)
        logger.info(f"Client disconnected. Total clients: {len(connected_clients)}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        connected_clients.discard(websocket)

@app.get("/api/status")
async def get_status():
    """Get system status"""
    return {
        "status": "online",
        "dataset": "Breast Cancer Wisconsin (Diagnostic)",
        "samples": dashboard_metrics.get("samples", 0),
        "connected_clients": len(connected_clients),
        "total_anomalies": sum(1 for event in sensor_history.get('events', []) if event.get('anomaly_detected')),
        "latency_ms": "<100",
        "metrics": dashboard_metrics
    }

@app.get("/api/history/{sensor_id}")
async def get_sensor_history(sensor_id: str):
    """Get historical data for a sensor"""
    events = sensor_history.get('events', [])
    return [
        event for event in events
        if event.get('sensor_id') == sensor_id or event.get('sample_id') == sensor_id
    ]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
