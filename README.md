# Wired Drill Pipe Telemetry System

A real-time anomaly detection platform for downhole telemetry using FastAPI, Kafka-ready streaming, and ensemble machine learning.

## Features

- **High-throughput pipeline** - Designed for 1000+ messages per second
- **Ensemble anomaly detection** - Isolation Forest, LSTM Autoencoder, and Z-Score methods
- **Low-latency delivery** - Sub-100 ms end-to-end target
- **Real-world evaluation metrics** - Accuracy, precision, recall, and F1 score on labeled data
- **Live operations dashboard** - Plotly-based monitoring for operators
- **WebSocket streaming** - Continuous push updates to connected clients

## Tech Stack

| Layer | Technology |
| ----- | ---------- |
| Streaming | Apache Kafka |
| Backend API | FastAPI |
| ML Models | Isolation Forest, LSTM Autoencoder, Z-Score |
| Frontend | Plotly Dashboard, WebSocket |
| Language | Python |

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Start Kafka (Optional)

```bash
docker-compose up -d
```

### 3. Run the Backend

```bash
python main.py
```

### 4. Access the Dashboard

Open your browser and navigate to:

```text
http://localhost:8000
```

## Architecture

```text
Downhole Sensors
      |
      v
 Kafka Topic (1000+ msg/sec)
      |
      v
Ensemble ML Anomaly Detector
 |- Isolation Forest
 |- LSTM Autoencoder
 `- Z-Score Detector
      |
      v
 FastAPI Backend --> WebSocket --> Plotly Dashboard
```

## Performance

- Throughput target: 1000+ messages/second
- Latency target: <100 ms
- Evaluation metrics: accuracy, precision, recall, F1 score

## Components

### anomaly_detector.py

Ensemble anomaly detection with three methods:

- **Isolation Forest**: Tree-based isolation of anomalies
- **LSTM Autoencoder**: Statistical reconstruction-based detection
- **Z-Score**: Statistical deviation detection

### kafka_consumer.py

Kafka consumer for telemetry data processing

### main.py

FastAPI backend with:

- WebSocket real-time streaming
- Interactive Plotly dashboard
- RESTful API endpoints
- Real-world dataset streaming and model metrics

## API Endpoints

- `GET /` - Interactive dashboard
- `GET /api/status` - System status
- `GET /api/history/{sensor_id}` - Historical sensor data
- `WS /ws` - WebSocket for real-time updates

## Configuration

Edit `.env` to configure:

- `KAFKA_BOOTSTRAP_SERVERS` - Kafka broker address
- `KAFKA_TOPIC` - Topic name
- `API_PORT` - API port (default: 8000)

## Notes

- The application trains anomaly detectors on startup using a labeled real-world dataset
- Kafka is optional; the dashboard runs with in-process dataset streaming by default
- The system keeps the most recent 1000 events in memory
- Connected WebSocket clients receive real-time updates
