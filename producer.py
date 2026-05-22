"""
producer.py — Simulates downhole sensor telemetry and publishes to Kafka.
Simulates RPM, torque, temperature, vibration with drift/anomaly injection.
"""

import json
import math
import random
import time
from dataclasses import asdict, dataclass

from kafka import KafkaProducer

KAFKA_BROKER = "localhost:9092"
TOPIC = "drill.telemetry"
PUBLISH_INTERVAL = 0.05  # 20 Hz → 1000+ msg/sec across multiple sensors


@dataclass
class TelemetryFrame:
    timestamp: float
    sensor_id: str
    depth_m: float
    rpm: float
    torque_nm: float
    temperature_c: float
    vibration_g: float
    wob: float          # Weight-on-bit (kN)
    flow_rate: float    # L/min
    is_anomaly: bool = False


class DrillSensorSimulator:
    """Physics-inspired sensor drift model for rotary steerable systems."""

    def __init__(self, sensor_id: str, drift_start: int = 200):
        self.sensor_id = sensor_id
        self.step = 0
        self.drift_start = drift_start
        self.depth = 1500.0  # starting depth in metres

    def _drift_factor(self) -> float:
        if self.step < self.drift_start:
            return 0.0
        t = self.step - self.drift_start
        return min(1.0, t / 300.0)  # ramps over ~300 steps

    def next(self) -> TelemetryFrame:
        drift = self._drift_factor()
        noise = lambda s: random.gauss(0, s)

        rpm         = 120.0 - drift * 15.0 + noise(2.0)
        torque      = 8500.0 + drift * 800.0 + noise(120.0)
        temperature = 65.0 + drift * 18.0 + noise(0.8)
        vibration   = 0.4 + drift * math.exp(drift * 2.5) + abs(noise(0.05))
        wob         = 120.0 + drift * 10.0 + noise(5.0)
        flow_rate   = 1200.0 - drift * 80.0 + noise(15.0)

        # spike injection (rare random anomalies)
        is_anomaly = False
        if random.random() < 0.02:
            vibration *= random.uniform(3.5, 6.0)
            is_anomaly = True

        self.depth += random.uniform(0.01, 0.03)
        self.step += 1

        return TelemetryFrame(
            timestamp=time.time(),
            sensor_id=self.sensor_id,
            depth_m=round(self.depth, 2),
            rpm=round(rpm, 2),
            torque_nm=round(torque, 2),
            temperature_c=round(temperature, 2),
            vibration_g=round(vibration, 4),
            wob=round(wob, 2),
            flow_rate=round(flow_rate, 2),
            is_anomaly=is_anomaly,
        )


def main():
    producer = KafkaProducer(
        bootstrap_servers=KAFKA_BROKER,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        acks="all",
        retries=5,
        linger_ms=5,
        batch_size=16384,
    )

    sensors = [DrillSensorSimulator(f"SENSOR_{i:02d}") for i in range(4)]
    print(f"[Producer] Publishing to topic '{TOPIC}' on {KAFKA_BROKER}")

    try:
        while True:
            for sim in sensors:
                frame = sim.next()
                producer.send(TOPIC, value=asdict(frame))
            time.sleep(PUBLISH_INTERVAL)
    except KeyboardInterrupt:
        print("\n[Producer] Shutting down.")
    finally:
        producer.flush()
        producer.close()


if __name__ == "__main__":
    main()
