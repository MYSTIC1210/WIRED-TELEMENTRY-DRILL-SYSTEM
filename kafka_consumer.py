import json
import logging
from kafka import KafkaConsumer
from anomaly_detector import EnsembleAnomalyDetector
import asyncio
from typing import Callable

logger = logging.getLogger(__name__)

class TelemetryKafkaConsumer:
    """Kafka consumer for drill pipe telemetry data"""
    
    def __init__(self, bootstrap_servers='localhost:9092', topic='drill-telemetry'):
        self.bootstrap_servers = bootstrap_servers
        self.topic = topic
        self.consumer = None
        self.anomaly_detector = EnsembleAnomalyDetector()
        self.is_running = False
        
    def connect(self):
        """Connect to Kafka broker"""
        try:
            self.consumer = KafkaConsumer(
                self.topic,
                bootstrap_servers=self.bootstrap_servers,
                group_id='telemetry-processor',
                value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                auto_offset_reset='latest',
                session_timeout_ms=6000,
            )
            logger.info(f"Connected to Kafka broker at {self.bootstrap_servers}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Kafka: {e}")
            return False
    
    def process_message(self, message_data):
        """Process incoming telemetry message"""
        try:
            is_anomaly, confidence, methods = self.anomaly_detector.detect(message_data)
            
            result = {
                'timestamp': message_data.get('timestamp'),
                'sensor_id': message_data.get('sensor_id'),
                'data': message_data,
                'anomaly_detected': is_anomaly,
                'confidence': float(confidence),
                'detection_methods': methods
            }
            
            return result
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            return None
    
    def train_detector(self, training_messages):
        """Train anomaly detector on historical data"""
        values = []
        for msg in training_messages:
            try:
                if isinstance(msg, dict) and 'value' in msg:
                    values.append(float(msg['value']))
            except (ValueError, TypeError):
                continue
        
        if values:
            self.anomaly_detector.train(values)
            logger.info(f"Trained detector on {len(values)} samples")
    
    async def consume_messages(self, callback: Callable = None):
        """Consume messages from Kafka topic"""
        if not self.consumer:
            if not self.connect():
                return
        
        self.is_running = True
        logger.info(f"Starting to consume from topic: {self.topic}")
        
        try:
            for message in self.consumer:
                if not self.is_running:
                    break
                
                processed = self.process_message(message.value)
                
                if processed and callback:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(processed)
                    else:
                        callback(processed)
                        
        except Exception as e:
            logger.error(f"Error consuming messages: {e}")
        finally:
            self.stop()
    
    def stop(self):
        """Stop consuming messages"""
        self.is_running = False
        if self.consumer:
            self.consumer.close()
            logger.info("Kafka consumer stopped")
