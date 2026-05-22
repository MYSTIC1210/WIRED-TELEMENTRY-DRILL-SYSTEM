import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.ensemble import RandomForestClassifier

class LSTMAutoencoder:
    """Simple LSTM-like anomaly detection using statistical methods"""
    def __init__(self, threshold=2.5):
        self.threshold = threshold
        self.mean = None
        self.std = None
        self.is_fitted = False
        
    def fit(self, data):
        """Fit the model on training data"""
        data_array = np.asarray(data)
        if data_array.ndim == 1:
            data_array = data_array.reshape(-1, 1)
        self.mean = np.mean(data_array, axis=0)
        self.std = np.std(data_array, axis=0)
        self.is_fitted = True
        
    def predict(self, value):
        """Predict if value is anomalous"""
        if not self.is_fitted:
            return False

        value_array = np.asarray(value)
        if value_array.ndim == 0:
            value_array = value_array.reshape(1)
        z_score = np.abs((value_array - self.mean) / (self.std + 1e-8))
        return bool(np.any(z_score > self.threshold))

class ZScoreDetector:
    """Z-Score based anomaly detection"""
    def __init__(self, threshold=3.0):
        self.threshold = threshold
        self.window_size = 100
        self.values = []
        self.mean = None
        self.std = None
        self.is_fitted = False
        
    def fit(self, data):
        data_array = np.asarray(data)
        if data_array.ndim == 1:
            data_array = data_array.reshape(-1, 1)
        self.mean = np.mean(data_array, axis=0)
        self.std = np.std(data_array, axis=0)
        self.is_fitted = True
        
    def detect(self, value):
        """Detect anomaly using Z-score"""
        value_array = np.asarray(value)
        if value_array.ndim == 0:
            value_array = value_array.reshape(1)

        self.values.append(value_array)
        if len(self.values) > self.window_size:
            self.values.pop(0)

        if self.is_fitted and self.mean is not None and self.std is not None:
            z_score = np.abs((value_array - self.mean) / (self.std + 1e-8))
            return bool(np.any(z_score > self.threshold))

        if len(self.values) < 10:
            return False

        window = np.asarray(self.values)
        mean = np.mean(window, axis=0)
        std = np.std(window, axis=0)
        z_score = np.abs((value_array - mean) / (std + 1e-8))
        return bool(np.any(z_score > self.threshold))

class EnsembleAnomalyDetector:
    """Ensemble anomaly detection combining multiple methods"""
    
    def __init__(self):
        self.isolation_forest = IsolationForest(
            contamination=0.1,
            random_state=42,
            n_estimators=100
        )
        self.classifier = RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight="balanced"
        )
        self.lstm_ae = LSTMAutoencoder(threshold=2.5)
        self.zscore_detector = ZScoreDetector(threshold=3.0)
        self.is_trained = False
        self.training_data = []
        self.feature_names = []
        self.metrics = {}
        
    def train(self, data_points, labels=None, feature_names=None):
        """Train the ensemble on historical data"""
        if len(data_points) < 20:
            return
            
        data_array = np.asarray(data_points)
        if data_array.ndim == 1:
            data_array = data_array.reshape(-1, 1)

        self.feature_names = feature_names or [f"feature_{i}" for i in range(data_array.shape[1])]

        if labels is not None:
            labels_array = np.asarray(labels).astype(int)
            normal_mask = labels_array == 0
            normal_data = data_array[normal_mask] if np.any(normal_mask) else data_array
            self.classifier.fit(data_array, labels_array)
            self.isolation_forest.fit(normal_data)
            self.lstm_ae.fit(normal_data)
            self.zscore_detector.fit(normal_data)
        else:
            self.isolation_forest.fit(data_array)
            self.lstm_ae.fit(data_array)
            self.zscore_detector.fit(data_array)
        
        self.is_trained = True

    def evaluate(self, data_points, labels):
        data_array = np.asarray(data_points)
        if data_array.ndim == 1:
            data_array = data_array.reshape(-1, 1)
        labels_array = np.asarray(labels).astype(int)

        # Preserve the rolling Z-score state so batch evaluation does not
        # change live detection behavior after startup.
        zscore_values = list(self.zscore_detector.values)

        try:
            predictions = []
            for row in data_array:
                is_anomaly, _, _ = self.detect({name: value for name, value in zip(self.feature_names, row)})
                predictions.append(1 if is_anomaly else 0)
        finally:
            self.zscore_detector.values = zscore_values

        predictions = np.asarray(predictions)
        accuracy = float(np.mean(predictions == labels_array))
        true_positive = int(np.sum((predictions == 1) & (labels_array == 1)))
        false_positive = int(np.sum((predictions == 1) & (labels_array == 0)))
        false_negative = int(np.sum((predictions == 0) & (labels_array == 1)))

        precision = true_positive / (true_positive + false_positive + 1e-8)
        recall = true_positive / (true_positive + false_negative + 1e-8)
        f1_score = 2 * precision * recall / (precision + recall + 1e-8)

        self.metrics = {
            "accuracy": round(accuracy * 100, 1),
            "precision": round(precision * 100, 1),
            "recall": round(recall * 100, 1),
            "f1_score": round(f1_score * 100, 1),
            "samples": int(len(labels_array))
        }
        return self.metrics

    def extract_vector(self, sensor_data):
        if not isinstance(sensor_data, dict):
            return None

        if 'sensor_values' in sensor_data and isinstance(sensor_data['sensor_values'], dict):
            ordered_values = [sensor_data['sensor_values'].get(name) for name in self.feature_names]
            if all(value is not None for value in ordered_values):
                return np.asarray(ordered_values, dtype=float)

        ordered_values = [sensor_data.get(name) for name in self.feature_names if name in sensor_data]
        if ordered_values and len(ordered_values) == len(self.feature_names):
            return np.asarray(ordered_values, dtype=float)

        numeric_values = [float(value) for key, value in sensor_data.items() if isinstance(value, (int, float))]
        if numeric_values:
            return np.asarray(numeric_values, dtype=float)
        return None
        
    def detect(self, sensor_data):
        """
        Detect anomalies in sensor data
        Returns: (is_anomaly, confidence, methods)
        """
        vector = self.extract_vector(sensor_data)
        if vector is None:
            return False, 0.0, []
            
        results = []
        anomaly_votes = 0

        if self.is_trained:
            try:
                classifier_pred = int(self.classifier.predict(vector.reshape(1, -1))[0])
                if classifier_pred == 1:
                    anomaly_votes += 1
                    results.append("random_forest")
            except Exception:
                pass
        
        # Isolation Forest detection
        if self.is_trained:
            if_pred = self.isolation_forest.predict(vector.reshape(1, -1))[0]
            if if_pred == -1:  # -1 indicates anomaly
                anomaly_votes += 1
                results.append("isolation_forest")
        
        # LSTM Autoencoder detection
        if self.lstm_ae.is_fitted and self.lstm_ae.predict(vector):
            anomaly_votes += 1
            results.append("lstm_autoencoder")
        
        # Z-Score detection
        if self.zscore_detector.detect(vector):
            anomaly_votes += 1
            results.append("zscore")
        
        # Majority voting across available methods
        method_count = 4 if self.is_trained else 2
        is_anomaly = anomaly_votes >= 2
        confidence = anomaly_votes / float(method_count)
        
        return is_anomaly, confidence, results
