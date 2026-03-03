import numpy as np
from typing import Dict, Any, Tuple, Optional
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class AnomalyDetector:
    """Multi-method anomaly detection (IF, LSTM, Prophet)."""
    
    def __init__(self, model_manager, forecast_manager):
        self.model_manager = model_manager
        self.forecast_manager = forecast_manager
        self.feature_names = [
            "battery_voltage", "bus_voltage", "wheel_rpm", "current", "wheel_temperature"
        ]
        
    def prepare_features(self, data: Dict[str, Any]) -> np.ndarray:
        """Extract features from telemetry data."""
        features = np.array([data[f] for f in self.feature_names], dtype=np.float32)
        return features.reshape(1, -1)
    
    def prepare_sequence(self, data_list: list, sequence_length: int = 10) -> Optional[np.ndarray]:
        """Prepare sequence for LSTM."""
        if len(data_list) < sequence_length:
            return None
        
        sequences = []
        for i in range(len(data_list) - sequence_length + 1):
            seq = []
            for j in range(sequence_length):
                features = self.prepare_features(data_list[i + j])
                seq.append(features[0])
            sequences.append(seq)
        
        return np.array(sequences, dtype=np.float32)
    
    def detect_isolation_forest(self, features: np.ndarray) -> Tuple[bool, float]:
        """Detect anomaly using Isolation Forest."""
        try:
            if_model = self.model_manager.get_isolation_forest()
            if if_model is None:
                return False, 0.0
            
            # IF returns -1 for anomalies, 1 for normal
            predictions = if_model.predict(features)
            anomaly_score = if_model.score_samples(features)[0]
            
            # Normalize score to [0, 1] where 1 is more anomalous
            is_anomaly = predictions[0] == -1
            confidence = 1 / (1 + np.exp(-anomaly_score))  # Sigmoid
            
            return is_anomaly, confidence
        except Exception as e:
            logger.error(f"Error in Isolation Forest detection: {e}")
            return False, 0.0
    
    def detect_lstm_reconstruction(self, current_data: Dict[str, Any], 
                                   recent_data: list) -> Tuple[bool, float]:
        """Detect anomaly using LSTM reconstruction error."""
        try:
            lstm_model = self.model_manager.get_lstm_autoencoder()
            if lstm_model is None:
                return False, 0.0
            
            # Add current data to recent data for sequence
            all_data = recent_data + [current_data]
            sequence = self.prepare_sequence(all_data, sequence_length=10)
            
            if sequence is None:
                return False, 0.0
            
            # Use only last sequence for prediction
            last_sequence = sequence[-1:].reshape(1, 10, 5)
            
            # Get reconstruction error
            reconstruction = lstm_model.predict(last_sequence, verbose=0)
            mse = np.mean((sequence[-1:] - reconstruction) ** 2)
            
            # Threshold-based detection
            threshold = 0.05
            is_anomaly = mse > threshold
            confidence = min(1.0, mse * 10)  # Scale MSE to confidence
            
            return is_anomaly, confidence
        except Exception as e:
            logger.error(f"Error in LSTM detection: {e}")
            return False, 0.0
    
    def detect_prophet_deviation(self, current_data: Dict[str, Any], 
                                 recent_data: list) -> Tuple[bool, float]:
        """Detect anomaly using Prophet forecast deviation."""
        try:
            # Use battery voltage for forecasting
            current_value = current_data["battery_voltage"]
            forecast = self.forecast_manager.forecast_next(recent_data)
            
            if forecast is None:
                return False, 0.0
            
            # Calculate deviation from forecast
            deviation = abs(current_value - forecast) / (abs(forecast) + 0.1)
            
            # Threshold-based detection
            threshold = 0.5
            is_anomaly = deviation > threshold
            confidence = min(1.0, deviation)
            
            return is_anomaly, confidence
        except Exception as e:
            logger.error(f"Error in Prophet detection: {e}")
            return False, 0.0
    
    def predict(self, current_data: Dict[str, Any], 
                recent_data: list) -> Dict[str, Any]:
        """Run all anomaly detection methods."""
        # Prepare features
        features = self.prepare_features(current_data)
        
        # Run detectors
        if_anomaly, if_confidence = self.detect_isolation_forest(features)
        lstm_anomaly, lstm_confidence = self.detect_lstm_reconstruction(current_data, recent_data)
        prophet_anomaly, prophet_confidence = self.detect_prophet_deviation(current_data, recent_data)
        
        # Ensemble decision (majority voting)
        anomaly_votes = sum([if_anomaly, lstm_anomaly, prophet_anomaly])
        is_anomaly = anomaly_votes >= 2
        
        # Determine detection method
        if is_anomaly:
            detection_methods = []
            if if_anomaly:
                detection_methods.append("IF")
            if lstm_anomaly:
                detection_methods.append("LSTM")
            if prophet_anomaly:
                detection_methods.append("Prophet")
            detection_method = ", ".join(detection_methods)
        else:
            detection_method = "Normal"
        
        # Average confidence
        avg_confidence = (if_confidence + lstm_confidence + prophet_confidence) / 3
        
        return {
            "timestamp": current_data.get("timestamp", datetime.utcnow()),
            "is_anomaly": is_anomaly,
            "isolation_forest_score": float(if_confidence),
            "lstm_reconstruction_error": float(lstm_confidence),
            "prophet_deviation": float(prophet_confidence),
            "detection_method": detection_method,
            "confidence": float(avg_confidence),
        }


# Will be instantiated after imports
anomaly_detector = None
