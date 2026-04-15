"""Anomaly detection using Isolation Forest, Local Outlier Factor, LSTM, and Autoencoder."""

import numpy as np
from typing import Dict, Any, Tuple, List
import logging

logger = logging.getLogger(__name__)


class AnomalyDetector:
    """Multi-model ensemble anomaly detection using Isolation Forest, LOF, LSTM, and Autoencoder."""

    def __init__(self, model_manager, config):
        self.model_manager = model_manager
        self.config = config
        self.feature_names = [
            "battery_voltage",
            "bus_voltage",
            "wheel_rpm",
            "current",
            "wheel_temperature"
        ]
        self.sequence_length = 10  # For LSTM: look at 10 previous timesteps

    def prepare_features(self, data: Dict[str, Any]) -> np.ndarray:
        """Extract features from telemetry data."""
        features = np.array(
            [data.get(f, 0.0) for f in self.feature_names],
            dtype=np.float32
        )
        return features.reshape(1, -1)

    def detect(self, data: Dict[str, Any]) -> Tuple[bool, float, Dict]:
        """
        Detect anomaly using ensemble of multiple models:
        - Isolation Forest
        - Local Outlier Factor (LOF)
        - LSTM
        - Autoencoder
        
        Returns:
            Tuple of (is_anomaly, confidence, details_dict)
        """
        try:
            results = {
                "isolation_forest": {"is_anomaly": False, "confidence": 0.0},
                "local_outlier_factor": {"is_anomaly": False, "confidence": 0.0},
                "lstm": {"is_anomaly": False, "confidence": 0.0},
                "autoencoder": {"is_anomaly": False, "confidence": 0.0},
                "ensemble_anomaly": False,
                "ensemble_confidence": 0.0,
                "detection_method": "ensemble (IF + LOF + LSTM + Autoencoder)"
            }

            # Isolation Forest detection
            if_score = self._detect_isolation_forest(data)
            if if_score is not None:
                results["isolation_forest"] = if_score

            # Local Outlier Factor detection
            lof_score = self._detect_local_outlier_factor(data)
            if lof_score is not None:
                results["local_outlier_factor"] = lof_score

            # LSTM detection
            lstm_score = self._detect_lstm(data)
            if lstm_score is not None:
                results["lstm"] = lstm_score

            # Autoencoder detection
            ae_score = self._detect_autoencoder(data)
            if ae_score is not None:
                results["autoencoder"] = ae_score

            # Ensemble decision: voting mechanism
            anomaly_votes = sum([
                results["isolation_forest"]["is_anomaly"],
                results["local_outlier_factor"]["is_anomaly"],
                results["lstm"]["is_anomaly"],
                results["autoencoder"]["is_anomaly"]
            ])
            
            # Flag as anomaly if 2 or more models agree
            results["ensemble_anomaly"] = anomaly_votes >= 2
            
            # Calculate combined confidence (average of all models)
            confidences = [
                results["isolation_forest"]["confidence"],
                results["local_outlier_factor"]["confidence"],
                results["lstm"]["confidence"],
                results["autoencoder"]["confidence"]
            ]
            results["ensemble_confidence"] = np.mean(confidences)

            return results["ensemble_anomaly"], results["ensemble_confidence"], results

        except Exception as e:
            logger.error(f"Error in anomaly detection: {e}")
            return False, 0.0, {"error": str(e)}

    def _detect_isolation_forest(self, data: Dict[str, Any]) -> Dict:
        """Detect anomaly using Isolation Forest."""
        try:
            if_model = self.model_manager.get_isolation_forest()
            if if_model is None:
                return {"is_anomaly": False, "confidence": 0.0}

            features = self.prepare_features(data)

            # IF returns -1 for anomalies, 1 for normal
            predictions = if_model.predict(features)
            anomaly_scores = if_model.score_samples(features)

            is_anomaly = predictions[0] == -1
            # Normalize score to [0, 1] where 1 is more anomalous
            confidence = 1 / (1 + np.exp(-anomaly_scores[0]))

            return {
                "is_anomaly": bool(is_anomaly),
                "confidence": float(confidence)
            }
        except Exception as e:
            logger.warning(f"Isolation Forest detection failed: {e}")
            return {"is_anomaly": False, "confidence": 0.0}

    def _detect_lstm(self, data: Dict[str, Any]) -> Dict:
        """Detect anomaly using LSTM model."""
        try:
            lstm_model = self.model_manager.get_lstm_model()
            if lstm_model is None:
                return {"is_anomaly": False, "confidence": 0.0}

            features = self.prepare_features(data)
            
            # LSTM expects 3D input: (batch_size, sequence_length, features)
            # Create a simple sequence by repeating the current sample
            sequence = np.repeat(features, self.sequence_length, axis=0).reshape(1, self.sequence_length, -1)
            
            # Get prediction (assumes model outputs anomaly score between 0 and 1)
            prediction = lstm_model.predict(sequence, verbose=0)
            confidence = float(prediction[0][0]) if prediction.shape[-1] > 0 else 0.0
            
            # Threshold for anomaly
            threshold = 0.5
            is_anomaly = confidence > threshold

            return {
                "is_anomaly": bool(is_anomaly),
                "confidence": min(confidence, 1.0)  # Ensure it's between 0 and 1
            }
        except Exception as e:
            logger.warning(f"LSTM detection failed: {e}")
            return {"is_anomaly": False, "confidence": 0.0}

    def _detect_local_outlier_factor(self, data: Dict[str, Any]) -> Dict:
        """Detect anomaly using Local Outlier Factor (LOF)."""
        try:
            lof_model = self.model_manager.get_lof_model()
            if lof_model is None:
                return {"is_anomaly": False, "confidence": 0.0}

            features = self.prepare_features(data)

            # LOF returns -1 for outliers, 1 for inliers
            predictions = lof_model.predict(features)
            is_anomaly = predictions[0] == -1
            
            # Get LOF scores (negative_outlier_factor)
            # LOF scores close to 1 are normal, > 1 are outliers
            lof_scores = lof_model.negative_outlier_factor_
            if len(lof_scores) > 0:
                # Normalize to [0, 1] using sigmoid
                mean_score = np.mean(lof_scores)
                confidence = 1 / (1 + np.exp(-(1 - mean_score) * 5))  # Scaled sigmoid
            else:
                confidence = 0.0

            return {
                "is_anomaly": bool(is_anomaly),
                "confidence": float(np.clip(confidence, 0, 1))
            }
        except Exception as e:
            logger.warning(f"Local Outlier Factor detection failed: {e}")
            return {"is_anomaly": False, "confidence": 0.0}

    def _detect_autoencoder(self, data: Dict[str, Any]) -> Dict:
        """Detect anomaly using Autoencoder reconstruction error."""
        try:
            autoencoder_model = self.model_manager.get_autoencoder_model()
            if autoencoder_model is None:
                return {"is_anomaly": False, "confidence": 0.0}

            features = self.prepare_features(data)
            
            # Autoencoder: compute reconstruction error
            reconstruction = autoencoder_model.predict(features, verbose=0)
            
            # Calculate Mean Squared Error (MSE) as anomaly score
            mse = np.mean((features - reconstruction) ** 2)
            
            # Use a threshold or normalized score
            # Normalize using sigmoid for confidence
            threshold = 0.05  # Typical threshold for MSE
            confidence = min(mse / threshold, 1.0)  # Confidence increases with error
            is_anomaly = mse > threshold

            return {
                "is_anomaly": bool(is_anomaly),
                "confidence": float(confidence)
            }
        except Exception as e:
            logger.warning(f"Autoencoder detection failed: {e}")
            return {"is_anomaly": False, "confidence": 0.0}
