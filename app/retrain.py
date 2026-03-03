import threading
import logging
import time
from datetime import datetime
import os
import numpy as np
from typing import Dict, Any
import pickle

logger = logging.getLogger(__name__)


class RetrainingScheduler:
    """Lightweight sliding-window retraining scheduler."""
    
    def __init__(self, model_manager, db_manager, forecast_manager):
        self.model_manager = model_manager
        self.db_manager = db_manager
        self.forecast_manager = forecast_manager
        
        self.retrain_interval = int(os.getenv("RETRAIN_INTERVAL", 3600))  # 60 minutes default
        self.sliding_window_size = int(os.getenv("SLIDING_WINDOW_SIZE", 3000))
        
        self.last_retrain_time = datetime.utcnow()
        self.retrain_lock = threading.Lock()
        self.is_retraining = False
        self.retrain_thread = None
        
    def should_retrain(self) -> bool:
        """Check if retraining is due."""
        elapsed = (datetime.utcnow() - self.last_retrain_time).total_seconds()
        return elapsed >= self.retrain_interval and not self.is_retraining
    
    def retrain_models(self) -> bool:
        """Retrain IF and LSTM models with sliding window."""
        with self.retrain_lock:
            try:
                self.is_retraining = True
                logger.info("Starting model retraining...")
                
                # Fetch recent data
                telemetry_data = self.db_manager.get_telemetry_for_retraining(
                    self.sliding_window_size
                )
                
                if len(telemetry_data) < 100:
                    logger.warning(f"Insufficient data for retraining: {len(telemetry_data)}")
                    return False
                
                # Prepare features
                feature_names = ["battery_voltage", "bus_voltage", "wheel_rpm", "current", "wheel_temperature"]
                X = np.array([
                    [record.get(f, 0.0) for f in feature_names]
                    for record in telemetry_data
                ], dtype=np.float32)
                
                # Retrain Isolation Forest
                if_success = self.model_manager.retrain_isolation_forest(X)
                
                # Retrain LSTM (lightweight with fewer epochs)
                # Reshape for LSTM: (samples, timesteps, features)
                sequence_length = 10
                if len(X) >= sequence_length:
                    sequences = []
                    for i in range(len(X) - sequence_length + 1):
                        sequences.append(X[i:i + sequence_length])
                    
                    if len(sequences) > 0:
                        X_seq = np.array(sequences, dtype=np.float32)
                        lstm_success = self.model_manager.retrain_lstm_autoencoder(
                            X_seq, epochs=3, batch_size=64  # Minimal epochs for EC2 t2.micro
                        )
                    else:
                        lstm_success = False
                else:
                    lstm_success = False
                
                # Retrain Prophet forecast model
                prophet_success = self.forecast_manager.train_forecast_model(telemetry_data)
                
                self.last_retrain_time = datetime.utcnow()
                logger.info(
                    f"Retraining completed - IF: {if_success}, LSTM: {lstm_success}, Prophet: {prophet_success}"
                )
                
                return if_success and lstm_success
            
            except Exception as e:
                logger.error(f"Error during retraining: {e}")
                return False
            
            finally:
                self.is_retraining = False
    
    def start_retraining_thread(self):
        """Start background retraining thread."""
        def retraining_loop():
            while True:
                try:
                    if self.should_retrain():
                        self.retrain_models()
                    time.sleep(60)  # Check every minute
                except Exception as e:
                    logger.error(f"Error in retraining thread: {e}")
                    time.sleep(60)
        
        self.retrain_thread = threading.Thread(target=retraining_loop, daemon=True)
        self.retrain_thread.start()
        logger.info("Retraining scheduler started")


# Will be instantiated in main.py
retraining_scheduler = None
