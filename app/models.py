import os
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.ensemble import IsolationForest
import logging
from typing import Tuple, Optional

logger = logging.getLogger(__name__)


class MockIsolationForest:
    """Mock Isolation Forest for testing when model file is unavailable."""
    
    def predict(self, X):
        """Return normal prediction (1 for all samples)."""
        return np.ones(len(X), dtype=int)
    
    def score_samples(self, X):
        """Return normal scores."""
        return np.zeros(len(X))
    
    def fit(self, X):
        """Dummy fit."""
        logger.info("MockIsolationForest.fit() called")
        return self


class ModelManager:
    """Manage pre-trained models (IF, LSTM)."""
    
    def __init__(self):
        self.isolation_forest = None
        self.lstm_autoencoder = None
        self.models_path = os.path.join(os.path.dirname(__file__), "..", "models")
        self.using_mock_models = False
        
    def load_models(self) -> bool:
        """Load pre-trained models at startup."""
        try:
            # Load Isolation Forest
            if_path = os.path.join(self.models_path, "isolation.pkl")
            if os.path.exists(if_path):
                try:
                    with open(if_path, "rb") as f:
                        self.isolation_forest = pickle.load(f)
                    logger.info("Isolation Forest model loaded successfully")
                except Exception as e:
                    logger.warning(f"Failed to load Isolation Forest pickle: {e}. Using mock model.")
                    self.isolation_forest = MockIsolationForest()
                    self.using_mock_models = True
            else:
                logger.warning(f"Isolation Forest model not found at {if_path}. Using mock model.")
                self.isolation_forest = MockIsolationForest()
                self.using_mock_models = True
            
            # Load LSTM Autoencoder
            lstm_path = os.path.join(self.models_path, "lstm.keras")
            if os.path.exists(lstm_path):
                try:
                    self.lstm_autoencoder = load_model(lstm_path)
                    logger.info("LSTM Autoencoder model loaded successfully")
                except Exception as e:
                    logger.warning(f"Failed to load LSTM model: {e}. Creating mock LSTM.")
                    self.lstm_autoencoder = self._create_mock_lstm()
                    self.using_mock_models = True
            else:
                logger.warning(f"LSTM model not found at {lstm_path}. Creating mock LSTM.")
                self.lstm_autoencoder = self._create_mock_lstm()
                self.using_mock_models = True
            
            if self.using_mock_models:
                logger.warning("⚠️  Using MOCK MODELS for testing. Replace models/isolation.pkl and models/lstm.keras with real trained models for production.")
            
            return True
        except Exception as e:
            logger.error(f"Critical error loading models: {e}")
            return False
    
    def _create_mock_lstm(self):
        """Create a simple mock LSTM for testing."""
        try:
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import LSTM, RepeatVector, TimeDistributed, Dense
            
            model = Sequential([
                LSTM(32, activation='relu', input_shape=(10, 5), return_sequences=False),
                RepeatVector(10),
                LSTM(32, activation='relu', return_sequences=True),
                TimeDistributed(Dense(5))
            ])
            model.compile(optimizer='adam', loss='mse')
            logger.info("Created mock LSTM autoencoder")
            return model
        except Exception as e:
            logger.error(f"Failed to create mock LSTM: {e}")
            return None
    
    def get_isolation_forest(self):
        """Get Isolation Forest model."""
        return self.isolation_forest
    
    def get_lstm_autoencoder(self):
        """Get LSTM Autoencoder model."""
        return self.lstm_autoencoder
    
    def retrain_isolation_forest(self, X: np.ndarray) -> bool:
        """Retrain Isolation Forest with new data."""
        try:
            if self.isolation_forest is None:
                logger.warning("Isolation Forest not loaded")
                return False
            
            self.isolation_forest.fit(X)
            logger.info(f"Isolation Forest retrained with {len(X)} samples")
            return True
        except Exception as e:
            logger.error(f"Error retraining Isolation Forest: {e}")
            return False
    
    def retrain_lstm_autoencoder(self, X: np.ndarray, epochs: int = 5, batch_size: int = 32) -> bool:
        """Retrain LSTM Autoencoder with new data."""
        try:
            if self.lstm_autoencoder is None:
                logger.warning("LSTM Autoencoder not loaded")
                return False
            
            # Memory-efficient training
            self.lstm_autoencoder.fit(
                X, X,
                epochs=epochs,
                batch_size=batch_size,
                verbose=0,
                shuffle=True
            )
            logger.info(f"LSTM Autoencoder retrained with {len(X)} samples")
            return True
        except Exception as e:
            logger.error(f"Error retraining LSTM: {e}")
            return False


# Global instance
model_manager = ModelManager()
