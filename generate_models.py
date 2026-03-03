"""
Generate dummy trained models for testing purposes.
Run this script to create isolation.pkl and lstm.keras files.
"""

import os
import pickle
import numpy as np
from sklearn.ensemble import IsolationForest
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, RepeatVector, TimeDistributed, Dense
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_models():
    """Generate dummy trained models."""
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    
    # Generate synthetic training data
    logger.info("Generating synthetic training data...")
    np.random.seed(42)
    
    # 1000 samples, 5 features (battery_voltage, bus_voltage, wheel_rpm, current, wheel_temperature)
    X_train = np.random.randn(1000, 5) * np.array([1, 0.5, 20, 0.5, 5]) + np.array([12, 5, 100, 2, 40])
    
    # Train Isolation Forest
    logger.info("Training Isolation Forest...")
    if_model = IsolationForest(
        n_estimators=100,
        contamination=0.05,
        random_state=42,
        n_jobs=-1
    )
    if_model.fit(X_train)
    
    # Save Isolation Forest
    if_path = os.path.join(models_dir, "isolation.pkl")
    with open(if_path, "wb") as f:
        pickle.dump(if_model, f)
    logger.info(f"✓ Isolation Forest saved to {if_path}")
    
    # Train LSTM Autoencoder
    logger.info("Building LSTM Autoencoder...")
    lstm_model = Sequential([
        LSTM(32, activation='relu', input_shape=(10, 5), return_sequences=False),
        RepeatVector(10),
        LSTM(32, activation='relu', return_sequences=True),
        TimeDistributed(Dense(5))
    ])
    lstm_model.compile(optimizer='adam', loss='mse')
    
    # Create sequences for LSTM training
    logger.info("Creating sequences for LSTM training...")
    sequences = []
    sequence_length = 10
    for i in range(len(X_train) - sequence_length):
        sequences.append(X_train[i:i + sequence_length])
    X_sequences = np.array(sequences)
    
    # Train LSTM (minimal training for quick testing)
    logger.info(f"Training LSTM with {len(X_sequences)} sequences...")
    lstm_model.fit(
        X_sequences, X_sequences,
        epochs=5,
        batch_size=32,
        verbose=1,
        validation_split=0.2
    )
    
    # Save LSTM
    lstm_path = os.path.join(models_dir, "lstm.keras")
    lstm_model.save(lstm_path)
    logger.info(f"✓ LSTM Autoencoder saved to {lstm_path}")
    
    logger.info("")
    logger.info("✓ Models generated successfully!")
    logger.info(f"  - {if_path}")
    logger.info(f"  - {lstm_path}")
    logger.info("")
    logger.info("You can now run the application:")
    logger.info("  python -m uvicorn app.main:app --reload")

if __name__ == "__main__":
    try:
        generate_models()
    except Exception as e:
        logger.error(f"Error generating models: {e}")
        raise
