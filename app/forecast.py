import numpy as np
from typing import Optional, List, Dict, Any
import logging
from prophet import Prophet
import pandas as pd
import warnings

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


class ForecastManager:
    """Prophet-based forecasting for trend deviation detection."""
    
    def __init__(self):
        self.prophet_model = None
        self.last_forecast_value = None
        self.min_data_required = 50  # Minimum data for forecast
        
    def train_forecast_model(self, data: List[Dict[str, Any]]) -> bool:
        """Train Prophet model with telemetry data."""
        try:
            if len(data) < self.min_data_required:
                logger.warning(f"Insufficient data for forecasting: {len(data)} < {self.min_data_required}")
                return False
            
            # Prepare data for Prophet (battery voltage)
            df_data = []
            for record in data:
                df_data.append({
                    "ds": pd.to_datetime(record["timestamp"]),
                    "y": record.get("battery_voltage", 12.0)
                })
            
            df = pd.DataFrame(df_data)
            
            # Train Prophet with minimal configuration for memory efficiency
            self.prophet_model = Prophet(
                yearly_seasonality=False,
                weekly_seasonality=False,
                daily_seasonality=False,
                interval_width=0.95,
                changepoint_prior_scale=0.05
            )
            self.prophet_model.fit(df)
            
            logger.info(f"Prophet model trained with {len(df)} samples")
            return True
        except Exception as e:
            logger.error(f"Error training Prophet model: {e}")
            return False
    
    def forecast_next(self, recent_data: List[Dict[str, Any]], 
                     periods: int = 1) -> Optional[float]:
        """Forecast next battery voltage value."""
        try:
            if self.prophet_model is None or len(recent_data) == 0:
                return None
            
            # Get last timestamp and create future dataframe
            last_record = recent_data[-1]
            last_timestamp = pd.to_datetime(last_record["timestamp"])
            
            # Simple extrapolation if Prophet not ready
            future = pd.DataFrame([{
                "ds": last_timestamp
            }])
            
            # Make forecast
            forecast = self.prophet_model.make_future_dataframe(periods=periods)
            forecast_result = self.prophet_model.predict(forecast)
            
            # Get last forecasted value
            last_forecast = forecast_result["yhat"].iloc[-1]
            self.last_forecast_value = last_forecast
            
            return float(last_forecast)
        except Exception as e:
            logger.error(f"Error in forecasting: {e}")
            # Return last known value as fallback
            if len(recent_data) > 0:
                return recent_data[-1].get("battery_voltage", 12.0)
            return None


# Global instance
forecast_manager = ForecastManager()
