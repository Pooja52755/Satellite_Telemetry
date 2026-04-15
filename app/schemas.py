"""Pydantic schemas for request/response validation."""

from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional, List


class TelemetryData(BaseModel):
    """Telemetry data schema."""
    timestamp: Optional[datetime] = Field(default_factory=datetime.utcnow)
    battery_voltage: float = Field(gt=0, description="Battery voltage in volts")
    bus_voltage: float = Field(gt=0, description="Bus voltage in volts")
    wheel_rpm: float = Field(ge=0, description="Wheel RPM")
    current: float = Field(ge=0, description="Current in amperes")
    wheel_temperature: float = Field(description="Wheel temperature in celsius")

    class Config:
        json_schema_extra = {
            "example": {
                "battery_voltage": 12.5,
                "bus_voltage": 5.0,
                "wheel_rpm": 150.0,
                "current": 2.3,
                "wheel_temperature": 45.2
            }
        }


class AnomalyResult(BaseModel):
    """Anomaly detection result."""
    timestamp: datetime
    is_anomaly: bool
    confidence: float = Field(ge=0.0, le=1.0)
    anomaly_score: float
    detection_method: str = "isolation_forest"


class ForecastResult(BaseModel):
    """Forecast prediction result."""
    timestamp: datetime
    feature: str
    forecast_value: float
    lower_bound: Optional[float] = None
    upper_bound: Optional[float] = None


class HealthStatus(BaseModel):
    """Health check response."""
    status: str
    timestamp: datetime
    models_loaded: bool
    database_connected: bool
    prophet_fitted: bool


class StatisticsResponse(BaseModel):
    """Statistics response."""
    total_records: int
    anomaly_count: int
    anomaly_percentage: float
    latest_timestamp: Optional[datetime] = None


class ForecasterStats(BaseModel):
    """Forecaster statistics."""
    is_fitted: bool
    feature: str
    last_training_data_points: int
