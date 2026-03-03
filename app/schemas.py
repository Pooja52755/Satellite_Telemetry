from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional, Dict, Any


class TelemetryData(BaseModel):
    """Telemetry data schema."""
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    battery_voltage: float
    bus_voltage: float
    wheel_rpm: float
    current: float
    wheel_temperature: float

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
    """Anomaly detection result schema."""
    timestamp: datetime
    is_anomaly: bool
    isolation_forest_score: Optional[float] = None
    lstm_reconstruction_error: Optional[float] = None
    prophet_deviation: Optional[float] = None
    detection_method: str
    confidence: float = Field(ge=0.0, le=1.0)


class HealthStatus(BaseModel):
    """Health check response."""
    status: str
    timestamp: datetime
    models_loaded: bool
    database_connected: bool
