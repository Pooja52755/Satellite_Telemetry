import random
from datetime import datetime
from typing import Dict, Any
import math


class TelemetrySimulator:
    """Simulate satellite telemetry data with realistic variations."""
    
    def __init__(self):
        self.base_voltage = 12.0
        self.base_bus = 5.0
        self.base_rpm = 100.0
        self.base_current = 2.0
        self.base_temp = 40.0
        self.anomaly_probability = 0.05  # 5% anomaly rate
        self.time_step = 0
        
    def generate_normal(self) -> Dict[str, Any]:
        """Generate normal telemetry data with realistic variations."""
        self.time_step += 1
        
        # Add normal variations using sine waves and noise
        time_factor = math.sin(self.time_step * 0.01) * 0.5
        
        return {
            "timestamp": datetime.utcnow(),
            "battery_voltage": self.base_voltage + random.gauss(0, 0.3) + time_factor,
            "bus_voltage": self.base_bus + random.gauss(0, 0.1),
            "wheel_rpm": self.base_rpm + random.gauss(0, 10) + time_factor * 20,
            "current": self.base_current + random.gauss(0, 0.15),
            "wheel_temperature": self.base_temp + random.gauss(0, 2) + time_factor * 5,
        }
    
    def generate_anomaly(self) -> Dict[str, Any]:
        """Generate anomalous telemetry data."""
        # Simulate various anomaly types
        anomaly_type = random.choice(["voltage_surge", "rpm_spike", "temperature_rise", "current_drop"])
        
        data = self.generate_normal()
        
        if anomaly_type == "voltage_surge":
            data["battery_voltage"] = data["battery_voltage"] * random.uniform(1.5, 2.0)
        elif anomaly_type == "rpm_spike":
            data["wheel_rpm"] = data["wheel_rpm"] * random.uniform(2.0, 3.0)
        elif anomaly_type == "temperature_rise":
            data["wheel_temperature"] += random.uniform(20, 40)
        elif anomaly_type == "current_drop":
            data["current"] = data["current"] * random.uniform(0.1, 0.3)
        
        return data
    
    def generate(self) -> Dict[str, Any]:
        """Generate telemetry data (normal or anomalous)."""
        if random.random() < self.anomaly_probability:
            return self.generate_anomaly()
        return self.generate_normal()
    
    def batch_generate(self, count: int) -> list:
        """Generate batch of telemetry records."""
        return [self.generate() for _ in range(count)]


# Global instance
simulator = TelemetrySimulator()
