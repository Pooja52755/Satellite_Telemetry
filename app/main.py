import logging
import os
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import asyncio
from contextlib import asynccontextmanager

# Import all modules
from app.schemas import TelemetryData, AnomalyResult, HealthStatus
from app.database import db_manager
from app.models import model_manager
from app.simulator import simulator
from app.inference import AnomalyDetector, anomaly_detector as ad
from app.forecast import forecast_manager
from app.retrain import RetrainingScheduler

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global state
recent_telemetry = []
anomaly_detector = None
retraining_scheduler = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage startup and shutdown."""
    # Startup
    logger.info("Application starting up...")
    
    # Connect to MongoDB
    db_connected = db_manager.connect()
    if not db_connected:
        logger.warning("MongoDB connection failed - proceeding with degraded mode")
    
    # Load models (will use mock models if real ones unavailable)
    model_manager.load_models()
    
    # Initialize anomaly detector
    global anomaly_detector
    anomaly_detector = AnomalyDetector(model_manager, forecast_manager)
    
    # Initialize retraining scheduler
    global retraining_scheduler
    retraining_scheduler = RetrainingScheduler(model_manager, db_manager, forecast_manager)
    retraining_scheduler.start_retraining_thread()
    
    # Train initial Prophet model if data exists
    initial_data = db_manager.get_recent_telemetry(limit=100)
    if len(initial_data) > 50:
        forecast_manager.train_forecast_model(initial_data)
    
    logger.info("Application started successfully")
    
    yield
    
    # Shutdown
    logger.info("Application shutting down...")
    db_manager.close()
    logger.info("Application shutdown complete")


# Create FastAPI app
app = FastAPI(
    title="Satellite Telemetry Anomaly Detection",
    description="Real-time anomaly detection for satellite telemetry",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/health", response_model=HealthStatus)
async def health_check():
    """Health check endpoint."""
    return HealthStatus(
        status="healthy",
        timestamp=datetime.utcnow(),
        models_loaded=model_manager.isolation_forest is not None,
        database_connected=db_manager.db is not None
    )


@app.post("/predict", response_model=AnomalyResult)
async def predict(data: TelemetryData):
    """Predict anomalies for given telemetry data."""
    if anomaly_detector is None:
        raise HTTPException(status_code=500, detail="Anomaly detector not initialized")
    
    try:
        # Get recent telemetry for context
        recent = db_manager.get_recent_telemetry(limit=100)
        
        # Make prediction
        result = anomaly_detector.predict(data.model_dump(), recent)
        
        # Store telemetry
        db_manager.insert_telemetry(data.model_dump())
        
        # Store anomaly if detected
        if result.get("is_anomaly"):
            db_manager.insert_anomaly(result)
        
        return AnomalyResult(**result)
    
    except Exception as e:
        logger.error(f"Error in prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/anomalies")
async def get_anomalies(limit: int = 50):
    """Get recent detected anomalies."""
    try:
        anomalies = db_manager.get_recent_anomalies(limit=limit)
        return {
            "count": len(anomalies),
            "anomalies": anomalies
        }
    except Exception as e:
        logger.error(f"Error fetching anomalies: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stream")
async def simulate_stream(count: int = 10):
    """Simulate continuous telemetry stream and detect anomalies."""
    try:
        results = []
        
        for _ in range(count):
            # Generate simulated telemetry
            data = simulator.generate()
            telemetry = TelemetryData(**data)
            
            # Get prediction
            recent = db_manager.get_recent_telemetry(limit=50)
            prediction = anomaly_detector.predict(data, recent)
            
            # Store data
            db_manager.insert_telemetry(data)
            if prediction.get("is_anomaly"):
                db_manager.insert_anomaly(prediction)
            
            results.append({
                "telemetry": data,
                "anomaly_result": prediction
            })
            
            # Small delay to allow async operations
            await asyncio.sleep(0.01)
        
        return {
            "generated_count": count,
            "results": results
        }
    
    except Exception as e:
        logger.error(f"Error in stream simulation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Satellite Telemetry Anomaly Detection System",
        "endpoints": {
            "health": "/health",
            "predict": "/predict (POST)",
            "anomalies": "/anomalies",
            "stream": "/stream",
            "docs": "/docs"
        }
    }


if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    
    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        workers=1,  # Single worker for t2.micro
        reload=False
    )
