# Quick Start - No Docker

## Prerequisites
- Python 3.10+
- MongoDB Atlas account (free tier)

## Setup (Windows)

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Create `.env` File
```bash
copy .env.example .env
```
Then edit `.env` and add your MongoDB Atlas URI:
```
MONGODB_URI=mongodb+srv://username:password@cluster.mongodb.net/?retryWrites=true&w=majority
```

### 3. Place Pre-trained Models
Copy these files to `models/` folder:
- `isolation.pkl`
- `lstm.keras`

### 4. Run Application
**Windows:**
```bash
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Or simply run:
```bash
run.bat
```

**Mac/Linux:**
```bash
bash run.sh
```

## Access API

- **API Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health
- **Swagger UI**: http://localhost:8000/redoc

## Test Endpoints

### Health Check
```bash
curl http://localhost:8000/health
```

### Make Prediction
```bash
curl -X POST http://localhost:8000/predict ^
  -H "Content-Type: application/json" ^
  -d "{\"battery_voltage\": 12.5, \"bus_voltage\": 5.0, \"wheel_rpm\": 150.0, \"current\": 2.3, \"wheel_temperature\": 45.2}"
```

### Simulate Stream (100 records)
```bash
curl http://localhost:8000/stream?count=100
```

### Get Anomalies
```bash
curl http://localhost:8000/anomalies?limit=20
```

## File Structure After Setup

```
satellite_telemetry/
├── app/
│   ├── main.py              # FastAPI app
│   ├── models.py            # Model loading
│   ├── database.py          # MongoDB ops
│   ├── inference.py         # Anomaly detection
│   ├── simulator.py         # Data generator
│   ├── retrain.py           # Retraining scheduler
│   ├── forecast.py          # Prophet forecasting
│   └── schemas.py           # Data schemas
├── models/
│   ├── isolation.pkl        # (copy here)
│   └── lstm.keras           # (copy here)
├── .env                     # (create from example)
├── requirements.txt
└── run.bat                  # (Windows start script)
```

## MongoDB Atlas Setup

1. Go to [MongoDB Atlas](https://www.mongodb.com/cloud/atlas)
2. Create free tier cluster
3. Create database user (username/password)
4. Click "Connect" → "Drivers"
5. Copy connection string
6. Replace in `.env`:
   ```
   mongodb+srv://USERNAME:PASSWORD@cluster-name.mongodb.net/?retryWrites=true&w=majority
   ```

## Environment Variables

### Required
- `MONGODB_URI` - MongoDB Atlas connection string

### Optional (defaults shown)
- `RETRAIN_INTERVAL=3600` - Retraining interval in seconds
- `SLIDING_WINDOW_SIZE=3000` - Max records for retraining
- `PORT=8000` - API port
- `HOST=0.0.0.0` - API host

## Troubleshooting

### Models Not Found
```
Error: isolation.pkl not found
```
**Solution**: Copy `isolation.pkl` and `lstm.keras` to `models/` folder

### MongoDB Connection Failed
```
Error: Failed to connect to MongoDB
```
**Solution**: 
- Check `.env` file has correct `MONGODB_URI`
- Verify IP whitelist in MongoDB Atlas (add your IP)
- Test connection: `python -c "from pymongo import MongoClient; MongoClient('YOUR_URI').admin.command('PING')"`

### Port 8000 Already In Use
```
OSError: [Errno 48] Address already in use
```
**Solution**: Kill the process or use different port:
```bash
python -m uvicorn app.main:app --port 9000
```

### Missing Dependencies
```
ModuleNotFoundError: No module named 'fastapi'
```
**Solution**: 
```bash
pip install -r requirements.txt
```

## Development Tips

- Use `--reload` flag for auto-restart on file changes
- Check logs for errors: Look at console output
- Test database: Use MongoDB Compass (free GUI tool)
- Monitor memory: Use Task Manager (Windows) or `top` (Linux)

---

**That's it!** Your system is now running locally without Docker. Perfect for development and testing before EC2 deployment.
