# Satellite Telemetry Anomaly Detection - EC2 Deployment Guide

## Prerequisites
- AWS EC2 t2.micro instance (1GB RAM)
- Ubuntu 22.04 LTS
- Pre-trained models: `models/isolation.pkl` and `models/lstm.keras`
- MongoDB Atlas free tier cluster (URI from Atlas)

## Environment Setup

### 1. Connect to EC2 Instance
```bash
ssh -i your-key.pem ubuntu@your-ec2-ip
```

### 2. Install Docker
```bash
sudo apt update
sudo apt install -y docker.io

# Start Docker service
sudo systemctl start docker
sudo systemctl enable docker

# Add ubuntu user to docker group
sudo usermod -aG docker ubuntu
```

### 3. Create `.env` File
```bash
cd ~/satellite_telemetry

cat > .env << EOF
MONGODB_URI=mongodb+srv://username:password@cluster.mongodb.net/
RETRAIN_INTERVAL=3600
SLIDING_WINDOW_SIZE=3000
PORT=8000
HOST=0.0.0.0
EOF
```

**Replace with your MongoDB Atlas connection string.**

### 4. Build Docker Image
```bash
docker build -t satellite-telemetry:latest .
```

### 5. Run Container
```bash
docker run -d \
  --name satellite-api \
  --env-file .env \
  -p 8000:8000 \
  -v ~/satellite_telemetry/models:/app/models:ro \
  --memory=800m \
  --cpus=0.5 \
  satellite-telemetry:latest
```

### 6. Verify Container
```bash
# Check logs
docker logs satellite-api

# Health check
curl http://localhost:8000/health

# View docs
# Access: http://your-ec2-ip:8000/docs
```

## Memory/CPU Optimization Notes

- **Single worker**: Set in uvicorn config (prevents multi-process overhead)
- **Sliding window**: Max 3000 records (configurable, typically 50MB)
- **Batch size**: LSTM batch size 64 (efficient for available RAM)
- **Epochs**: 3 epochs per retrain (fast, memory-safe)
- **Memory limit**: 800MB Docker limit
- **CPU limit**: 0.5 cores to share with system

## API Endpoints

### Health Check
```bash
curl http://localhost:8000/health
```

### Make Prediction
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "battery_voltage": 12.5,
    "bus_voltage": 5.0,
    "wheel_rpm": 150.0,
    "current": 2.3,
    "wheel_temperature": 45.2
  }'
```

### Get Recent Anomalies
```bash
curl http://localhost:8000/anomalies?limit=20
```

### Simulate Telemetry Stream
```bash
curl http://localhost:8000/stream?count=100
```

### Interactive API Docs
Open browser: `http://your-ec2-ip:8000/docs`

## Monitoring & Maintenance

### View Logs
```bash
docker logs -f satellite-api
```

### Restart Container
```bash
docker restart satellite-api
```

### Stop Container
```bash
docker stop satellite-api
```

### Remove Container
```bash
docker rm satellite-api
```

### Check Memory Usage
```bash
docker stats satellite-api
```

## Database Connection Tips

### MongoDB Atlas Connection
1. Go to [MongoDB Atlas](https://www.mongodb.com/cloud/atlas)
2. Create free tier cluster
3. Get connection string from "Connect" button
4. String format: `mongodb+srv://user:pass@cluster.mongodb.net/?retryWrites=true&w=majority`
5. Replace in `.env` file

### Network Security
- Add EC2 IP to MongoDB Atlas IP Whitelist (0.0.0.0/0 for demo, restrict for production)
- Ensure MongoDB URI has credentials encoded

## Performance Tuning

### For Limited RAM (1GB)
```bash
# Reduce sliding window size
SLIDING_WINDOW_SIZE=1000  # Instead of 3000

# Reduce batch size
# Edit app/retrain.py: batch_size=32 (instead of 64)

# Reduce Prophet data
# Edit app/retrain.py: limit=50 (instead of 100)
```

### For CPU Optimization
```bash
# Already optimized:
# - Single uvicorn worker
# - Minimal epochs (3)
# - Threading for retraining
# - Efficient numpy operations
```

## Troubleshooting

### MongoDB Connection Failed
- Check `.env` file has correct URI
- Verify IP is whitelisted in MongoDB Atlas
- Check network connectivity: `curl -I mongodb+srv://...`

### Out of Memory (OOM)
```bash
# Reduce window size or batch size
# Check: docker stats satellite-api
# If >90% used, adjust parameters
```

### Port 8000 Already In Use
```bash
# Check what's using port 8000
sudo lsof -i :8000

# Use different port
docker run -p 9000:8000 ...
```

### Models Not Loading
- Verify `models/` directory exists with `.pkl` and `.keras` files
- Check file permissions: `ls -la models/`

## Auto-Start on Reboot
```bash
docker update --restart always satellite-api
```

## Deployment Notes

- **Retrain interval**: Every 60 minutes (configurable)
- **Sliding window**: 3000 latest records (configurable)
- **Anomaly detection**: Multi-method (IF, LSTM, Prophet)
- **Database**: Write-heavy (stores all telemetry + anomalies)
- **API response time**: <500ms typical

---

**For production**, consider:
- Setup CloudWatch for monitoring
- Use AWS RDS instead of MongoDB Atlas
- Enable VPC security groups
- Setup SSL certificates for HTTPS
- Consider AWS Lambda for serverless (but cold starts may be slow)
