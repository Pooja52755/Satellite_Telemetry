#!/bin/bash
# Quick start script for Mac/Linux

echo "Installing dependencies..."
pip install -r requirements.txt

echo ""
echo "Creating .env file..."
if [ ! -f .env ]; then
    cp .env.example .env
    echo ".env file created. Please edit it with your MongoDB URI."
fi

echo ""
echo "Starting application..."
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
