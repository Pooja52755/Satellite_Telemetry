@echo off
REM Quick start script for Windows

echo Installing dependencies...
pip install -r requirements.txt

echo.
echo Creating .env file...
if not exist .env (
    copy .env.example .env
    echo .env file created. Please edit it with your MongoDB URI.
    pause
)

echo.
echo Starting application...
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
