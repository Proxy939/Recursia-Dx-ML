#!/bin/bash
# RecursiaDx - Unix/Linux/macOS Startup Script

echo ""
echo "========================================"
echo "  RecursiaDx Full Stack Startup"
echo "========================================"
echo ""

# Check if we're in the right directory
if [ ! -f "ml/api/app.py" ]; then
    echo "ERROR: Please run this script from the RecursiaDx root directory"
    exit 1
fi

echo "[1/4] Starting Brain Tumor API (Port 5002)..."
cd ml && python api/brain_tumor_api.py --port 5002 &
cd ..
sleep 3

echo "[2/4] Starting Main ML API (Port 5000)..."
cd ml && python api/app.py &
cd ..
sleep 3

echo "[3/4] Starting Backend Server (Port 5001)..."
cd backend && node server.js &
cd ..
sleep 3

echo "[4/4] Starting Frontend (Port 5173)..."
cd client && npm run dev &
cd ..

echo ""
echo "========================================"
echo "  All Services Started!"
echo "========================================"
echo ""
echo "  Frontend:     http://localhost:5173"
echo "  Backend:      http://localhost:5001"
echo "  ML API:       http://localhost:5000"
echo "  Brain Tumor API: http://localhost:5002"
echo ""
echo "  Press Ctrl+C to stop all services."
echo "========================================"
echo ""

# Wait for user interrupt
wait
