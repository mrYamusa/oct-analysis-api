#!/bin/bash
# This script runs your FastAPI application on Render

# Make the script executable
chmod +x ./start.sh

# Run Gunicorn with your FastAPI app
# Increase timeout to 120 seconds to allow for model loading
exec gunicorn app:app --workers 1 --worker-class uvicorn.workers.UvicornWorker --timeout 120 --bind 0.0.0.0:$PORT