#!/bin/bash
# This script runs your FastAPI application on Render

# Make the script executable
chmod +x ./start.sh

# Run Gunicorn with your FastAPI app
exec gunicorn app:app --workers 1 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:$PORT