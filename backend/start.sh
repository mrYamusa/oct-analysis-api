#!/bin/bash
# This script runs your FastAPI application on Render with memory optimizations

# Export environment variables to limit memory usage
export MALLOC_ARENA_MAX=2  # Limit number of memory pools
export PYTHONMALLOC=malloc  # Use system malloc

echo "Starting OCT Image Analysis API..."
echo "Using optimized memory settings"

# Check if we have enough memory (if free command is available)
if command -v free &> /dev/null; then
    free -m
fi

# Check if PORT is set (for Render), otherwise use default 8000
PORT="${PORT:-8000}"

# Check if gunicorn_config.py exists, use it if available
if [ -f "gunicorn_config.py" ]; then
    echo "Using gunicorn_config.py for configuration"
    exec gunicorn app:app \
        -c gunicorn_config.py \
        --preload \
        --bind 0.0.0.0:$PORT
else
    # Fallback to command-line options if no config file
    echo "No gunicorn_config.py found, using command line options"
    exec gunicorn app:app \
        --workers 1 \
        --worker-class uvicorn.workers.UvicornWorker \
        --timeout 120 \
        --max-requests 100 \
        --max-requests-jitter 10 \
        --preload \
        --bind 0.0.0.0:$PORT
fi