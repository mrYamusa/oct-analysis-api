import os
import multiprocessing

# Number of workers - limit to 1 for memory-intensive applications
workers = 1
threads = 4

# Prevent memory leaks with max requests
max_requests = 100
max_requests_jitter = 10

# Set timeouts
timeout = 120
keepalive = 5

# Logging
accesslog = "-"
errorlog = "-"
loglevel = "info"

# Memory optimization
worker_tmp_dir = "/dev/shm" if os.path.exists("/dev/shm") else None
worker_class = "uvicorn.workers.UvicornWorker"

# Process name
proc_name = "retinal-oct-api"

# Preload application to avoid loading the model multiple times
preload_app = True

# Define post_fork to set memory limits
def post_fork(server, worker):
    # Set memory limit for the worker process (in bytes)
    # This is Linux-specific and requires the resource module
    try:
        import resource
        # Set soft limit to 1GB, hard limit to 1.2GB
        # Adjust these values based on your server's available memory
        resource.setrlimit(resource.RLIMIT_AS, (1024 * 1024 * 1024, int(1024 * 1024 * 1024 * 1.2)))
    except (ImportError, ValueError) as e:
        print(f"Could not set memory limit: {e}")