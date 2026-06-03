# syntax=docker/dockerfile:1
# =============================================================================
#  surveychat – Dockerfile
# =============================================================================
#
#  Build:
#      docker build -t surveychat .
#
#  Run (pass secrets at runtime, never bake them into the image):
#      docker run --rm -p 8501:8501 --env-file .env surveychat
#
#  Then open http://localhost:8501 in your browser.
#
# =============================================================================

FROM python:3.12-slim

# Keeps Python from writing .pyc files and enables unbuffered stdout/stderr
# so logs appear immediately in `docker logs`.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Needed by the Docker HEALTHCHECK below.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies first so this layer is cached unless requirements.txt
# changes - speeds up rebuilds during development.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project.
# .dockerignore keeps .env, .git, and __pycache__ out of the image.
COPY . .

# Streamlit listens on 8501 by default.
EXPOSE 8501

# Health check - Docker will mark the container unhealthy if Streamlit stops
# responding, which is useful when running behind a load balancer.
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Run the app.
# CMD (not ENTRYPOINT) so deployment platforms can override this command
# for one-off / release containers without "-c" being misrouted to Streamlit.
# --server.headless=true   suppresses the browser-open prompt.
# --server.address=0.0.0.0 makes the app reachable outside the container.
CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
