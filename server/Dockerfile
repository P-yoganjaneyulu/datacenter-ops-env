# DataCenterOps — Server Dockerfile
# Multi-stage build using openenv-base image (official OpenEnv pattern)

ARG BASE_IMAGE=python:3.11-slim
FROM ${BASE_IMAGE} AS builder

WORKDIR /app

# Install build deps
RUN pip install --no-cache-dir uv

# Copy and install dependencies
COPY server/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy environment code
COPY environment.py /app/
COPY models.py /app/
COPY grader.py /app/
COPY baseline.py /app/
COPY config.json /app/
COPY openenv.yaml /app/
COPY server/ /app/server/
COPY __init__.py /app/
COPY client.py /app/

# Runtime stage
FROM ${BASE_IMAGE}

WORKDIR /app

# Copy installed packages and code from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /app /app

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import json,urllib.request; s=json.loads(urllib.request.urlopen('http://127.0.0.1:7860/health', timeout=2).read()).get('status'); raise SystemExit(0 if s=='healthy' else 1)"

# Set task via env var (override at runtime: -e DATACENTER_TASK=hard)
ENV DATACENTER_TASK=easy
ENV PYTHONPATH=/app

EXPOSE 7860

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
