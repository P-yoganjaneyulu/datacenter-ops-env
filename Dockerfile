FROM python:3.11-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Import smoke check during build to fail fast on missing modules
RUN python -m py_compile server/app.py environment.py inference.py pre_validation.py

# Container health check via stdlib (no curl dependency)
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import json,urllib.request; s=json.loads(urllib.request.urlopen('http://127.0.0.1:7860/health', timeout=3).read()).get('status'); raise SystemExit(0 if s=='healthy' else 1)"

# Expose port for Hugging Face Spaces
EXPOSE 7860

# Run server - use server.app module
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
