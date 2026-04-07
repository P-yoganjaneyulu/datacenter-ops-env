FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Expose port for Hugging Face Spaces
EXPOSE 7860

# Run server - use server.app module
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
