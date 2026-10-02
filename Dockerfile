FROM python:3.12-slim

WORKDIR /app

# Install system build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY . .

# Expose FastAPI port (8000), gRPC port (50051), and Prometheus metrics
EXPOSE 8000 50051

CMD ["uvicorn", "serving.api_server:app", "--host", "0.0.0.0", "--port", "8000"]
