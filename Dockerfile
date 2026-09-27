FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency definition
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code and models
COPY src/ ./src/
COPY models/ ./models/
COPY data/ ./data/

EXPOSE 8000
EXPOSE 8501

# Default command: runs the FastAPI engine
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
