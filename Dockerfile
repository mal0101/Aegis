FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends gcc && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code
COPY backend/app ./app
COPY backend/scripts ./scripts

# Create storage directory
RUN mkdir -p /app/storage/chromadb

# Note: Ollama must be running on the host or accessible via network
# Set OLLAMA_BASE_URL env var if Ollama is not on localhost
# ENV OLLAMA_BASE_URL=http://host.docker.internal:11434/v1
# RUN python -m scripts.load_data

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
