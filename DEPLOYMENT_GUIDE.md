# 🐳 Production Deployment Guide — Pro v4.0

This document outlines the containerization, orchestration, persistent storage, and local LLM runtime setup required to deploy the Agentic AI Data Scientist Platform in production.

---

## 1. Overview Architecture

A production deployment runs the FastAPI server, SQLite session database, and ChromaDB vector store inside a unified network, connecting to an Ollama server running locally (or on a dedicated GPU host).

```
[ Web Browser / Client ]
           │ (HTTP / WebSockets)
           ▼
[ Reverse Proxy (Nginx / Caddy) ]
           │ (Port 8000)
           ▼
[ FastAPI App Container (Uvicorn) ] ── (SQLite Volume) ──> [ sessions.db ]
           │
           ├── (ChromaDB Volume) ──> [ data/chroma_db/ ]
           │
           └── (Ollama API Port 11434) ──> [ Ollama Service (GPU or CPU) ]
```

---

## 2. Docker Containerization

We run the platform using a multi-stage Docker setup to keep images lean and lightweight.

### 2.1 Backend Dockerfile (`Dockerfile`)
The repository includes a ready-to-run `Dockerfile` in the root:
```dockerfile
# Stage 1: Build dependencies
FROM python:3.12-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Stage 2: Final image
FROM python:3.12-slim

WORKDIR /app

# Copy python dependencies from builder
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Install standard runtime utils
RUN apt-get update && apt-get install -y --no-install-recommends \
    sqlite3 \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY . .

# Expose FastAPI port
EXPOSE 8000

ENV PYTHONUNBUFFERED=1

CMD ["python", "server.py"]
```

### 2.2 Orchestration (`docker-compose.yml`)
Use the following compose template to orchestrate the server, map persistent database volumes, and automatically run the model preloading hooks:

```yaml
version: '3.8'

services:
  app:
    build: .
    container_name: ai_data_scientist_app
    ports:
      - "8000:8000"
    environment:
      - OLLAMA_HOST=http://ollama:11434
      - SESSION_DB_PATH=/app/data/sessions.db
      - CHROMA_DB_PATH=/app/data/chroma_db
    volumes:
      - platform_data:/app/data
    depends_on:
      - ollama
    restart: unless-stopped

  ollama:
    image: ollama/ollama:latest
    container_name: ai_data_scientist_ollama
    ports:
      - "11434:11434"
    volumes:
      - ollama_storage:/root/.ollama
    # If utilizing an NVIDIA GPU host, uncomment the deploy block below:
    # deploy:
    #   resources:
    #     reservations:
    #       devices:
    #         - driver: nvidia
    #           count: all
    #           capabilities: [gpu]
    restart: unless-stopped

  # Helper service to preload required models automatically on start
  model_loader:
    image: curlimages/curl:latest
    container_name: ollama_model_loader
    depends_on:
      - ollama
    entrypoint: >
      /bin/sh -c "
      echo 'Waiting for Ollama to start...';
      until curl -s http://ollama:11434/api/tags > /dev/null; do sleep 2; done;
      echo 'Pulling nomic-embed-text for memory...';
      curl -X POST http://ollama:11434/api/pull -d '{\"name\": \"nomic-embed-text\"}';
      echo 'Pulling mistral-nemo for reasoning...';
      curl -X POST http://ollama:11434/api/pull -d '{\"name\": \"mistral-nemo\"}';
      echo 'Models loaded successfully!';
      "

volumes:
  platform_data:
    driver: local
  ollama_storage:
    driver: local
```

---

## 3. Environment Variables Configuration

Create a `.env` file in the root directory to manage production settings:

```ini
# FastAPI Service Configuration
PORT=8000
HOST=0.0.0.0
ENV=production

# Local Databases Paths
SESSION_DB_PATH=/app/data/sessions.db
CHROMA_DB_PATH=/app/data/chroma_db

# Ollama LLM Connection Host
OLLAMA_HOST=http://localhost:11434

# Memory Settings
MEMORY_INJECTION_THRESHOLD=0.65
PREFERRED_MODEL_ORDER=nomic-embed-text:latest,mistral-nemo:latest,qwen2.5-coder:7b
```

---

## 4. Production Health Checks & Backups

### 4.1 Health Monitoring Endpoint
To monitor container health inside an orchestrator (Kubernetes, AWS ECS, Docker Swarm), target the platform health API:
```bash
curl -f http://localhost:8000/api/health
```
A healthy response returns `200 OK` with JSON statistics on memory retrieval status, SQLite connection, and Ollama model availability.

### 4.2 SQLite database Backups
Because SQLite uses a single file, backup scripts can run regularly in the background:
```bash
# Safely backup the database without locks using SQLite's backup API
sqlite3 /app/data/sessions.db ".backup /backups/sessions_$(date +%F).db"
```
The ChromaDB vector index folders under `data/chroma_db/` should also be zipped and archived concurrently.
