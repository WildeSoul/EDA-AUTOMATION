# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the project files
COPY . .

# Expose ports for FastAPI (8000) and Streamlit (8501)
EXPOSE 8000 8501

# Create a startup script to run both servers
RUN echo '#!/bin/bash\n\
uvicorn server:app --host 0.0.0.0 --port 8000 &\n\
streamlit run app.py --server.port 8501 --server.address 0.0.0.0\n\
' > start.sh && chmod +x start.sh

# Run the startup script
CMD ["./start.sh"]
