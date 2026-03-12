FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Source
COPY . .

# Set env vars for internal networking
ENV BMR_PORT=8001

# Make start script executable
RUN chmod +x start.sh

# Entrypoint runs the start script
CMD ["./start.sh"]
