#!/bin/bash

# Start BMR API in background
# We use port 8001 internally for the API
echo "Starting BMR API on port 8001..."
python bmr_server.py &

# Start Streamlit Dashboard
# Cloud Run injects the PORT env var (usually 8080).
# We serve the dashboard as the main interface.
echo "Starting Streamlit Dashboard on port ${PORT:-8080}..."
streamlit run ndis/dashboard.py --server.port "${PORT:-8080}" --server.address 0.0.0.0
