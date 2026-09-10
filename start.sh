#!/bin/bash

set -e

echo "Starting FastAPI..."
uvicorn main:app --host 0.0.0.0 --port 8000 &

echo "Starting Streamlit..."
exec streamlit run irrigation.py \
    --server.address 0.0.0.0 \
    --server.port 8501 \
    --server.headless true