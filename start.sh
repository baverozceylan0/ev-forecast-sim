#!/bin/bash

mkdir -p ./models/{agg,usr}
mkdir -p ./data/raw
mkdir -p ./outputs

# Optional: run other setup steps here
echo "Starting MLflow UI..."
mlflow ui --host 0.0.0.0 --port 5000 > /app/outputs/mlflow.log 2>&1 &

sleep 2

# Keep container alive
tail -f /app/outputs/mlflow.log