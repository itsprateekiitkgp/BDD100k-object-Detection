#!/bin/bash
# ====================================================
# Simple runner for Dockerized analysis.py
# Usage: ./run_analysis.sh --mode summary
# ====================================================

# Image name (change if you like)
IMAGE_NAME="analysis-app"

# Build the image if it doesn't exist
if ! docker image inspect $IMAGE_NAME > /dev/null 2>&1; then
    echo "🔧 Building Docker image..."
    docker build -t $IMAGE_NAME .
fi

# Run the container with any arguments passed to the script
docker run --rm \
    -v "$(pwd)":/app \
    $IMAGE_NAME "$@"

