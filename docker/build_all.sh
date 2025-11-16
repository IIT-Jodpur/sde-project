#!/bin/bash
# Build all Docker images for GPU time slicing experiments

set -e

echo "Building Docker images for GPU Time Slicing Project..."
echo "========================================================"

# Base image
echo ""
echo "[1/4] Building base image..."
docker build -f Dockerfile.base -t gpu-timeslicing/base:latest ..

# Training image
echo ""
echo "[2/4] Building training image..."
docker build -f Dockerfile.training -t gpu-timeslicing/training:latest ..

# Inference image
echo ""
echo "[3/4] Building inference image..."
docker build -f Dockerfile.inference -t gpu-timeslicing/inference:latest ..

# Interactive image
echo ""
echo "[4/4] Building interactive image..."
docker build -f Dockerfile.interactive -t gpu-timeslicing/interactive:latest ..

echo ""
echo "========================================================"
echo "All images built successfully!"
echo ""
echo "Available images:"
docker images | grep "gpu-timeslicing"

