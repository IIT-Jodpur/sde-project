#!/bin/bash
# Disable NVIDIA Multi-Process Service (MPS)

set -e

echo "Disabling NVIDIA Multi-Process Service (MPS)..."
echo "========================================"

# Check if MPS is running
if ! pgrep -x "nvidia-cuda-mps" > /dev/null; then
    echo "MPS is not running."
    exit 0
fi

# Stop MPS
echo "Stopping MPS control daemon..."
echo "quit" | nvidia-cuda-mps-control

# Wait for shutdown
sleep 2

# Verify MPS is stopped
if pgrep -x "nvidia-cuda-mps" > /dev/null; then
    echo "Warning: MPS may still be running. Force killing..."
    pkill -9 nvidia-cuda-mps || true
fi

echo ""
echo "========================================"
echo "MPS disabled successfully!"
echo "========================================"

