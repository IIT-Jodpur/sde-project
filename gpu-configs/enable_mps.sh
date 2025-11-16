#!/bin/bash
# Enable NVIDIA Multi-Process Service (MPS)

set -e

echo "Enabling NVIDIA Multi-Process Service (MPS)..."
echo "========================================"

# Check if nvidia-smi is available
if ! command -v nvidia-smi &> /dev/null; then
    echo "Error: nvidia-smi not found. Please install NVIDIA drivers."
    exit 1
fi

# Check if running as root
if [ "$EUID" -ne 0 ]; then 
    echo "Warning: MPS setup typically requires root privileges."
    echo "If this fails, please run with sudo."
fi

# Set MPS pipe directory
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log

# Create directories if they don't exist
mkdir -p $CUDA_MPS_PIPE_DIRECTORY
mkdir -p $CUDA_MPS_LOG_DIRECTORY

# Check if MPS is already running
if pgrep -x "nvidia-cuda-mps" > /dev/null; then
    echo "MPS is already running. Restarting..."
    echo "quit" | nvidia-cuda-mps-control
    sleep 2
fi

# Start MPS control daemon
echo "Starting MPS control daemon..."
nvidia-cuda-mps-control -d

# Configure MPS settings
echo ""
echo "Configuring MPS settings..."

# Set default active thread percentage (limits GPU compute partition per client)
# Default: 100 (each client can use full GPU)
THREAD_PERCENTAGE=${1:-100}
echo "set_default_active_thread_percentage $THREAD_PERCENTAGE" | nvidia-cuda-mps-control

# Set default device pinned memory limit (MB per client)
# Default: unlimited (set to 0)
PINNED_MEM_LIMIT=${2:-0}
if [ $PINNED_MEM_LIMIT -gt 0 ]; then
    echo "set_default_device_pinned_mem_limit $PINNED_MEM_LIMIT" | nvidia-cuda-mps-control
fi

echo ""
echo "MPS Status:"
echo "get_server_list" | nvidia-cuda-mps-control

echo ""
echo "========================================"
echo "MPS enabled successfully!"
echo ""
echo "Environment variables set:"
echo "  CUDA_MPS_PIPE_DIRECTORY=$CUDA_MPS_PIPE_DIRECTORY"
echo "  CUDA_MPS_LOG_DIRECTORY=$CUDA_MPS_LOG_DIRECTORY"
echo ""
echo "Configuration:"
echo "  Active thread percentage: $THREAD_PERCENTAGE%"
echo "  Pinned memory limit: $PINNED_MEM_LIMIT MB"
echo ""
echo "To use MPS in containers, mount these directories:"
echo "  -v $CUDA_MPS_PIPE_DIRECTORY:$CUDA_MPS_PIPE_DIRECTORY"
echo "  -v $CUDA_MPS_LOG_DIRECTORY:$CUDA_MPS_LOG_DIRECTORY"
echo ""
echo "To disable MPS, run: ./disable_mps.sh"
echo "========================================"

