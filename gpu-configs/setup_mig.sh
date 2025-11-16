#!/bin/bash
# Configure NVIDIA Multi-Instance GPU (MIG)
# Note: MIG is only supported on A100, A30, and H100 GPUs

set -e

echo "Configuring NVIDIA Multi-Instance GPU (MIG)..."
echo "========================================"

# Check if nvidia-smi is available
if ! command -v nvidia-smi &> /dev/null; then
    echo "Error: nvidia-smi not found. Please install NVIDIA drivers."
    exit 1
fi

# Check if running as root
if [ "$EUID" -ne 0 ]; then 
    echo "Error: MIG configuration requires root privileges."
    echo "Please run with sudo."
    exit 1
fi

# Get GPU information
echo "Checking GPU compatibility..."
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)
echo "Detected GPU: $GPU_NAME"

# Check if GPU supports MIG
if ! nvidia-smi -i 0 --query-gpu=mig.mode.current --format=csv,noheader &> /dev/null; then
    echo ""
    echo "Error: This GPU does not support MIG."
    echo "MIG is only available on A100, A30, and H100 GPUs."
    echo "Current GPU: $GPU_NAME"
    exit 1
fi

# Get current MIG mode
CURRENT_MODE=$(nvidia-smi -i 0 --query-gpu=mig.mode.current --format=csv,noheader)
echo "Current MIG mode: $CURRENT_MODE"

# Enable MIG mode if not already enabled
if [ "$CURRENT_MODE" != "Enabled" ]; then
    echo ""
    echo "Enabling MIG mode..."
    nvidia-smi -i 0 -mig 1
    echo "MIG mode enabled. A system reboot may be required."
    echo "After reboot, run this script again to create MIG instances."
    exit 0
fi

# MIG profile configuration
# Common profiles for A100-40GB:
# - 1g.5gb  : 1 GPU slice, 5GB memory
# - 2g.10gb : 2 GPU slices, 10GB memory  
# - 3g.20gb : 3 GPU slices, 20GB memory
# - 4g.20gb : 4 GPU slices, 20GB memory
# - 7g.40gb : 7 GPU slices, 40GB memory (full GPU)

PROFILE=${1:-"1g.5gb"}
NUM_INSTANCES=${2:-4}

echo ""
echo "Creating MIG instances..."
echo "Profile: $PROFILE"
echo "Number of instances: $NUM_INSTANCES"

# Destroy existing instances
echo "Clearing existing MIG instances..."
nvidia-smi mig -dci || true
nvidia-smi mig -dgi || true

# Create GPU instances
echo "Creating GPU instances..."
for i in $(seq 1 $NUM_INSTANCES); do
    nvidia-smi mig -cgi $(echo $PROFILE | cut -d'.' -f1) -C
done

# Create compute instances
echo "Creating compute instances..."
nvidia-smi mig -cci -gi 0

echo ""
echo "MIG configuration complete!"
echo ""
echo "Created instances:"
nvidia-smi mig -lgi
echo ""
nvidia-smi mig -lci

echo ""
echo "========================================"
echo "MIG configured successfully!"
echo ""
echo "Configuration:"
echo "  Profile: $PROFILE"
echo "  Instances: $NUM_INSTANCES"
echo ""
echo "To use MIG instances in containers, set:"
echo "  NVIDIA_VISIBLE_DEVICES=MIG-<UUID>"
echo ""
echo "To list MIG UUIDs:"
echo "  nvidia-smi -L"
echo ""
echo "To disable MIG:"
echo "  sudo nvidia-smi mig -dci"
echo "  sudo nvidia-smi mig -dgi"
echo "  sudo nvidia-smi -i 0 -mig 0"
echo "========================================"

