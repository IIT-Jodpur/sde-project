#!/bin/bash
# Disable NVIDIA Multi-Instance GPU (MIG)

set -e

echo "Disabling NVIDIA Multi-Instance GPU (MIG)..."
echo "========================================"

# Check if running as root
if [ "$EUID" -ne 0 ]; then 
    echo "Error: MIG configuration requires root privileges."
    echo "Please run with sudo."
    exit 1
fi

# Destroy compute instances
echo "Destroying compute instances..."
nvidia-smi mig -dci || echo "No compute instances to destroy"

# Destroy GPU instances
echo "Destroying GPU instances..."
nvidia-smi mig -dgi || echo "No GPU instances to destroy"

# Disable MIG mode
echo "Disabling MIG mode..."
nvidia-smi -i 0 -mig 0

echo ""
echo "========================================"
echo "MIG disabled successfully!"
echo "A system reboot may be required for changes to take full effect."
echo "========================================"

