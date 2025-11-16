#!/bin/bash
# Configure NVIDIA GPU for Time Slicing

set -e

echo "Configuring GPU Time Slicing..."
echo "========================================"

# Check if nvidia-smi is available
if ! command -v nvidia-smi &> /dev/null; then
    echo "Error: nvidia-smi not found. Please install NVIDIA drivers."
    exit 1
fi

# Display current GPU configuration
echo ""
echo "Current GPU Configuration:"
nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv

# Time slicing configuration for Docker/containerd
# This sets up the device plugin to allow multiple containers to share the GPU

echo ""
echo "Setting up Time Slicing configuration..."

# Number of replicas (how many containers can share the GPU)
REPLICAS=${1:-4}

echo "Configuring GPU to support $REPLICAS concurrent containers"

# Create device plugin config for time slicing
cat <<EOF > /tmp/dp-config.yaml
version: v1
sharing:
  timeSlicing:
    replicas: $REPLICAS
    failRequestsGreaterThanOne: true
EOF

echo ""
echo "Time Slicing Configuration (dp-config.yaml):"
cat /tmp/dp-config.yaml

echo ""
echo "========================================"
echo "Time Slicing configured successfully!"
echo ""
echo "Note: To apply this configuration with Kubernetes:"
echo "  1. Apply the ConfigMap:"
echo "     kubectl create -n gpu-operator configmap time-slicing-config --from-file=config.yaml=/tmp/dp-config.yaml"
echo "  2. Patch the ClusterPolicy:"
echo "     kubectl patch clusterpolicy/cluster-policy -n gpu-operator --type merge -p '{\"spec\":{\"devicePlugin\":{\"config\":{\"name\":\"time-slicing-config\"}}}}'"
echo ""
echo "For Docker Compose, use the provided docker-compose.yml"
echo "All containers will share the GPU via time slicing."
echo "========================================"

