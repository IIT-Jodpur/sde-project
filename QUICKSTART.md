# Quick Start Guide

This guide will help you get started with GPU time slicing experiments.

## Prerequisites

Before you begin, ensure you have:

1. **NVIDIA GPU** with latest drivers (460.x or newer)
2. **NVIDIA Container Toolkit** installed
3. **Docker** (20.10 or newer)
4. **Python 3.8+** with pip
5. **(Optional) Kubernetes cluster** for orchestration

## Installation Steps

### 1. Verify GPU Setup

```bash
# Check GPU availability
nvidia-smi

# Verify Docker can access GPU
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

### 2. Install Python Dependencies

```bash
# Clone or navigate to project directory
cd /Users/subramanianmariappan/Desktop/Personal/IIT/M.Tech/SDE

# Install requirements
pip install -r requirements.txt
```

### 3. Build Docker Images

```bash
cd docker
chmod +x build_all.sh
./build_all.sh
```

This will build:
- `gpu-timeslicing/base:latest`
- `gpu-timeslicing/training:latest`
- `gpu-timeslicing/inference:latest`
- `gpu-timeslicing/interactive:latest`

## Running Experiments

### Experiment 1: Single Workload Baseline

Run a single training workload to establish baseline performance:

```bash
# Native execution
python3 workloads/training/resnet_training.py --epochs 5

# Container execution
docker run --gpus all -v $(pwd)/results:/workspace/results \
  gpu-timeslicing/training:latest
```

### Experiment 2: Time Slicing (4 Concurrent Workloads)

```bash
# Setup time slicing
cd gpu-configs
chmod +x setup_time_slicing.sh
./setup_time_slicing.sh 4

# Run concurrent workloads using Docker Compose
cd ../docker
docker-compose up
```

This runs:
- 2x Training workloads (ResNet + BERT)
- 1x Inference server
- 1x Interactive simulation

### Experiment 3: NVIDIA MPS

```bash
# Enable MPS
cd gpu-configs
chmod +x enable_mps.sh
sudo ./enable_mps.sh

# Run workloads with MPS
docker run --gpus all \
  -v /tmp/nvidia-mps:/tmp/nvidia-mps \
  -v /tmp/nvidia-log:/tmp/nvidia-log \
  -e CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps \
  -e CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log \
  -v $(pwd)/results:/workspace/results \
  gpu-timeslicing/training:latest &

docker run --gpus all \
  -v /tmp/nvidia-mps:/tmp/nvidia-mps \
  -v /tmp/nvidia-log:/tmp/nvidia-log \
  -e CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps \
  -e CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log \
  -v $(pwd)/results:/workspace/results \
  gpu-timeslicing/inference:latest &

# Disable MPS when done
./disable_mps.sh
```

### Experiment 4: MIG (A100/H100 only)

```bash
# Enable MIG and create 4 instances
cd gpu-configs
chmod +x setup_mig.sh
sudo ./setup_mig.sh 1g.5gb 4

# List MIG UUIDs
nvidia-smi -L

# Run workload on specific MIG instance
docker run --gpus '"device=MIG-<UUID>"' \
  -v $(pwd)/results:/workspace/results \
  gpu-timeslicing/training:latest

# Disable MIG when done
sudo ./disable_mig.sh
```

## Benchmarking

### Run Comprehensive Benchmark

```bash
# Benchmark time slicing with mixed workloads
python3 benchmarking/benchmark_suite.py \
  --mode time-slicing \
  --workloads 4 \
  --mix mixed \
  --duration 600

# Benchmark MPS
python3 benchmarking/benchmark_suite.py \
  --mode mps \
  --workloads 4 \
  --mix mixed \
  --duration 600
```

### Monitor GPU During Experiments

In a separate terminal:

```bash
# Real-time monitoring
python3 monitoring/gpu_monitor.py --interval 1.0 --duration 600

# Or use nvidia-smi
watch -n 1 nvidia-smi
```

### Analyze Results

```bash
# Generate comparison charts and analysis
python3 benchmarking/analyze_results.py \
  --results-dir results \
  --generate-plots

# View report
cat results/analysis_report.txt

# View charts
open results/mode_comparison.png
open results/utilization_timeline.png
```

## Testing Inference Server

Start inference server:

```bash
docker run --gpus all -p 5000:5000 \
  gpu-timeslicing/inference:latest
```

Test with sample requests:

```bash
# Health check
curl http://localhost:5000/health

# Batch inference benchmark
curl -X POST http://localhost:5000/batch_predict \
  -H "Content-Type: application/json" \
  -d '{"batch_size": 32}'

# View statistics
curl http://localhost:5000/stats
```

## Common Issues

### Issue: "no CUDA-capable device is detected"

**Solution:**
```bash
# Verify nvidia-smi works
nvidia-smi

# Check Docker runtime
docker info | grep -i nvidia

# Reinstall NVIDIA Container Toolkit if needed
```

### Issue: Docker can't access GPU

**Solution:**
```bash
# Add nvidia runtime to Docker daemon.json
sudo nano /etc/docker/daemon.json

# Add:
{
  "runtimes": {
    "nvidia": {
      "path": "nvidia-container-runtime",
      "runtimeArgs": []
    }
  },
  "default-runtime": "nvidia"
}

# Restart Docker
sudo systemctl restart docker
```

### Issue: Out of memory errors

**Solution:**
- Reduce batch size in workloads
- Use smaller models
- Reduce number of concurrent workloads

### Issue: MPS not working

**Solution:**
```bash
# Check if MPS daemon is running
pgrep nvidia-cuda-mps

# Check logs
cat /tmp/nvidia-log/control.log

# Restart MPS
./gpu-configs/disable_mps.sh
sudo ./gpu-configs/enable_mps.sh
```

## Next Steps

1. **Customize Workloads**: Modify scripts in `workloads/` for your use case
2. **Add Metrics**: Extend monitoring to track application-specific metrics
3. **Kubernetes**: Deploy on K8s cluster using manifests in `kubernetes/`
4. **Research**: Analyze results and document findings

## Getting Help

- Check README.md for detailed documentation
- Review gpu-configs/README.md for configuration details
- Check kubernetes/README.md for K8s deployment guide
- Open an issue for bugs or questions

## Example Workflow

Complete example from setup to analysis:

```bash
# 1. Setup
cd /Users/subramanianmariappan/Desktop/Personal/IIT/M.Tech/SDE
pip install -r requirements.txt
cd docker && ./build_all.sh && cd ..

# 2. Run baseline (exclusive mode)
python3 workloads/training/resnet_training.py --epochs 5

# 3. Setup time slicing
./gpu-configs/setup_time_slicing.sh 4

# 4. Run concurrent workloads with monitoring
python3 monitoring/gpu_monitor.py --duration 300 --output results/monitor_timeslice.json &
MONITOR_PID=$!

python3 benchmarking/benchmark_suite.py --mode time-slicing --workloads 4 --mix mixed

kill $MONITOR_PID

# 5. Analyze results
python3 benchmarking/analyze_results.py \
  --results-dir results \
  --monitor-file results/monitor_timeslice.json \
  --generate-plots

# 6. View results
cat results/analysis_report.txt
open results/mode_comparison.png
```

Happy experimenting! 🚀

