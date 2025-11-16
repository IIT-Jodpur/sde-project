# GPU Time Slicing for Containerized ML Workloads

## Project Overview

This project explores GPU time slicing as a cost-efficient method to run multiple containerized machine learning workloads on a single GPU. By experimenting with different GPU sharing modes—such as time slicing, NVIDIA MPS, and MIG (Multi-Instance GPU)—we aim to identify the most effective configuration for various workload patterns.

## Key Features

- **Multiple GPU Sharing Modes**: Time Slicing, NVIDIA MPS, MIG
- **Diverse ML Workloads**: Training, Inference, Interactive Tasks
- **Benchmarking Tools**: Latency, Throughput, and Utilization Analysis
- **Container Orchestration**: Docker and Kubernetes support
- **Real-time Monitoring**: GPU resource allocation visualization

## Project Structure

```
.
├── workloads/              # ML workload implementations
│   ├── training/           # Training workloads
│   ├── inference/          # Inference workloads
│   └── interactive/        # Interactive/notebook workloads
├── docker/                 # Docker configurations
├── gpu-configs/            # GPU sharing configurations
├── benchmarking/           # Benchmarking scripts
├── monitoring/             # Monitoring tools
├── kubernetes/             # K8s manifests
└── results/                # Benchmark results and analysis
```

## GPU Sharing Modes

### 1. Time Slicing
- Multiple containers share GPU by alternating access
- Best for: Mixed workloads with varying GPU utilization
- Trade-off: Context switching overhead

### 2. NVIDIA MPS (Multi-Process Service)
- Parallel execution of CUDA kernels from different processes
- Best for: Small, concurrent GPU tasks
- Trade-off: Limited isolation

### 3. MIG (Multi-Instance GPU)
- Hardware-level GPU partitioning (A100/H100)
- Best for: Guaranteed performance isolation
- Trade-off: Requires specific hardware

## Quick Start

### Prerequisites
```bash
- NVIDIA GPU with latest drivers
- Docker with NVIDIA Container Toolkit
- Python 3.8+
- CUDA 11.8+
```

### Installation
```bash
# Clone the repository
git clone https://github.com/IIT-Jodpur/sde-project.git
cd SDE

# Install Python dependencies
pip install -r requirements.txt

# Verify GPU access
python -c "import torch; print(torch.cuda.is_available())"
```

### Running Workloads

#### 1. Training Workload
```bash
cd workloads/training
python resnet_training.py
```

#### 2. Inference Workload
```bash
cd workloads/inference
python inference_server.py
```

#### 3. Benchmarking
```bash
cd benchmarking
python benchmark_suite.py --mode time-slicing --workloads 4
```

## Docker Usage

### Build Containers
```bash
cd docker
./build_all.sh
```

### Run with Time Slicing
```bash
docker run --gpus all nvidia/ml-training:latest
```

### Run with MPS
```bash
./gpu-configs/enable_mps.sh
docker run --gpus all nvidia/ml-training:latest
```

## Benchmarking Results

Results will be saved in `results/` directory with metrics including:
- GPU Utilization (%)
- Latency (ms)
- Throughput (samples/sec)
- Memory Usage (GB)

## Research Questions

1. What is the optimal GPU sharing strategy for mixed ML workloads?
2. How does context switching overhead impact time slicing performance?
3. What are the isolation guarantees for each sharing mode?
4. How do different workload patterns affect resource utilization?

## Contributing

Contributions are welcome! Please see CONTRIBUTING.md for guidelines.

## License

MIT License - see LICENSE file for details.

## References

- [NVIDIA Time Slicing Documentation](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/gpu-sharing.html)
- [NVIDIA MPS Documentation](https://docs.nvidia.com/deploy/mps/index.html)
- [NVIDIA MIG Documentation](https://docs.nvidia.com/datacenter/tesla/mig-user-guide/)
