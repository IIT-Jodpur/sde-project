# GPU Time Slicing for Containerized ML Workloads

Research project exploring GPU sharing strategies (Time Slicing, MPS, MIG) for running multiple containerized ML workloads on a single GPU.

## Quick Setup

### Prerequisites
- NVIDIA GPU with drivers (460.x+)
- Docker with NVIDIA Container Toolkit
- Python 3.8+
- CUDA 11.8+

### Local Setup

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/SDE.git
cd SDE

# Install Python dependencies
pip install -r requirements.txt

# Verify GPU access
python -c "import torch; print(torch.cuda.is_available())"

# Build Docker images
cd docker && ./build_all.sh && cd ..
```

## Running Experiments

### 1. Single Workload (Baseline)
```bash
python3 workloads/training/resnet_training.py --epochs 5
```

### 2. Time Slicing (4 Concurrent Workloads)
```bash
./gpu-configs/setup_time_slicing.sh 4
python3 benchmarking/benchmark_suite.py --mode time-slicing --workloads 4
```

### 3. NVIDIA MPS
```bash
sudo ./gpu-configs/enable_mps.sh
python3 benchmarking/benchmark_suite.py --mode mps --workloads 4
sudo ./gpu-configs/disable_mps.sh
```

### 4. Full Automated Suite
```bash
./run_experiments.sh
```

### View Results
```bash
ls results/
python3 benchmarking/analyze_results.py --results-dir results --generate-plots
cat results/analysis_report.txt
```

## GCP Deployment

### Setup
```bash
# Install gcloud CLI
brew install --cask google-cloud-sdk  # Mac

# Login and configure
gcloud auth login
gcloud config set project YOUR_PROJECT_ID

# Request GPU quota (wait for approval)
# Visit: https://console.cloud.google.com/iam-admin/quotas
```

### Create GPU Instance
```bash
# Create T4 GPU instance (~$0.58/hour)
gcloud compute instances create gpu-vm \
  --zone=us-central1-a \
  --machine-type=n1-standard-4 \
  --accelerator=type=nvidia-tesla-t4,count=1 \
  --image-family=ubuntu-2004-lts \
  --image-project=ubuntu-os-cloud \
  --boot-disk-size=100GB \
  --maintenance-policy=TERMINATE \
  --metadata=install-nvidia-driver=True

# SSH into instance
gcloud compute ssh gpu-vm --zone=us-central1-a
```

### Install on GCP Instance
```bash
# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker

# Install Python dependencies
sudo apt-get install -y python3-pip git
pip3 install -r requirements.txt

# Clone project
git clone https://github.com/YOUR_USERNAME/SDE.git
cd SDE

# Run experiments
./run_experiments.sh
```

### Download Results
```bash
# From your local machine
gcloud compute scp --recurse gpu-vm:~/SDE/results ~/Desktop/ --zone=us-central1-a
```

### Cleanup
```bash
# Stop instance (keeps data, minimal cost)
gcloud compute instances stop gpu-vm --zone=us-central1-a

# Delete instance (stops all charges)
gcloud compute instances delete gpu-vm --zone=us-central1-a
```

## Project Structure

```
.
├── workloads/          # ML workload implementations
│   ├── training/       # ResNet, BERT training
│   ├── inference/      # Inference server, batch inference
│   └── interactive/    # Jupyter simulation
├── docker/             # Docker images and compose
├── gpu-configs/        # GPU sharing configuration scripts
├── benchmarking/       # Benchmark suite and analysis
├── kubernetes/         # K8s deployment manifests
└── results/            # Benchmark results (generated)
```

## GPU Sharing Modes

| Mode | Works On | Isolation | Overhead | Setup |
|------|----------|-----------|----------|-------|
| **Time Slicing** | All GPUs | Process | Medium | Easy |
| **MPS** | All GPUs | Process | Low | Medium |
| **MIG** | A100/H100 | Hardware | None | Complex |

## Documentation

- **QUICKSTART.md** - Detailed setup guide with troubleshooting
- **EXPERIMENTS_GUIDE.md** - Complete explanation of all experiments and datasets
- **PROJECT_STRUCTURE.md** - Detailed project architecture and components
- **GCP_DEPLOYMENT.md** - Complete GCP deployment guide with cost estimates
- **gpu-configs/README.md** - GPU configuration details
- **kubernetes/README.md** - Kubernetes deployment guide
- **RESULTS_TEMPLATE.md** - Template for documenting experimental results

## Quick Commands

```bash
# Training workloads
python3 workloads/training/resnet_training.py
python3 workloads/training/bert_training.py

# Inference workloads
python3 workloads/inference/batch_inference.py
python3 workloads/inference/inference_server.py

# Interactive workload
python3 workloads/interactive/jupyter_simulation.py

# Monitor GPU
python3 monitoring/gpu_monitor.py --duration 300
watch -n 1 nvidia-smi

# Docker Compose
cd docker && docker-compose up
```

## Troubleshooting

**GPU not detected:**
```bash
nvidia-smi  # Verify GPU works
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

**Out of memory:**
- Reduce batch size in workload scripts
- Reduce number of concurrent workloads

**MPS issues:**
```bash
sudo ./gpu-configs/disable_mps.sh
sudo ./gpu-configs/enable_mps.sh
```

## Research Questions

1. What is the optimal GPU sharing strategy for mixed ML workloads?
2. How does context switching overhead impact time slicing performance?
3. What are the isolation guarantees for each sharing mode?
4. How do different workload patterns affect resource utilization?

## License

MIT License

## References

- [NVIDIA Time Slicing](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/gpu-sharing.html)
- [NVIDIA MPS](https://docs.nvidia.com/deploy/mps/index.html)
- [NVIDIA MIG](https://docs.nvidia.com/datacenter/tesla/mig-user-guide/)
