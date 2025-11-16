# Google Cloud Platform (GCP) Deployment Guide

Complete guide to run GPU Time Slicing experiments on Google Cloud Platform.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [GCP Setup](#gcp-setup)
3. [Create GPU Instance](#create-gpu-instance)
4. [Install Dependencies](#install-dependencies)
5. [Deploy Project](#deploy-project)
6. [Run Experiments](#run-experiments)
7. [Monitor and Results](#monitor-and-results)
8. [Cost Management](#cost-management)
9. [Cleanup](#cleanup)

---

## Prerequisites

### 1. Install Google Cloud SDK

**On Mac:**
```bash
# Install via Homebrew
brew install --cask google-cloud-sdk

# Or download from: https://cloud.google.com/sdk/docs/install
```

**Verify installation:**
```bash
gcloud --version
```

### 2. Create GCP Account

1. Go to https://console.cloud.google.com
2. Create account (Free tier: $300 credit)
3. Enable billing

### 3. Initialize gcloud

```bash
# Login to your Google account
gcloud auth login

# Set your project (create one if needed)
gcloud config set project YOUR_PROJECT_ID

# Set default region (choose closest to you)
gcloud config set compute/region us-central1
gcloud config set compute/zone us-central1-a
```

---

## GCP Setup

### 1. Enable Required APIs

```bash
# Enable Compute Engine API
gcloud services enable compute.googleapis.com

# Enable Container Registry (optional, for Docker)
gcloud services enable containerregistry.googleapis.com
```

### 2. Request GPU Quota

**Important:** GPU quota is 0 by default. You must request it.

```bash
# Check current quota
gcloud compute project-info describe --project=YOUR_PROJECT_ID

# Request quota increase:
# 1. Go to: https://console.cloud.google.com/iam-admin/quotas
# 2. Filter by "GPUs (all regions)"
# 3. Select your region
# 4. Click "EDIT QUOTAS"
# 5. Request increase (suggest: 4-8 GPUs)
# 6. Wait for approval (usually 1-2 business days)
```

**Free Tier Note:** GPU instances are NOT included in free tier, but $300 credit can be used.

---

## Create GPU Instance

### Option 1: Quick Start (T4 GPU - Recommended for Testing)

```bash
# Create instance with NVIDIA T4 GPU
gcloud compute instances create gpu-timeslicing-vm \
  --zone=us-central1-a \
  --machine-type=n1-standard-4 \
  --accelerator=type=nvidia-tesla-t4,count=1 \
  --image-family=ubuntu-2004-lts \
  --image-project=ubuntu-os-cloud \
  --boot-disk-size=100GB \
  --boot-disk-type=pd-balanced \
  --maintenance-policy=TERMINATE \
  --metadata=install-nvidia-driver=True
```

**Specs:**
- GPU: NVIDIA T4 (16GB VRAM)
- CPU: 4 vCPUs
- RAM: 15 GB
- Disk: 100 GB
- Cost: ~$0.50/hour

### Option 2: More Powerful (V100 GPU)

```bash
# Create instance with NVIDIA V100 GPU
gcloud compute instances create gpu-timeslicing-vm \
  --zone=us-central1-a \
  --machine-type=n1-standard-8 \
  --accelerator=type=nvidia-tesla-v100,count=1 \
  --image-family=ubuntu-2004-lts \
  --image-project=ubuntu-os-cloud \
  --boot-disk-size=200GB \
  --boot-disk-type=pd-balanced \
  --maintenance-policy=TERMINATE \
  --metadata=install-nvidia-driver=True
```

**Specs:**
- GPU: NVIDIA V100 (16GB VRAM)
- CPU: 8 vCPUs
- RAM: 30 GB
- Disk: 200 GB
- Cost: ~$2.50/hour

### Option 3: A100 for MIG Testing (Advanced)

```bash
# Create instance with NVIDIA A100 GPU (supports MIG)
gcloud compute instances create gpu-timeslicing-a100 \
  --zone=us-central1-a \
  --machine-type=a2-highgpu-1g \
  --maintenance-policy=TERMINATE \
  --image-family=ubuntu-2004-lts \
  --image-project=ubuntu-os-cloud \
  --boot-disk-size=200GB \
  --metadata=install-nvidia-driver=True
```

**Specs:**
- GPU: NVIDIA A100 (40GB VRAM)
- MIG Support: YES
- Cost: ~$3-4/hour

### Available GPU Types

| GPU Type | VRAM | MIG Support | Cost/hour | Best For |
|----------|------|-------------|-----------|----------|
| T4 | 16GB | No | ~$0.35 | Development, Testing |
| P4 | 8GB | No | ~$0.60 | Light workloads |
| V100 | 16GB | No | ~$2.50 | Training, Benchmarking |
| P100 | 16GB | No | ~$1.50 | Medium workloads |
| A100 | 40GB | Yes | ~$3.50 | MIG testing, Production |

---

## Install Dependencies

### 1. SSH into Instance

```bash
# SSH into your instance
gcloud compute ssh gpu-timeslicing-vm --zone=us-central1-a
```

### 2. Verify GPU Access

```bash
# Check if NVIDIA driver installed
nvidia-smi

# Should show your GPU (T4, V100, or A100)
```

**If nvidia-smi fails:**
```bash
# Install NVIDIA drivers manually
curl https://raw.githubusercontent.com/GoogleCloudPlatform/compute-gpu-installation/main/linux/install_gpu_driver.py --output install_gpu_driver.py
sudo python3 install_gpu_driver.py

# Reboot
sudo reboot

# SSH back in after reboot
gcloud compute ssh gpu-timeslicing-vm --zone=us-central1-a

# Verify
nvidia-smi
```

### 3. Install Docker

```bash
# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Add user to docker group
sudo usermod -aG docker $USER

# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker

# Verify Docker GPU access
sudo docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

### 4. Install Python Dependencies

```bash
# Update system
sudo apt-get update && sudo apt-get upgrade -y

# Install Python 3.10
sudo apt-get install -y python3.10 python3-pip python3.10-venv git

# Create virtual environment
python3 -m venv ~/venv
source ~/venv/bin/activate

# Upgrade pip
pip install --upgrade pip
```

---

## Deploy Project

### 1. Clone Repository

```bash
# Clone your project
cd ~
git clone https://github.com/YOUR_USERNAME/SDE.git
cd SDE

# Or upload from your Mac:
# From your Mac terminal:
# gcloud compute scp --recurse ~/Desktop/Personal/IIT/M.Tech/SDE \
#   gpu-timeslicing-vm:~ --zone=us-central1-a
```

### 2. Install Requirements

```bash
# Activate virtual environment
source ~/venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Verify PyTorch can see GPU
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None"}')"
```

### 3. Build Docker Images

```bash
# Make scripts executable
chmod +x docker/build_all.sh
chmod +x gpu-configs/*.sh
chmod +x *.sh

# Build Docker images
cd docker
./build_all.sh
cd ..

# Verify images
docker images | grep gpu-timeslicing
```

---

## Run Experiments

### 1. Quick Test (Single Workload)

```bash
# Test single training workload
python3 workloads/training/resnet_training.py --epochs 2

# Should complete in 2-5 minutes
# Check results/
ls -lh results/
```

### 2. Run Time Slicing Experiment

```bash
# Setup time slicing for 4 concurrent workloads
./gpu-configs/setup_time_slicing.sh 4

# Run benchmark
python3 benchmarking/benchmark_suite.py \
  --mode time-slicing \
  --workloads 4 \
  --mix mixed \
  --duration 300
```

### 3. Run MPS Experiment

```bash
# Enable MPS (requires sudo)
sudo ./gpu-configs/enable_mps.sh

# Run benchmark
python3 benchmarking/benchmark_suite.py \
  --mode mps \
  --workloads 4 \
  --mix mixed \
  --duration 300

# Disable MPS
sudo ./gpu-configs/disable_mps.sh
```

### 4. Run MIG Experiment (A100 only)

```bash
# Setup MIG (requires A100 GPU and sudo)
sudo ./gpu-configs/setup_mig.sh 1g.5gb 4

# Run benchmark
python3 benchmarking/benchmark_suite.py \
  --mode mig \
  --workloads 4 \
  --mix mixed \
  --duration 300

# Disable MIG
sudo ./gpu-configs/disable_mig.sh
```

### 5. Run Full Automated Suite

```bash
# Run all experiments (30-45 minutes)
./run_experiments.sh

# Results will be in results/experiment_TIMESTAMP/
```

### 6. Run with Docker Compose

```bash
# Run containerized workloads
cd docker
docker-compose up

# View logs
docker-compose logs -f

# Stop containers
docker-compose down
```

---

## Monitor and Results

### 1. Real-time GPU Monitoring

**Terminal 1 (SSH session):**
```bash
# Watch GPU usage in real-time
watch -n 1 nvidia-smi
```

**Terminal 2 (another SSH session):**
```bash
gcloud compute ssh gpu-timeslicing-vm --zone=us-central1-a

cd SDE
source ~/venv/bin/activate

# Run GPU monitor
python3 monitoring/gpu_monitor.py --interval 1 --duration 600
```

### 2. View Results

```bash
# List all results
ls -lh results/

# View latest benchmark
cat results/benchmark_*.json | jq

# View analysis report
cat results/analysis_report.txt

# View monitoring data
cat results/gpu_monitor_*.json | jq
```

### 3. Generate Analysis and Charts

```bash
# Generate comparison charts and analysis
python3 benchmarking/analyze_results.py \
  --results-dir results \
  --generate-plots

# View generated files
ls -lh results/*.png
```

### 4. Download Results to Your Mac

**From your Mac terminal:**
```bash
# Download all results
gcloud compute scp --recurse \
  gpu-timeslicing-vm:~/SDE/results \
  ~/Desktop/GPU-Results \
  --zone=us-central1-a \
  --project=data-science-dev-356009

# View on your Mac
cd ~/Desktop/GPU-Results
open mode_comparison.png
cat analysis_report.txt
```

gsutil -m cp -r ~/sde-project/results gs:// gpu-time-slicing-experiment/gpu-results/
---

## Cost Management

### 1. Check Current Costs

```bash
# Check billing from command line
gcloud billing accounts list

# Or visit: https://console.cloud.google.com/billing
```

### 2. Estimate Costs

**T4 Instance (n1-standard-4 + T4):**
- Compute: $0.19/hour
- GPU: $0.35/hour
- Disk (100GB): $0.04/hour
- **Total: ~$0.58/hour**

**Full experiment suite (3 hours): ~$1.74**

### 3. Stop Instance (to avoid charges)

```bash
# Stop instance (keeps disk, data preserved)
gcloud compute instances stop gpu-timeslicing-vm --zone=us-central1-a

# You only pay for storage (~$0.04/hour)

# Start again when needed
gcloud compute instances start gpu-timeslicing-vm --zone=us-central1-a
```

### 4. Use Preemptible Instances (Save 60-91%)

```bash
# Create preemptible instance (much cheaper but can be terminated)
gcloud compute instances create gpu-timeslicing-preempt \
  --zone=us-central1-a \
  --machine-type=n1-standard-4 \
  --accelerator=type=nvidia-tesla-t4,count=1 \
  --preemptible \
  --image-family=ubuntu-2004-lts \
  --image-project=ubuntu-os-cloud \
  --boot-disk-size=100GB \
  --metadata=install-nvidia-driver=True

# Cost: ~$0.15/hour (vs $0.58/hour)
# Note: Can be terminated by GCP with 30 second notice
```

### 5. Set Budget Alerts

```bash
# Set up budget alerts at: https://console.cloud.google.com/billing/budgets
# Recommended: Set alert at $10, $25, $50
```

---

## Cleanup

### 1. Delete Instance (when completely done)

```bash
# Delete instance (cannot be recovered)
gcloud compute instances delete gpu-timeslicing-vm --zone=us-central1-a

# Confirm deletion
# This stops all charges
```

### 2. Delete Disk (if kept separately)

```bash
# List disks
gcloud compute disks list

# Delete disk
gcloud compute disks delete DISK_NAME --zone=us-central1-a
```

### 3. Clean Up Images (if using Container Registry)

```bash
# List container images
gcloud container images list

# Delete images
gcloud container images delete IMAGE_NAME
```

---

## Troubleshooting

### Issue 1: GPU Quota Error

**Error:** "Quota 'NVIDIA_T4_GPUS' exceeded"

**Solution:**
1. Request quota increase (see GCP Setup section)
2. Try different region (some have more availability)
3. Use preemptible instance (higher availability)

### Issue 2: nvidia-smi Not Found

**Solution:**
```bash
# Install drivers manually
curl https://raw.githubusercontent.com/GoogleCloudPlatform/compute-gpu-installation/main/linux/install_gpu_driver.py --output install_gpu_driver.py
sudo python3 install_gpu_driver.py
sudo reboot
```

### Issue 3: Docker GPU Access Denied

**Solution:**
```bash
# Reinstall NVIDIA Container Toolkit
sudo apt-get purge nvidia-docker2
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker

# Log out and back in
exit
gcloud compute ssh gpu-timeslicing-vm --zone=us-central1-a
```

### Issue 4: Out of Memory

**Solution:**
- Reduce batch size in workloads
- Use smaller model
- Or upgrade to larger GPU instance

### Issue 5: SSH Connection Timeout

**Solution:**
```bash
# Allow SSH in firewall
gcloud compute firewall-rules create allow-ssh \
  --allow tcp:22 \
  --source-ranges 0.0.0.0/0

# Or use browser-based SSH from GCP console
```

---

## Advanced: Multi-GPU Setup

### Create Instance with Multiple GPUs

```bash
# Create instance with 4 T4 GPUs
gcloud compute instances create gpu-timeslicing-multi \
  --zone=us-central1-a \
  --machine-type=n1-standard-16 \
  --accelerator=type=nvidia-tesla-t4,count=4 \
  --image-family=ubuntu-2004-lts \
  --image-project=ubuntu-os-cloud \
  --boot-disk-size=200GB \
  --maintenance-policy=TERMINATE \
  --metadata=install-nvidia-driver=True

# Cost: ~$2.20/hour
```

Test each GPU independently:
```bash
# Test GPU 0
CUDA_VISIBLE_DEVICES=0 python3 workloads/training/resnet_training.py --epochs 2

# Test GPU 1
CUDA_VISIBLE_DEVICES=1 python3 workloads/training/resnet_training.py --epochs 2
```

---

## Quick Reference Commands

```bash
# Create instance
gcloud compute instances create gpu-timeslicing-vm \
  --zone=us-central1-a --machine-type=n1-standard-4 \
  --accelerator=type=nvidia-tesla-t4,count=1 \
  --image-family=ubuntu-2004-lts --image-project=ubuntu-os-cloud \
  --boot-disk-size=100GB --maintenance-policy=TERMINATE \
  --metadata=install-nvidia-driver=True

# SSH
gcloud compute ssh gpu-timeslicing-vm --zone=us-central1-a

# Copy files TO instance
gcloud compute scp --recurse LOCAL_PATH gpu-timeslicing-vm:~ --zone=us-central1-a

# Copy files FROM instance  
gcloud compute scp --recurse gpu-timeslicing-vm:~/SDE/results ~/Desktop/ --zone=us-central1-a

# Stop instance
gcloud compute instances stop gpu-timeslicing-vm --zone=us-central1-a

# Start instance
gcloud compute instances start gpu-timeslicing-vm --zone=us-central1-a

# Delete instance
gcloud compute instances delete gpu-timeslicing-vm --zone=us-central1-a

# Check status
gcloud compute instances list
```

---

## Summary

**Setup Steps:**
1. Install gcloud CLI
2. Create GCP project and enable billing
3. Request GPU quota
4. Create GPU instance
5. Install dependencies
6. Deploy project
7. Run experiments
8. Download results
9. Stop/delete instance

**Estimated Total Cost:**
- Setup: 30 minutes (~$0.30)
- Testing: 1 hour (~$0.60)
- Full experiments: 3 hours (~$1.80)
- **Total: ~$2.70**

**Pro Tips:**
- Use T4 for development/testing (cheapest)
- Use preemptible instances to save 60-80%
- Stop instance when not in use
- Set budget alerts
- Download results regularly
- Delete instance when completely done

Ready to deploy! Start with the Quick Start command and you'll be running experiments in 15 minutes.

