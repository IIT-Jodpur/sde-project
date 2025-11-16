# Experiments Guide - Detailed Explanation

This document provides a comprehensive explanation of all experiments in the GPU Time Slicing project, including datasets used, objectives, and expected outcomes.

---

## Table of Contents

1. [Training Experiments](#training-experiments)
2. [Inference Experiments](#inference-experiments)
3. [Interactive Experiments](#interactive-experiments)
4. [Benchmarking Experiments](#benchmarking-experiments)
5. [GPU Sharing Mode Experiments](#gpu-sharing-mode-experiments)
6. [Datasets Reference](#datasets-reference)

---

## Training Experiments

### Experiment 1: ResNet50 Training on CIFAR-10

**File:** `workloads/training/resnet_training.py`

#### Objective
Simulate a typical computer vision training workload to measure GPU performance under sustained compute-intensive operations.

#### Dataset: CIFAR-10
- **Source:** torchvision.datasets.CIFAR10 (auto-downloads)
- **Description:** 60,000 color images (32x32 pixels) in 10 classes
- **Split:** 50,000 training images, 10,000 test images
- **Classes:** airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
- **Size:** ~170 MB
- **License:** MIT License (public domain)

#### Model Architecture
- **Network:** ResNet50 (Deep Residual Network)
- **Parameters:** ~25.6 million
- **Pre-trained:** No (trained from scratch)
- **Output Classes:** 10 (modified from original 1000 for ImageNet)

#### Data Preprocessing
```python
# Training transforms
- RandomCrop(32, padding=4)
- RandomHorizontalFlip()
- ToTensor()
- Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])

# Test transforms
- ToTensor()
- Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
```

#### Configurable Parameters
```bash
python3 workloads/training/resnet_training.py \
  --batch-size 128      # Default: 128
  --epochs 5            # Default: 5
  --lr 0.1              # Learning rate, default: 0.1
  --save-model          # Flag to save trained model
```

#### What Gets Measured
- Training loss per epoch
- Training accuracy
- Test loss per epoch
- Test accuracy
- Epoch duration (seconds)
- Samples per second (throughput)
- GPU utilization (%)
- GPU memory usage (MB)

#### Expected Performance (Single GPU, No Sharing)
- **Training Time per Epoch:** ~60-120 seconds (depending on GPU)
- **GPU Utilization:** 85-95%
- **Memory Usage:** ~4-6 GB
- **Final Test Accuracy:** 70-85% (after 5 epochs)

#### Use Case
Represents typical CNN training workloads common in:
- Image classification
- Object detection preprocessing
- Transfer learning base models

---

### Experiment 2: BERT Fine-tuning for Text Classification

**File:** `workloads/training/bert_training.py`

#### Objective
Simulate transformer-based NLP training to test GPU performance with attention mechanisms and sequential processing.

#### Dataset: Synthetic Text Classification
- **Source:** Programmatically generated (no download required)
- **Description:** Synthetic sentiment analysis dataset
- **Size:** Configurable (default: 1,000 samples)
- **Task:** Binary classification (positive/negative sentiment)
- **Sentence Templates:**
  - Positive: "I really enjoyed the {topic} experience."
  - Positive: "The {topic} exceeded all my expectations."
  - Negative: "The {topic} was terrible and disappointing."
  - Negative: "Not satisfied with the {topic} at all."
  - Neutral: "This is a positive sentence about {topic}."
- **Topics:** movie, product, service, book, restaurant
- **Max Sequence Length:** 128 tokens

#### Why Synthetic Data?
1. **Reproducibility:** Same data across all runs
2. **No Download Time:** Immediate execution
3. **Controlled Workload:** Consistent patterns for benchmarking
4. **Privacy:** No real user data required

#### Model Architecture
- **Model:** BERT-base-uncased (Bidirectional Encoder Representations from Transformers)
- **Parameters:** ~110 million
- **Pre-trained:** Yes (from Hugging Face)
- **Fine-tuning Layers:** Classification head (2 classes)
- **Tokenizer:** BERT WordPiece tokenizer

#### Configurable Parameters
```bash
python3 workloads/training/bert_training.py \
  --batch-size 16           # Default: 16 (lower than CNN due to memory)
  --epochs 3                # Default: 3
  --lr 2e-5                 # Learning rate (typical for BERT)
  --num-samples 1000        # Number of training samples
```

#### What Gets Measured
- Training loss per epoch
- Training accuracy
- Epoch duration (seconds)
- Samples per second (throughput)
- GPU utilization (%)
- Memory usage (MB)

#### Expected Performance (Single GPU, No Sharing)
- **Training Time per Epoch:** ~30-60 seconds
- **GPU Utilization:** 70-85%
- **Memory Usage:** ~6-8 GB
- **Final Accuracy:** 95-100% (synthetic data is learnable)

#### Use Case
Represents transformer-based workloads common in:
- Sentiment analysis
- Text classification
- Named entity recognition (NER)
- Question answering fine-tuning

---

## Inference Experiments

### Experiment 3: Batch Inference Benchmark

**File:** `workloads/inference/batch_inference.py`

#### Objective
Measure inference latency and throughput under different batch sizes, simulating production serving scenarios.

#### Dataset: Random Tensor Data
- **Source:** Synthetic (torch.randn)
- **Description:** Randomly generated image tensors
- **Shape:** (batch_size, 3, 224, 224)
- **Purpose:** Pure GPU compute benchmarking without I/O overhead
- **No Real Images:** Focuses on model forward pass performance

#### Why Random Data for Inference?
1. **Consistent Workload:** Eliminates data loading variability
2. **Pure GPU Testing:** Isolates GPU performance from disk I/O
3. **Reproducible:** Same computational cost every run
4. **Fast Setup:** No dataset download required

#### Model Options
```bash
# ResNet50 (default)
python3 workloads/inference/batch_inference.py --model resnet50

# ResNet18 (lighter)
python3 workloads/inference/batch_inference.py --model resnet18

# VGG16 (heavier)
python3 workloads/inference/batch_inference.py --model vgg16
```

#### Configurable Parameters
```bash
python3 workloads/inference/batch_inference.py \
  --model resnet50          # Model architecture
  --batch-size 32           # Default: 32
  --num-batches 100         # Default: 100
```

#### What Gets Measured
- Per-batch latency (ms)
- Throughput (samples/second)
- Mean, Median, Std Dev latency
- P50, P95, P99 percentiles
- Min/Max latency

#### Expected Performance (ResNet50, Single GPU)
- **Batch Size 1:** ~5-10 ms latency, ~100-200 samples/sec
- **Batch Size 32:** ~30-50 ms latency, ~600-1000 samples/sec
- **Batch Size 64:** ~60-100 ms latency, ~800-1200 samples/sec

#### Use Case
Represents production inference scenarios:
- Image classification APIs
- Real-time object detection
- Batch processing pipelines

---

### Experiment 4: Inference Server with HTTP API

**File:** `workloads/inference/inference_server.py`

#### Objective
Simulate a production-ready inference server to test GPU sharing under concurrent request patterns.

#### Dataset: User-Provided Images or Synthetic
- **Source:** HTTP POST requests
- **Description:** JPEG/PNG images uploaded via API
- **Expected Size:** Any size (auto-resized to 224x224)
- **Format:** Standard image files

#### API Endpoints

##### 1. `/health` - Health Check
```bash
curl http://localhost:5000/health
```
Returns: GPU device status

##### 2. `/predict` - Single Image Inference
```bash
curl -X POST http://localhost:5000/predict \
  -F "image=@your_image.jpg"
```
Returns: Top-5 predictions with probabilities and latency

##### 3. `/batch_predict` - Synthetic Batch Test
```bash
curl -X POST http://localhost:5000/batch_predict \
  -H "Content-Type: application/json" \
  -d '{"batch_size": 32}'
```
Returns: Batch latency and throughput

##### 4. `/stats` - Server Statistics
```bash
curl http://localhost:5000/stats
```
Returns: Request count, average latency, P50/P95/P99, QPS

#### Configurable Parameters
```bash
python3 workloads/inference/inference_server.py \
  --model resnet50          # Model to serve
  --port 5000               # Server port
  --host 0.0.0.0            # Server host
```

#### What Gets Measured
- Request latency per prediction (ms)
- Total requests served
- Success rate (%)
- Queries per second (QPS)
- P50, P95, P99 latency percentiles
- Throughput statistics

#### Expected Performance
- **Single Request Latency:** 20-50 ms
- **Throughput:** 20-50 QPS (depends on GPU and batch processing)
- **GPU Utilization:** 30-60% (varies with request rate)

#### Use Case
Represents real-world serving scenarios:
- REST APIs for ML models
- Microservices architecture
- Production model deployment

---

### Experiment 5: Load Testing for Inference Server

**File:** `workloads/inference/load_test.py`

#### Objective
Stress test the inference server with concurrent clients to measure performance under load.

#### Dataset: Synthetic Images
- **Source:** Programmatically generated (PIL + NumPy)
- **Description:** Random RGB images (224x224)
- **Format:** JPEG (in-memory)
- **Purpose:** Simulate concurrent user requests

#### Test Configuration
```bash
python3 workloads/inference/load_test.py \
  --server http://localhost:5000    # Server URL
  --clients 10                       # Concurrent clients (default: 10)
  --duration 60                      # Test duration in seconds (default: 60)
```

#### What Gets Measured
- Total requests sent
- Successful requests
- Failed requests
- Success rate (%)
- Mean latency (ms)
- Median (P50) latency
- P95 and P99 latency
- Queries per second (QPS)
- Requests per minute

#### Expected Performance (10 Clients)
- **Total Requests:** 600-1200 (in 60 seconds)
- **Success Rate:** 95-100%
- **Mean Latency:** 50-100 ms
- **P99 Latency:** 150-300 ms
- **QPS:** 10-20

#### Use Case
Tests scalability of inference serving:
- API performance under load
- Concurrent request handling
- GPU sharing effectiveness

---

## Interactive Experiments

### Experiment 6: Jupyter-Style Interactive Workload

**File:** `workloads/interactive/jupyter_simulation.py`

#### Objective
Simulate typical data science/ML development workflow with bursty GPU usage patterns.

#### Dataset: Synthetic Tensors
- **Source:** torch.randn() for matrix operations
- **Description:** Random matrices and data
- **Purpose:** Simulate exploratory data analysis and model development

#### Simulated Operations

##### 1. Matrix Operations (Data Exploration)
- Size: 1000x1000 to 2000x2000 matrices
- Operations: Matrix multiplication, ReLU activation
- Repeats: 100 iterations
- **Represents:** Data preprocessing, feature engineering

##### 2. Model Training Bursts (Quick Experiments)
- Model: Simple feedforward network (784 -> 256 -> 10)
- Training: 5-10 epochs on synthetic data
- **Represents:** Quick model prototyping, hyperparameter testing

##### 3. Inference Testing
- Model: ResNet18 (pre-trained)
- Samples: 50-100 images
- **Represents:** Model evaluation, prediction testing

##### 4. Data Preprocessing
- Operations: Normalization, interpolation
- Iterations: 30-50 operations
- **Represents:** Image augmentation, data transformation

##### 5. Idle Periods
- Duration: 2-5 seconds between operations
- **Represents:** Thinking time, code writing, result analysis

#### Workflow Pattern
```
┌─────────────────────────────────────────────────┐
│ Matrix Ops → Idle → Data Prep → Idle → Training│
│    ↓                                             │
│ → Idle → Inference → Idle → Matrix Ops → ...   │
└─────────────────────────────────────────────────┘
```

#### Configurable Parameters
```bash
python3 workloads/interactive/jupyter_simulation.py \
  --num-operations 11      # Number of operations (default: 11)
```

#### What Gets Measured
- Operation type and duration
- GPU active time
- Idle time
- Total session time
- GPU utilization percentage
- Operation timestamps

#### Expected Performance
- **Total Session Time:** 60-120 seconds
- **GPU Active Time:** 30-50 seconds (~40-60%)
- **Idle Time:** 20-40 seconds (~30-50%)
- **GPU Utilization:** 40-60% (bursty pattern)

#### Use Case
Represents development workflows:
- Jupyter notebook usage
- Interactive Python sessions
- Exploratory data analysis
- Model prototyping

---

## Benchmarking Experiments

### Experiment 7: Comprehensive Benchmark Suite

**File:** `benchmarking/benchmark_suite.py`

#### Objective
Run multiple workloads concurrently to compare GPU sharing modes.

#### Workload Mix Options

##### 1. Mixed (Default)
```bash
python3 benchmarking/benchmark_suite.py --mix mixed --workloads 4
```
- Training: ResNet50
- Inference: Batch inference
- Interactive: Jupyter simulation
- Training: BERT fine-tuning

##### 2. Training-Only
```bash
python3 benchmarking/benchmark_suite.py --mix training --workloads 4
```
- 4x ResNet50 training workloads

##### 3. Inference-Only
```bash
python3 benchmarking/benchmark_suite.py --mix inference --workloads 4
```
- 4x Batch inference workloads

##### 4. Interactive-Only
```bash
python3 benchmarking/benchmark_suite.py --mix interactive --workloads 4
```
- 4x Interactive simulation workloads

#### GPU Sharing Modes

##### Mode 1: Time Slicing
```bash
./gpu-configs/setup_time_slicing.sh 4
python3 benchmarking/benchmark_suite.py --mode time-slicing --workloads 4
```

##### Mode 2: NVIDIA MPS
```bash
sudo ./gpu-configs/enable_mps.sh
python3 benchmarking/benchmark_suite.py --mode mps --workloads 4
```

##### Mode 3: MIG (A100/H100 only)
```bash
sudo ./gpu-configs/setup_mig.sh 1g.5gb 4
python3 benchmarking/benchmark_suite.py --mode mig --workloads 4
```

#### What Gets Measured
- Per-workload duration
- Success rate (%)
- Average workload duration
- Total GPU time
- Concurrent execution overhead

#### Expected Results

**Time Slicing:**
- Success Rate: 90-100%
- Overhead: 10-30%
- GPU Utilization: 80-95%

**MPS:**
- Success Rate: 95-100%
- Overhead: 5-15%
- GPU Utilization: 85-95%

**MIG:**
- Success Rate: 100%
- Overhead: 0% (isolated)
- GPU Utilization: 60-70% per instance

---

## GPU Sharing Mode Experiments

### Experiment 8: Time Slicing Scalability Test

**Script:** `stress_test.sh`

#### Objective
Test GPU time slicing with increasing numbers of concurrent workloads to find the breaking point.

```bash
./stress_test.sh time-slicing 8
```

#### Test Matrix
- 1 workload (baseline)
- 2 workloads
- 4 workloads
- 6 workloads
- 8 workloads

#### What Gets Measured
- Success rate per workload count
- Average duration per workload
- GPU utilization
- Breaking point identification

---

### Experiment 9: Full Automated Suite

**Script:** `run_experiments.sh`

#### Objective
Run complete comparison of all GPU sharing modes automatically.

```bash
./run_experiments.sh
```

#### Experiment Sequence

**Phase 1: Baseline (5 minutes)**
- Single ResNet50 training
- No GPU sharing
- Establishes performance baseline

**Phase 2: Time Slicing (10 minutes)**
- 4 concurrent mixed workloads
- Monitors GPU utilization
- Collects performance metrics

**Phase 3: MPS (10 minutes)**
- 4 concurrent mixed workloads
- With MPS enabled
- Compares to time slicing

**Phase 4: MIG (10 minutes, if supported)**
- 4 MIG instances
- 1 workload per instance
- Tests isolation

**Phase 5: Analysis**
- Generates comparison charts
- Creates analysis report
- Summarizes findings

---

## Datasets Reference

### Summary Table

| Experiment | Dataset | Source | Size | Type | License |
|-----------|---------|--------|------|------|---------|
| ResNet Training | CIFAR-10 | torchvision | 170 MB | Real images | MIT |
| BERT Training | Synthetic Text | Generated | <1 MB | Synthetic | N/A |
| Batch Inference | Random Tensors | torch.randn | N/A | Synthetic | N/A |
| Inference Server | User Images | HTTP Upload | Variable | Real/Test | N/A |
| Load Testing | Synthetic Images | PIL/NumPy | N/A | Synthetic | N/A |
| Interactive | Random Tensors | torch.randn | N/A | Synthetic | N/A |

### Why This Mix?

#### Real Dataset (CIFAR-10)
- **Pros:** 
  - Realistic workload
  - Standard benchmark
  - Comparable to literature
- **Cons:** 
  - Download time
  - Storage requirement

#### Synthetic Datasets
- **Pros:**
  - Instant setup
  - Reproducible
  - No privacy concerns
  - Controlled workload
- **Cons:**
  - May not reflect real-world complexity

### Adding Real Datasets

Want to use other datasets? Easy to modify:

#### ImageNet
```python
# In resnet_training.py
trainset = torchvision.datasets.ImageNet(
    root='./data', 
    split='train', 
    transform=transform_train
)
```

#### Real Text Dataset (e.g., IMDB)
```python
# In bert_training.py
from datasets import load_dataset
dataset = load_dataset("imdb")
```

#### Custom Dataset
```python
# Create your own dataset class
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data_path):
        # Load your data
        pass
```

---

## Quick Reference: Running All Experiments

```bash
# 1. Training Experiments
python3 workloads/training/resnet_training.py --epochs 5
python3 workloads/training/bert_training.py --epochs 3

# 2. Inference Experiments
python3 workloads/inference/batch_inference.py --batch-size 32
python3 workloads/inference/inference_server.py &
python3 workloads/inference/load_test.py --clients 10

# 3. Interactive Experiment
python3 workloads/interactive/jupyter_simulation.py

# 4. Comprehensive Benchmark
python3 benchmarking/benchmark_suite.py --mode time-slicing --workloads 4

# 5. Automated Full Suite
./run_experiments.sh

# 6. Stress Test
./stress_test.sh time-slicing 8
```

---

## Performance Expectations Summary

| Experiment | Duration | GPU Util | Memory | Dataset Size |
|-----------|----------|----------|--------|--------------|
| ResNet Training (1 epoch) | 60-120s | 85-95% | 4-6 GB | 170 MB |
| BERT Training (1 epoch) | 30-60s | 70-85% | 6-8 GB | <1 MB |
| Batch Inference (100 batches) | 10-30s | 80-95% | 2-4 GB | N/A |
| Inference Server (idle) | Continuous | 5-20% | 2-3 GB | N/A |
| Interactive Session | 60-120s | 40-60% | 2-4 GB | N/A |
| Full Benchmark Suite | 5-10min | 70-90% | 6-8 GB | 170 MB |

---

## Conclusion

This comprehensive experiment suite covers:
- **3 Workload Types:** Training, Inference, Interactive
- **6 ML Models:** ResNet50, ResNet18, VGG16, BERT, Simple NN
- **1 Real Dataset:** CIFAR-10
- **5 Synthetic Datasets:** For reproducible benchmarking
- **3 GPU Sharing Modes:** Time Slicing, MPS, MIG
- **Multiple Scenarios:** Single workload, concurrent, stress testing

All experiments are designed to be:
- **Reproducible:** Same results every run
- **Configurable:** Adjust parameters as needed
- **Measurable:** Comprehensive metrics collection
- **Practical:** Represent real-world use cases

