# Project Structure Overview

## Directory Layout

```
SDE/
├── README.md                          # Main project documentation
├── QUICKSTART.md                      # Quick start guide
├── CONTRIBUTING.md                    # Contribution guidelines
├── LICENSE                            # MIT License
├── RESULTS_TEMPLATE.md               # Template for documenting results
├── .gitignore                        # Git ignore patterns
├── requirements.txt                  # Python dependencies
├── run_experiments.sh                # Automated experiment suite
├── stress_test.sh                    # Stress testing script
│
├── workloads/                        # ML Workload Implementations
│   ├── training/                     # Training workloads
│   │   ├── resnet_training.py        # ResNet50 on CIFAR-10
│   │   └── bert_training.py          # BERT fine-tuning
│   ├── inference/                    # Inference workloads
│   │   ├── inference_server.py       # Flask-based inference API
│   │   ├── batch_inference.py        # Batch inference benchmark
│   │   └── load_test.py              # Load testing tool
│   └── interactive/                  # Interactive workloads
│       └── jupyter_simulation.py     # Jupyter-style simulation
│
├── docker/                           # Docker Configurations
│   ├── Dockerfile.base               # Base image with ML libraries
│   ├── Dockerfile.training           # Training workload container
│   ├── Dockerfile.inference          # Inference workload container
│   ├── Dockerfile.interactive        # Interactive workload container
│   ├── docker-compose.yml            # Multi-container orchestration
│   └── build_all.sh                  # Build all images script
│
├── gpu-configs/                      # GPU Sharing Configurations
│   ├── README.md                     # Configuration guide
│   ├── setup_time_slicing.sh         # Configure time slicing
│   ├── enable_mps.sh                 # Enable NVIDIA MPS
│   ├── disable_mps.sh                # Disable MPS
│   ├── setup_mig.sh                  # Configure MIG
│   └── disable_mig.sh                # Disable MIG
│
├── benchmarking/                     # Benchmarking Tools
│   ├── benchmark_suite.py            # Comprehensive benchmark suite
│   └── analyze_results.py            # Performance analysis & visualization
│
├── monitoring/                       # Monitoring Tools
│   └── gpu_monitor.py                # Real-time GPU monitoring
│
├── kubernetes/                       # Kubernetes Manifests
│   ├── README.md                     # K8s deployment guide
│   ├── gpu-workloads.yaml            # Basic pod definitions
│   ├── inference-deployment.yaml     # Inference service deployment
│   └── training-job.yaml             # Training job definition
│
└── results/                          # Results Directory (created at runtime)
    ├── benchmark_*.json              # Benchmark results
    ├── gpu_monitor_*.json            # GPU monitoring data
    ├── mode_comparison.png           # Performance comparison chart
    ├── utilization_timeline.png      # Utilization timeline chart
    └── analysis_report.txt           # Analysis report
```

## Component Descriptions

### Workloads

#### Training Workloads
- **resnet_training.py**: CNN training on CIFAR-10 dataset
  - Configurable epochs, batch size, learning rate
  - Tracks GPU metrics during training
  - Saves models and training statistics

- **bert_training.py**: Transformer fine-tuning
  - Synthetic text classification task
  - Demonstrates NLP workload patterns
  - Smaller memory footprint than ResNet

#### Inference Workloads
- **inference_server.py**: HTTP API for model inference
  - Flask-based REST API
  - Health checks and statistics endpoints
  - Batch inference support

- **batch_inference.py**: Batch inference benchmarking
  - Measures latency and throughput
  - Configurable batch sizes
  - Statistical analysis of results

- **load_test.py**: Concurrent load testing
  - Simulates multiple clients
  - Measures server performance under load
  - Generates latency percentiles

#### Interactive Workloads
- **jupyter_simulation.py**: Jupyter-style workflow
  - Bursty GPU usage pattern
  - Simulates typical data science workflow
  - Tracks idle vs. active time

### Docker Images

All images based on NVIDIA CUDA 11.8 with cuDNN:

1. **base**: Common ML libraries (PyTorch, TensorFlow, Transformers)
2. **training**: Optimized for training workloads
3. **inference**: Optimized for serving (Flask, minimal dependencies)
4. **interactive**: Includes Jupyter and visualization tools

### GPU Sharing Modes

#### Time Slicing
- Software-based GPU sharing
- Multiple containers time-multiplex GPU access
- Works on all NVIDIA GPUs
- Configurable number of replicas (concurrent containers)

#### NVIDIA MPS (Multi-Process Service)
- Parallel CUDA kernel execution
- Lower overhead than time slicing
- Requires root privileges
- Configurable thread percentage and memory limits

#### MIG (Multi-Instance GPU)
- Hardware-level partitioning (A100, A30, H100)
- Dedicated compute and memory per instance
- Multiple profiles (1g.5gb, 2g.10gb, 3g.20gb, 7g.40gb)
- Requires system reboot for initial setup

### Benchmarking Tools

#### benchmark_suite.py
- Orchestrates concurrent workload execution
- Supports all GPU sharing modes
- Configurable workload mix (training/inference/interactive)
- Collects execution metrics and success rates

#### analyze_results.py
- Aggregates benchmark results
- Generates comparison charts
- Calculates performance statistics
- Produces analysis reports

### Monitoring Tools

#### gpu_monitor.py
- Real-time GPU metrics collection
- Tracks utilization, memory, temperature, power
- Configurable sampling interval
- Saves time-series data
- Generates utilization timelines

### Kubernetes Support

#### Deployment Options
- **Pods**: Individual workload pods
- **Deployments**: Scalable inference services
- **Jobs**: Batch training jobs
- **ConfigMaps**: Time slicing configuration

#### Features
- Resource quotas and limits
- Health checks (readiness/liveness)
- Persistent volume claims
- LoadBalancer services

### Automation Scripts

#### run_experiments.sh
- Automated experiment suite
- Runs all GPU sharing modes sequentially
- Collects comprehensive results
- Generates final analysis report

#### stress_test.sh
- Tests with increasing concurrent workloads
- Identifies breaking points
- Measures scalability
- Creates summary report

## Usage Workflows

### Basic Experiment Workflow

1. **Setup**: Install dependencies and build containers
2. **Baseline**: Run single workload for baseline metrics
3. **Configure**: Setup GPU sharing mode (time-slicing/MPS/MIG)
4. **Benchmark**: Run concurrent workloads with monitoring
5. **Analyze**: Generate performance comparison and charts
6. **Document**: Fill in results template with findings

### Development Workflow

1. **Modify Workloads**: Customize ML models and parameters
2. **Test Locally**: Run workloads natively before containerizing
3. **Build Containers**: Rebuild Docker images with changes
4. **Benchmark**: Compare performance across modes
5. **Iterate**: Optimize based on results

### Production Deployment

1. **Build Images**: Create production Docker images
2. **Push to Registry**: Upload to container registry
3. **Deploy to K8s**: Apply Kubernetes manifests
4. **Configure GPU Sharing**: Setup cluster-wide GPU policy
5. **Monitor**: Deploy monitoring dashboards
6. **Scale**: Adjust replicas based on load

## Key Metrics

### Performance Metrics
- Workload duration (seconds)
- Throughput (samples/second)
- Latency (milliseconds) - P50, P95, P99
- Success rate (percentage)

### GPU Metrics
- GPU utilization (percentage)
- Memory utilization (percentage)
- Temperature (Celsius)
- Power consumption (Watts)
- Number of processes

### Efficiency Metrics
- GPU time / Wall time ratio
- Cost per inference
- Workloads per GPU
- Resource utilization efficiency

## Research Questions Addressed

1. **Performance Trade-offs**: How do different sharing modes impact workload performance?
2. **Scalability**: How many concurrent workloads can each mode support?
3. **Isolation**: What are the isolation guarantees of each mode?
4. **Overhead**: What is the context switching overhead in time slicing?
5. **Workload Patterns**: Which mode is best for specific workload types?
6. **Cost Efficiency**: What is the cost-benefit analysis of each mode?

## Expected Outcomes

### Time Slicing
- High GPU utilization (>80%)
- Moderate performance overhead (10-30%)
- Good for mixed workloads
- Variable latency

### MPS
- Higher GPU utilization (>85%)
- Lower overhead than time slicing (<10%)
- Best for many small kernels
- Requires careful memory management

### MIG
- Guaranteed performance
- Predictable latency
- Lower utilization per instance (60-70%)
- Best for strict SLAs

## Future Extensions

Potential areas for expansion:

1. **Advanced Workloads**: Add LLM inference, diffusion models
2. **CUDA Streams**: Implement stream priority experiments
3. **Power Analysis**: Add power consumption monitoring
4. **Network I/O**: Test distributed training scenarios
5. **Multi-GPU**: Extend to multi-GPU systems
6. **AutoML**: Integrate hyperparameter tuning
7. **Observability**: Add Prometheus/Grafana dashboards
8. **Cost Analysis**: Add cloud cost calculations

## References

- [NVIDIA GPU Operator Documentation](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/)
- [NVIDIA MPS Documentation](https://docs.nvidia.com/deploy/mps/)
- [NVIDIA MIG User Guide](https://docs.nvidia.com/datacenter/tesla/mig-user-guide/)
- [Kubernetes Device Plugins](https://kubernetes.io/docs/concepts/extend-kubernetes/compute-storage-net/device-plugins/)

## Troubleshooting Guide

See individual README files for detailed troubleshooting:
- Docker issues: `docker/README.md`
- GPU configuration: `gpu-configs/README.md`
- Kubernetes: `kubernetes/README.md`
- General setup: `QUICKSTART.md`

---

**Note**: This project is designed for research and educational purposes. For production deployments, additional considerations around security, monitoring, and reliability are recommended.

