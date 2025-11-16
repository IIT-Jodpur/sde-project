# GPU Time Slicing Project - Setup Complete! 🚀

## Project Summary

A comprehensive research project for exploring GPU time slicing and different GPU sharing modes (Time Slicing, NVIDIA MPS, and MIG) for containerized machine learning workloads.

## ✅ What's Been Created

### 📁 Core Components

#### 1. ML Workloads (6 implementations)
- ✅ **Training**: ResNet50 on CIFAR-10, BERT fine-tuning
- ✅ **Inference**: HTTP API server, batch inference, load testing
- ✅ **Interactive**: Jupyter-style workflow simulation

#### 2. Docker Infrastructure (5 images)
- ✅ Base image with ML libraries
- ✅ Training-optimized container
- ✅ Inference-optimized container
- ✅ Interactive development container
- ✅ Docker Compose orchestration

#### 3. GPU Sharing Configurations (6 scripts)
- ✅ Time Slicing setup
- ✅ NVIDIA MPS enable/disable
- ✅ MIG configuration (A100/H100)
- ✅ Comprehensive configuration guide

#### 4. Benchmarking & Monitoring (3 tools)
- ✅ Benchmark suite with multi-mode support
- ✅ Real-time GPU monitoring
- ✅ Performance analysis with visualization

#### 5. Kubernetes Support (4 manifests)
- ✅ Pod definitions for all workloads
- ✅ Inference service deployment
- ✅ Training job specification
- ✅ Complete K8s deployment guide

#### 6. Automation Scripts (2 tools)
- ✅ Automated experiment suite
- ✅ Stress testing framework

#### 7. Documentation (7 files)
- ✅ Comprehensive README
- ✅ Quick start guide
- ✅ Project structure overview
- ✅ Contributing guidelines
- ✅ Results template
- ✅ MIT License
- ✅ .gitignore

## 🎯 Quick Start

### Step 1: Install Dependencies
```bash
cd /Users/subramanianmariappan/Desktop/Personal/IIT/M.Tech/SDE
pip install -r requirements.txt
```

### Step 2: Build Docker Images
```bash
cd docker
./build_all.sh
```

### Step 3: Run Your First Experiment
```bash
# Option A: Run single workload
python3 workloads/training/resnet_training.py --epochs 5

# Option B: Run automated experiment suite
./run_experiments.sh

# Option C: Test inference server
docker run -p 5000:5000 --gpus all gpu-timeslicing/inference:latest
```

## 📊 Research Capabilities

### GPU Sharing Modes
1. **Time Slicing**: Software-based GPU multiplexing
2. **NVIDIA MPS**: Parallel CUDA kernel execution
3. **MIG**: Hardware-level GPU partitioning

### Workload Types
1. **Training**: Deep learning model training
2. **Inference**: Model serving and prediction
3. **Interactive**: Jupyter-style development

### Metrics Tracked
- GPU Utilization (%)
- Memory Usage (GB)
- Latency (ms) - P50, P95, P99
- Throughput (samples/sec)
- Temperature (°C)
- Power Consumption (W)

## 🔬 Example Experiments

### Experiment 1: Baseline Performance
```bash
python3 workloads/training/resnet_training.py --epochs 5 --batch-size 128
```

### Experiment 2: Time Slicing (4 concurrent workloads)
```bash
./gpu-configs/setup_time_slicing.sh 4
docker-compose -f docker/docker-compose.yml up
```

### Experiment 3: Benchmarking
```bash
python3 benchmarking/benchmark_suite.py --mode time-slicing --workloads 4 --mix mixed
```

### Experiment 4: Analysis
```bash
python3 benchmarking/analyze_results.py --results-dir results --generate-plots
```

## 📈 Expected Results

### Performance Comparison
- **Time Slicing**: 10-30% overhead, >80% GPU utilization
- **MPS**: <10% overhead, >85% GPU utilization
- **MIG**: Predictable performance, 60-70% per instance

### Use Case Recommendations
- **Development**: Time Slicing (flexible, easy setup)
- **Production Training**: MIG (isolation, predictability)
- **Production Inference**: MPS (low overhead, high throughput)
- **Cost Optimization**: Time Slicing (maximum GPU sharing)

## 🛠️ Technology Stack

- **Languages**: Python 3.10+, Bash
- **ML Frameworks**: PyTorch 2.0, TensorFlow 2.13, Transformers
- **Containerization**: Docker, Docker Compose
- **Orchestration**: Kubernetes
- **GPU**: NVIDIA CUDA 11.8, cuDNN 8
- **Monitoring**: pynvml, gpustat
- **Visualization**: Matplotlib, Seaborn

## 📚 Documentation

- **README.md**: Main project documentation
- **QUICKSTART.md**: Step-by-step getting started guide
- **PROJECT_STRUCTURE.md**: Detailed component overview
- **CONTRIBUTING.md**: Guidelines for contributors
- **RESULTS_TEMPLATE.md**: Template for documenting experiments

### Subdirectory READMEs
- **gpu-configs/README.md**: GPU sharing configuration guide
- **kubernetes/README.md**: Kubernetes deployment guide

## 🎓 Research Questions

This project helps answer:

1. What is the optimal GPU sharing strategy for mixed ML workloads?
2. How does context switching overhead impact time slicing performance?
3. What are the isolation guarantees for each sharing mode?
4. How do different workload patterns affect resource utilization?
5. What is the cost-benefit trade-off of each GPU sharing mode?

## 🔍 Project Statistics

- **Total Files**: 35+
- **Python Scripts**: 10
- **Shell Scripts**: 7
- **Docker Images**: 4
- **Kubernetes Manifests**: 4
- **Documentation Files**: 7
- **Lines of Code**: ~3,500+

## 🚀 Next Steps

1. **Test on your GPU**:
   ```bash
   nvidia-smi  # Verify GPU access
   pip install -r requirements.txt
   python3 workloads/training/resnet_training.py --epochs 1
   ```

2. **Explore different modes**:
   - Start with Time Slicing (works on all GPUs)
   - Try MPS for comparison
   - If you have A100/H100, test MIG

3. **Run benchmarks**:
   ```bash
   ./run_experiments.sh
   ```

4. **Analyze results**:
   ```bash
   cat results/analysis_report.txt
   open results/mode_comparison.png
   ```

5. **Customize for your needs**:
   - Modify workloads in `workloads/`
   - Adjust batch sizes and model parameters
   - Add your own ML models

## 📝 Notes

- All scripts are executable (chmod +x already applied)
- Results directory will be created automatically
- Docker images need to be built before use
- GPU drivers and NVIDIA Container Toolkit required

## 🤝 Contributing

See `CONTRIBUTING.md` for guidelines on:
- Adding new workloads
- Implementing new GPU sharing modes
- Contributing benchmarks
- Improving documentation

## 📄 License

MIT License - See `LICENSE` file for details

## 🎉 You're Ready!

The project is fully set up and ready for experimentation. Start with the QUICKSTART.md guide or dive straight into running experiments.

For questions or issues:
1. Check documentation in each subdirectory
2. Review QUICKSTART.md for common solutions
3. Examine PROJECT_STRUCTURE.md for component details

**Happy GPU Sharing Research! 🎓🔬**

---

**Pro Tip**: Start with a single workload to verify GPU access, then gradually increase complexity to concurrent workloads and benchmarking.

