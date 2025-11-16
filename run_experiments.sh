#!/bin/bash
# Run all experiments sequentially and collect results

set -e

echo "======================================================================"
echo "GPU TIME SLICING - AUTOMATED EXPERIMENT SUITE"
echo "======================================================================"
echo ""
echo "This script will run multiple experiments to compare GPU sharing modes"
echo "Duration: ~30-45 minutes depending on hardware"
echo ""
read -p "Press Enter to start or Ctrl+C to cancel..."

# Create results directory
mkdir -p results
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
EXPERIMENT_DIR="results/experiment_${TIMESTAMP}"
mkdir -p $EXPERIMENT_DIR

echo ""
echo "Results will be saved to: $EXPERIMENT_DIR"
echo ""

# Experiment 1: Baseline (Single workload, exclusive mode)
echo "======================================================================"
echo "EXPERIMENT 1: Baseline (Exclusive Mode)"
echo "======================================================================"
python3 monitoring/gpu_monitor.py --duration 180 --output ${EXPERIMENT_DIR}/monitor_baseline.json &
MONITOR_PID=$!
sleep 2

python3 workloads/training/resnet_training.py --epochs 3 --batch-size 128

kill $MONITOR_PID 2>/dev/null || true
echo "✓ Baseline experiment completed"
echo ""
sleep 5

# Experiment 2: Time Slicing (4 concurrent workloads)
echo "======================================================================"
echo "EXPERIMENT 2: Time Slicing (4 Concurrent Workloads)"
echo "======================================================================"
./gpu-configs/setup_time_slicing.sh 4

python3 monitoring/gpu_monitor.py --duration 300 --output ${EXPERIMENT_DIR}/monitor_timeslicing.json &
MONITOR_PID=$!
sleep 2

python3 benchmarking/benchmark_suite.py \
  --mode time-slicing \
  --workloads 4 \
  --mix mixed \
  --duration 300

kill $MONITOR_PID 2>/dev/null || true
echo "✓ Time slicing experiment completed"
echo ""
sleep 5

# Experiment 3: MPS (if user confirms)
echo "======================================================================"
echo "EXPERIMENT 3: NVIDIA MPS"
echo "======================================================================"
echo "MPS requires root privileges. Run this experiment? (y/n)"
read -r response
if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
    echo "Enabling MPS..."
    sudo ./gpu-configs/enable_mps.sh
    
    python3 monitoring/gpu_monitor.py --duration 300 --output ${EXPERIMENT_DIR}/monitor_mps.json &
    MONITOR_PID=$!
    sleep 2
    
    python3 benchmarking/benchmark_suite.py \
      --mode mps \
      --workloads 4 \
      --mix mixed \
      --duration 300
    
    kill $MONITOR_PID 2>/dev/null || true
    
    echo "Disabling MPS..."
    sudo ./gpu-configs/disable_mps.sh
    echo "✓ MPS experiment completed"
else
    echo "⊘ Skipping MPS experiment"
fi
echo ""
sleep 5

# Experiment 4: MIG (if GPU supports it)
echo "======================================================================"
echo "EXPERIMENT 4: MIG (A100/H100 only)"
echo "======================================================================"
if nvidia-smi -i 0 --query-gpu=mig.mode.current --format=csv,noheader &>/dev/null; then
    echo "MIG support detected. Run MIG experiment? (y/n)"
    read -r response
    if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
        echo "Configuring MIG..."
        sudo ./gpu-configs/setup_mig.sh 1g.5gb 4
        
        python3 monitoring/gpu_monitor.py --duration 300 --output ${EXPERIMENT_DIR}/monitor_mig.json &
        MONITOR_PID=$!
        sleep 2
        
        python3 benchmarking/benchmark_suite.py \
          --mode mig \
          --workloads 4 \
          --mix mixed \
          --duration 300
        
        kill $MONITOR_PID 2>/dev/null || true
        
        echo "Disabling MIG..."
        sudo ./gpu-configs/disable_mig.sh
        echo "✓ MIG experiment completed"
    else
        echo "⊘ Skipping MIG experiment"
    fi
else
    echo "⊘ MIG not supported on this GPU"
fi
echo ""

# Analyze results
echo "======================================================================"
echo "ANALYZING RESULTS"
echo "======================================================================"
python3 benchmarking/analyze_results.py \
  --results-dir results \
  --generate-plots

# Move analysis results to experiment directory
mv results/mode_comparison.png ${EXPERIMENT_DIR}/ 2>/dev/null || true
mv results/utilization_timeline.png ${EXPERIMENT_DIR}/ 2>/dev/null || true
mv results/analysis_report.txt ${EXPERIMENT_DIR}/ 2>/dev/null || true

echo ""
echo "======================================================================"
echo "EXPERIMENT SUITE COMPLETED!"
echo "======================================================================"
echo ""
echo "Results saved to: $EXPERIMENT_DIR"
echo ""
echo "Files generated:"
ls -lh ${EXPERIMENT_DIR}/
echo ""
echo "View analysis report:"
echo "  cat ${EXPERIMENT_DIR}/analysis_report.txt"
echo ""
echo "View comparison charts:"
echo "  open ${EXPERIMENT_DIR}/mode_comparison.png"
echo "  open ${EXPERIMENT_DIR}/utilization_timeline.png"
echo ""
echo "======================================================================"

