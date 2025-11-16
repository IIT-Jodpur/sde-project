#!/bin/bash
# Stress test GPU sharing with increasing concurrent workloads

set -e

MODE=${1:-time-slicing}
MAX_WORKLOADS=${2:-8}

echo "======================================================================"
echo "GPU SHARING STRESS TEST"
echo "======================================================================"
echo "Mode: $MODE"
echo "Max concurrent workloads: $MAX_WORKLOADS"
echo ""

mkdir -p results/stress_test

for NUM_WORKLOADS in $(seq 1 $MAX_WORKLOADS); do
    echo ""
    echo "======================================================================"
    echo "Testing with $NUM_WORKLOADS concurrent workload(s)"
    echo "======================================================================"
    
    # Start monitoring
    python3 monitoring/gpu_monitor.py \
      --duration 120 \
      --output results/stress_test/monitor_${MODE}_${NUM_WORKLOADS}w.json &
    MONITOR_PID=$!
    sleep 2
    
    # Run benchmark
    python3 benchmarking/benchmark_suite.py \
      --mode $MODE \
      --workloads $NUM_WORKLOADS \
      --mix mixed \
      --duration 120
    
    # Stop monitoring
    kill $MONITOR_PID 2>/dev/null || true
    
    echo "✓ Completed test with $NUM_WORKLOADS workload(s)"
    
    # Cool down period
    echo "Cooling down for 10 seconds..."
    sleep 10
done

echo ""
echo "======================================================================"
echo "STRESS TEST COMPLETED"
echo "======================================================================"
echo ""
echo "Generating summary report..."

# Create summary
python3 << EOF
import json
import glob
import os

print("\n" + "="*70)
print("STRESS TEST SUMMARY - $MODE")
print("="*70)
print(f"\n{'Workloads':<12} {'Success Rate':<15} {'Avg Duration':<15} {'GPU Util':<15}")
print("-"*70)

for i in range(1, $MAX_WORKLOADS + 1):
    benchmark_files = glob.glob(f'results/benchmark_${MODE}_{i}w_*.json')
    monitor_file = f'results/stress_test/monitor_${MODE}_{i}w.json'
    
    if benchmark_files:
        with open(benchmark_files[-1], 'r') as f:
            data = json.load(f)
            metrics = data.get('aggregate_metrics', {})
            success_rate = metrics.get('success_rate', 0)
            avg_duration = metrics.get('avg_workload_duration', 0)
            
            gpu_util = "N/A"
            if os.path.exists(monitor_file):
                with open(monitor_file, 'r') as mf:
                    monitor_data = json.load(mf)
                    stats = monitor_data.get('statistics', {})
                    if stats and 'devices' in stats and stats['devices']:
                        gpu_util = f"{stats['devices'][0]['gpu_utilization']['mean']:.1f}%"
            
            print(f"{i:<12} {success_rate:<14.1f}% {avg_duration:<14.2f}s {gpu_util:<15}")

print("="*70)
EOF

echo ""
echo "Results saved to: results/stress_test/"
echo ""

