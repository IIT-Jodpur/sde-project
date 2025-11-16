"""
Comprehensive Benchmark Suite for GPU Sharing Modes
Runs multiple workloads and measures performance metrics
"""

import subprocess
import time
import json
import argparse
import os
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading


class BenchmarkSuite:
    """Orchestrate benchmarking across different GPU sharing modes"""
    
    def __init__(self, mode='time-slicing', num_workloads=4, duration=300):
        self.mode = mode
        self.num_workloads = num_workloads
        self.duration = duration
        self.results = {
            'mode': mode,
            'num_workloads': num_workloads,
            'duration': duration,
            'start_time': datetime.now().isoformat(),
            'workload_results': [],
            'aggregate_metrics': {}
        }
        self.lock = threading.Lock()
    
    def run_workload(self, workload_type, workload_id):
        """Run a single workload and collect metrics"""
        print(f"[Workload {workload_id}] Starting {workload_type}...")
        
        start_time = time.time()
        
        if workload_type == 'training':
            cmd = [
                'python3', 'workloads/training/resnet_training.py',
                '--epochs', '5',
                '--batch-size', '64'
            ]
        elif workload_type == 'inference':
            cmd = [
                'python3', 'workloads/inference/batch_inference.py',
                '--batch-size', '32',
                '--num-batches', '50'
            ]
        elif workload_type == 'interactive':
            cmd = [
                'python3', 'workloads/interactive/jupyter_simulation.py',
                '--num-operations', '10'
            ]
        else:
            raise ValueError(f"Unknown workload type: {workload_type}")
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.duration
            )
            
            end_time = time.time()
            elapsed = end_time - start_time
            
            workload_result = {
                'workload_id': workload_id,
                'workload_type': workload_type,
                'duration': elapsed,
                'success': result.returncode == 0,
                'stdout': result.stdout[-500:] if result.stdout else '',  # Last 500 chars
                'stderr': result.stderr[-500:] if result.stderr else ''
            }
            
            with self.lock:
                self.results['workload_results'].append(workload_result)
            
            status = "✓" if result.returncode == 0 else "✗"
            print(f"[Workload {workload_id}] {status} Completed in {elapsed:.2f}s")
            
            return workload_result
            
        except subprocess.TimeoutExpired:
            print(f"[Workload {workload_id}] ✗ Timeout after {self.duration}s")
            return {
                'workload_id': workload_id,
                'workload_type': workload_type,
                'duration': self.duration,
                'success': False,
                'error': 'Timeout'
            }
        except Exception as e:
            print(f"[Workload {workload_id}] ✗ Error: {str(e)}")
            return {
                'workload_id': workload_id,
                'workload_type': workload_type,
                'success': False,
                'error': str(e)
            }
    
    def run_concurrent_workloads(self, workload_configs):
        """Run multiple workloads concurrently"""
        print(f"\n{'='*60}")
        print(f"Running {len(workload_configs)} concurrent workloads...")
        print(f"Mode: {self.mode}")
        print(f"{'='*60}\n")
        
        with ThreadPoolExecutor(max_workers=len(workload_configs)) as executor:
            futures = {
                executor.submit(self.run_workload, wl_type, wl_id): (wl_type, wl_id)
                for wl_id, wl_type in enumerate(workload_configs)
            }
            
            for future in as_completed(futures):
                wl_type, wl_id = futures[future]
                try:
                    result = future.result()
                except Exception as e:
                    print(f"Workload {wl_id} ({wl_type}) generated exception: {e}")
    
    def calculate_aggregate_metrics(self):
        """Calculate aggregate performance metrics"""
        successful = [r for r in self.results['workload_results'] if r['success']]
        
        if not successful:
            print("Warning: No successful workloads to aggregate")
            return
        
        total_duration = sum(r['duration'] for r in successful)
        avg_duration = total_duration / len(successful)
        
        self.results['aggregate_metrics'] = {
            'total_workloads': len(self.results['workload_results']),
            'successful_workloads': len(successful),
            'failed_workloads': len(self.results['workload_results']) - len(successful),
            'avg_workload_duration': avg_duration,
            'total_gpu_time': total_duration,
            'success_rate': len(successful) / len(self.results['workload_results']) * 100
        }
    
    def save_results(self):
        """Save benchmark results to file"""
        self.results['end_time'] = datetime.now().isoformat()
        
        os.makedirs('results', exist_ok=True)
        filename = f'results/benchmark_{self.mode}_{self.num_workloads}w_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n{'='*60}")
        print("BENCHMARK RESULTS")
        print(f"{'='*60}")
        print(f"Mode: {self.mode}")
        print(f"Total Workloads: {self.results['aggregate_metrics']['total_workloads']}")
        print(f"Successful: {self.results['aggregate_metrics']['successful_workloads']}")
        print(f"Failed: {self.results['aggregate_metrics']['failed_workloads']}")
        print(f"Success Rate: {self.results['aggregate_metrics']['success_rate']:.1f}%")
        print(f"Avg Duration: {self.results['aggregate_metrics']['avg_workload_duration']:.2f}s")
        print(f"{'='*60}")
        print(f"\nResults saved to: {filename}")
    
    def run(self, workload_mix):
        """Run complete benchmark suite"""
        print(f"\n{'='*70}")
        print(f"GPU SHARING BENCHMARK SUITE")
        print(f"{'='*70}")
        print(f"Mode: {self.mode}")
        print(f"Workloads: {self.num_workloads}")
        print(f"Mix: {workload_mix}")
        print(f"Max Duration: {self.duration}s per workload")
        print(f"{'='*70}\n")
        
        # Configure workload distribution
        workload_configs = []
        if workload_mix == 'mixed':
            # Mix of all workload types
            types = ['training', 'inference', 'interactive']
            for i in range(self.num_workloads):
                workload_configs.append(types[i % len(types)])
        elif workload_mix == 'training':
            workload_configs = ['training'] * self.num_workloads
        elif workload_mix == 'inference':
            workload_configs = ['inference'] * self.num_workloads
        elif workload_mix == 'interactive':
            workload_configs = ['interactive'] * self.num_workloads
        else:
            raise ValueError(f"Unknown workload mix: {workload_mix}")
        
        # Run workloads
        self.run_concurrent_workloads(workload_configs)
        
        # Calculate and save results
        self.calculate_aggregate_metrics()
        self.save_results()


def main():
    parser = argparse.ArgumentParser(description='GPU Sharing Benchmark Suite')
    parser.add_argument(
        '--mode',
        type=str,
        default='time-slicing',
        choices=['time-slicing', 'mps', 'mig', 'exclusive'],
        help='GPU sharing mode'
    )
    parser.add_argument(
        '--workloads',
        type=int,
        default=4,
        help='Number of concurrent workloads'
    )
    parser.add_argument(
        '--mix',
        type=str,
        default='mixed',
        choices=['mixed', 'training', 'inference', 'interactive'],
        help='Workload mix'
    )
    parser.add_argument(
        '--duration',
        type=int,
        default=300,
        help='Maximum duration per workload (seconds)'
    )
    
    args = parser.parse_args()
    
    # Run benchmark
    suite = BenchmarkSuite(
        mode=args.mode,
        num_workloads=args.workloads,
        duration=args.duration
    )
    suite.run(workload_mix=args.mix)


if __name__ == '__main__':
    main()

