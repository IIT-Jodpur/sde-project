"""
Performance Analysis and Visualization Tool
Analyzes benchmark results and generates comparison charts
"""

import json
import os
import argparse
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


class PerformanceAnalyzer:
    """Analyze and visualize benchmark results"""
    
    def __init__(self):
        self.results = []
        sns.set_style("whitegrid")
        sns.set_palette("husl")
    
    def load_results(self, results_dir='results'):
        """Load all benchmark result files"""
        print(f"Loading results from {results_dir}...")
        
        for filename in os.listdir(results_dir):
            if filename.startswith('benchmark_') and filename.endswith('.json'):
                filepath = os.path.join(results_dir, filename)
                try:
                    with open(filepath, 'r') as f:
                        data = json.load(f)
                        self.results.append(data)
                        print(f"  Loaded: {filename}")
                except Exception as e:
                    print(f"  Error loading {filename}: {e}")
        
        print(f"Loaded {len(self.results)} result files\n")
    
    def compare_modes(self):
        """Compare performance across GPU sharing modes"""
        if not self.results:
            print("No results to analyze")
            return
        
        # Group by mode
        mode_data = {}
        for result in self.results:
            mode = result.get('mode', 'unknown')
            if mode not in mode_data:
                mode_data[mode] = []
            mode_data[mode].append(result)
        
        print("="*60)
        print("GPU SHARING MODE COMPARISON")
        print("="*60)
        
        comparison = []
        for mode, results in mode_data.items():
            metrics = [r.get('aggregate_metrics', {}) for r in results if 'aggregate_metrics' in r]
            
            if not metrics:
                continue
            
            avg_duration = np.mean([m.get('avg_workload_duration', 0) for m in metrics])
            success_rate = np.mean([m.get('success_rate', 0) for m in metrics])
            total_gpu_time = np.mean([m.get('total_gpu_time', 0) for m in metrics])
            
            comparison.append({
                'mode': mode,
                'avg_duration': avg_duration,
                'success_rate': success_rate,
                'total_gpu_time': total_gpu_time,
                'num_runs': len(metrics)
            })
            
            print(f"\n{mode.upper()}:")
            print(f"  Runs: {len(metrics)}")
            print(f"  Avg Workload Duration: {avg_duration:.2f}s")
            print(f"  Success Rate: {success_rate:.1f}%")
            print(f"  Total GPU Time: {total_gpu_time:.2f}s")
        
        print("="*60)
        
        return comparison
    
    def plot_mode_comparison(self, save_path='results/mode_comparison.png'):
        """Generate comparison charts"""
        comparison = self.compare_modes()
        
        if not comparison:
            print("No data to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('GPU Sharing Mode Performance Comparison', fontsize=16, fontweight='bold')
        
        modes = [c['mode'] for c in comparison]
        
        # 1. Average workload duration
        ax = axes[0, 0]
        durations = [c['avg_duration'] for c in comparison]
        bars = ax.bar(modes, durations, color=sns.color_palette("husl", len(modes)))
        ax.set_ylabel('Duration (seconds)')
        ax.set_title('Average Workload Duration (Lower is Better)')
        ax.set_xlabel('GPU Sharing Mode')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}s', ha='center', va='bottom')
        
        # 2. Success rate
        ax = axes[0, 1]
        success_rates = [c['success_rate'] for c in comparison]
        bars = ax.bar(modes, success_rates, color=sns.color_palette("husl", len(modes)))
        ax.set_ylabel('Success Rate (%)')
        ax.set_title('Workload Success Rate (Higher is Better)')
        ax.set_xlabel('GPU Sharing Mode')
        ax.set_ylim(0, 105)
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%', ha='center', va='bottom')
        
        # 3. Total GPU time
        ax = axes[1, 0]
        gpu_times = [c['total_gpu_time'] for c in comparison]
        bars = ax.bar(modes, gpu_times, color=sns.color_palette("husl", len(modes)))
        ax.set_ylabel('Time (seconds)')
        ax.set_title('Total GPU Time')
        ax.set_xlabel('GPU Sharing Mode')
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}s', ha='center', va='bottom')
        
        # 4. Number of runs
        ax = axes[1, 1]
        num_runs = [c['num_runs'] for c in comparison]
        bars = ax.bar(modes, num_runs, color=sns.color_palette("husl", len(modes)))
        ax.set_ylabel('Number of Runs')
        ax.set_title('Benchmark Runs per Mode')
        ax.set_xlabel('GPU Sharing Mode')
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nComparison chart saved to: {save_path}")
        plt.close()
    
    def analyze_gpu_utilization(self, monitor_file):
        """Analyze GPU utilization from monitoring data"""
        print(f"\nAnalyzing GPU utilization from {monitor_file}...")
        
        try:
            with open(monitor_file, 'r') as f:
                data = json.load(f)
        except Exception as e:
            print(f"Error loading monitoring data: {e}")
            return
        
        stats = data.get('statistics', {})
        
        if not stats or 'devices' not in stats:
            print("No statistics available in monitoring data")
            return
        
        print("\n" + "="*60)
        print("GPU UTILIZATION ANALYSIS")
        print("="*60)
        
        for device_stats in stats['devices']:
            device_id = device_stats['device_id']
            gpu_util = device_stats['gpu_utilization']
            mem_util = device_stats['memory_utilization']
            
            print(f"\nGPU {device_id}:")
            print(f"  GPU Utilization:")
            print(f"    Mean: {gpu_util['mean']:.1f}%")
            print(f"    P50:  {gpu_util['p50']:.1f}%")
            print(f"    P95:  {gpu_util['p95']:.1f}%")
            print(f"    Range: {gpu_util['min']:.1f}% - {gpu_util['max']:.1f}%")
            print(f"  Memory Utilization:")
            print(f"    Mean: {mem_util['mean']:.1f}%")
            print(f"    P50:  {mem_util['p50']:.1f}%")
            print(f"    P95:  {mem_util['p95']:.1f}%")
            print(f"    Range: {mem_util['min']:.1f}% - {mem_util['max']:.1f}%")
        
        print("="*60)
        
        # Plot utilization over time
        self.plot_utilization_timeline(data)
    
    def plot_utilization_timeline(self, monitor_data, save_path='results/utilization_timeline.png'):
        """Plot GPU utilization over time"""
        raw_metrics = monitor_data.get('raw_metrics', [])
        
        if not raw_metrics:
            print("No raw metrics to plot")
            return
        
        # Extract time series data
        timestamps = []
        gpu_utils = []
        mem_utils = []
        
        start_time = datetime.fromisoformat(raw_metrics[0]['timestamp'])
        
        for metric in raw_metrics:
            timestamp = datetime.fromisoformat(metric['timestamp'])
            elapsed = (timestamp - start_time).total_seconds()
            timestamps.append(elapsed)
            
            # Average across all devices
            if metric['devices']:
                avg_gpu_util = np.mean([d['gpu_utilization_pct'] for d in metric['devices']])
                avg_mem_util = np.mean([d['memory_utilization_pct'] for d in metric['devices']])
                gpu_utils.append(avg_gpu_util)
                mem_utils.append(avg_mem_util)
        
        # Create plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
        fig.suptitle('GPU Utilization Timeline', fontsize=16, fontweight='bold')
        
        # GPU utilization
        ax1.plot(timestamps, gpu_utils, linewidth=2, color='#2E86AB', label='GPU Utilization')
        ax1.fill_between(timestamps, gpu_utils, alpha=0.3, color='#2E86AB')
        ax1.set_ylabel('GPU Utilization (%)', fontsize=12)
        ax1.set_xlabel('Time (seconds)', fontsize=12)
        ax1.set_ylim(0, 100)
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Memory utilization
        ax2.plot(timestamps, mem_utils, linewidth=2, color='#A23B72', label='Memory Utilization')
        ax2.fill_between(timestamps, mem_utils, alpha=0.3, color='#A23B72')
        ax2.set_ylabel('Memory Utilization (%)', fontsize=12)
        ax2.set_xlabel('Time (seconds)', fontsize=12)
        ax2.set_ylim(0, 100)
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Utilization timeline saved to: {save_path}")
        plt.close()
    
    def generate_report(self, output_file='results/analysis_report.txt'):
        """Generate comprehensive analysis report"""
        print("\nGenerating analysis report...")
        
        with open(output_file, 'w') as f:
            f.write("="*70 + "\n")
            f.write("GPU TIME SLICING PERFORMANCE ANALYSIS REPORT\n")
            f.write("="*70 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total benchmark runs analyzed: {len(self.results)}\n")
            f.write("="*70 + "\n\n")
            
            # Mode comparison
            comparison = self.compare_modes()
            if comparison:
                f.write("GPU SHARING MODE COMPARISON\n")
                f.write("-"*70 + "\n\n")
                
                for c in sorted(comparison, key=lambda x: x['avg_duration']):
                    f.write(f"Mode: {c['mode'].upper()}\n")
                    f.write(f"  Number of runs: {c['num_runs']}\n")
                    f.write(f"  Average workload duration: {c['avg_duration']:.2f}s\n")
                    f.write(f"  Success rate: {c['success_rate']:.1f}%\n")
                    f.write(f"  Total GPU time: {c['total_gpu_time']:.2f}s\n\n")
            
            f.write("="*70 + "\n")
            f.write("RECOMMENDATIONS\n")
            f.write("-"*70 + "\n\n")
            
            if comparison:
                # Find best mode
                best_duration = min(comparison, key=lambda x: x['avg_duration'])
                best_success = max(comparison, key=lambda x: x['success_rate'])
                
                f.write(f"• Best performance (lowest duration): {best_duration['mode']}\n")
                f.write(f"  Average duration: {best_duration['avg_duration']:.2f}s\n\n")
                f.write(f"• Most reliable (highest success rate): {best_success['mode']}\n")
                f.write(f"  Success rate: {best_success['success_rate']:.1f}%\n\n")
            
            f.write("="*70 + "\n")
        
        print(f"Analysis report saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Performance Analysis Tool')
    parser.add_argument(
        '--results-dir',
        type=str,
        default='results',
        help='Directory containing benchmark results'
    )
    parser.add_argument(
        '--monitor-file',
        type=str,
        default=None,
        help='GPU monitoring file to analyze'
    )
    parser.add_argument(
        '--generate-plots',
        action='store_true',
        help='Generate visualization plots'
    )
    
    args = parser.parse_args()
    
    analyzer = PerformanceAnalyzer()
    analyzer.load_results(args.results_dir)
    
    if analyzer.results:
        analyzer.compare_modes()
        
        if args.generate_plots:
            analyzer.plot_mode_comparison()
        
        analyzer.generate_report()
    
    if args.monitor_file and os.path.exists(args.monitor_file):
        analyzer.analyze_gpu_utilization(args.monitor_file)


if __name__ == '__main__':
    main()

