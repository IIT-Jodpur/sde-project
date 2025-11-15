"""
Real-time GPU Monitoring Tool
Tracks GPU metrics during workload execution
"""

import time
import json
import argparse
import os
from datetime import datetime
from collections import deque
import threading


try:
    import pynvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False
    print("Warning: pynvml not available. Install with: pip install pynvml")


class GPUMonitor:
    """Monitor GPU utilization and performance metrics"""
    
    def __init__(self, sample_interval=1.0, duration=None):
        self.sample_interval = sample_interval
        self.duration = duration
        self.running = False
        self.metrics_history = []
        
        if not NVML_AVAILABLE:
            raise RuntimeError("pynvml is required for GPU monitoring")
        
        try:
            pynvml.nvmlInit()
            self.device_count = pynvml.nvmlDeviceGetCount()
            self.handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(self.device_count)]
            print(f"Initialized GPU monitoring for {self.device_count} device(s)")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize NVML: {e}")
    
    def get_gpu_metrics(self):
        """Collect current GPU metrics"""
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'devices': []
        }
        
        for i, handle in enumerate(self.handles):
            try:
                # Memory info
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                
                # Utilization rates
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                
                # Temperature
                temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                
                # Power usage
                power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert mW to W
                
                # Clock speeds
                sm_clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM)
                mem_clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM)
                
                # Process info
                try:
                    processes = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
                    num_processes = len(processes)
                    process_mem = sum(p.usedGpuMemory for p in processes) / (1024**2)  # MB
                except:
                    num_processes = 0
                    process_mem = 0
                
                device_metrics = {
                    'device_id': i,
                    'memory_used_mb': mem_info.used / (1024**2),
                    'memory_total_mb': mem_info.total / (1024**2),
                    'memory_utilization_pct': (mem_info.used / mem_info.total) * 100,
                    'gpu_utilization_pct': util.gpu,
                    'memory_bandwidth_utilization_pct': util.memory,
                    'temperature_c': temp,
                    'power_watts': power,
                    'sm_clock_mhz': sm_clock,
                    'memory_clock_mhz': mem_clock,
                    'num_processes': num_processes,
                    'process_memory_mb': process_mem
                }
                
                metrics['devices'].append(device_metrics)
                
            except Exception as e:
                print(f"Error collecting metrics for GPU {i}: {e}")
        
        return metrics
    
    def monitor_loop(self):
        """Main monitoring loop"""
        start_time = time.time()
        
        print(f"\nStarting GPU monitoring (interval: {self.sample_interval}s)...")
        print(f"{'='*80}")
        print(f"{'Time':<12} {'GPU':<6} {'GPU%':<8} {'Mem%':<8} {'Temp':<8} {'Power':<10} {'Procs':<8}")
        print(f"{'='*80}")
        
        sample_count = 0
        
        while self.running:
            try:
                metrics = self.get_gpu_metrics()
                self.metrics_history.append(metrics)
                
                # Print current status
                for device in metrics['devices']:
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    print(f"{timestamp:<12} "
                          f"GPU{device['device_id']:<4} "
                          f"{device['gpu_utilization_pct']:>5.1f}%  "
                          f"{device['memory_utilization_pct']:>5.1f}%  "
                          f"{device['temperature_c']:>5.0f}°C  "
                          f"{device['power_watts']:>7.1f}W   "
                          f"{device['num_processes']:>5}")
                
                sample_count += 1
                
                # Check duration limit
                if self.duration and (time.time() - start_time) >= self.duration:
                    print(f"\nReached duration limit ({self.duration}s)")
                    break
                
                time.sleep(self.sample_interval)
                
            except KeyboardInterrupt:
                print("\n\nMonitoring interrupted by user")
                break
            except Exception as e:
                print(f"\nError in monitoring loop: {e}")
                break
        
        print(f"{'='*80}")
        print(f"Collected {sample_count} samples\n")
    
    def start(self):
        """Start monitoring in background thread"""
        self.running = True
        self.monitor_thread = threading.Thread(target=self.monitor_loop)
        self.monitor_thread.start()
    
    def stop(self):
        """Stop monitoring"""
        self.running = False
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join()
    
    def get_statistics(self):
        """Calculate statistics from collected metrics"""
        if not self.metrics_history:
            return {}
        
        stats = {'devices': []}
        
        for device_id in range(self.device_count):
            device_samples = [
                m['devices'][device_id] 
                for m in self.metrics_history 
                if device_id < len(m['devices'])
            ]
            
            if not device_samples:
                continue
            
            gpu_utils = [s['gpu_utilization_pct'] for s in device_samples]
            mem_utils = [s['memory_utilization_pct'] for s in device_samples]
            temps = [s['temperature_c'] for s in device_samples]
            powers = [s['power_watts'] for s in device_samples]
            
            device_stats = {
                'device_id': device_id,
                'gpu_utilization': {
                    'mean': sum(gpu_utils) / len(gpu_utils),
                    'min': min(gpu_utils),
                    'max': max(gpu_utils),
                    'p50': sorted(gpu_utils)[len(gpu_utils)//2],
                    'p95': sorted(gpu_utils)[int(len(gpu_utils)*0.95)]
                },
                'memory_utilization': {
                    'mean': sum(mem_utils) / len(mem_utils),
                    'min': min(mem_utils),
                    'max': max(mem_utils),
                    'p50': sorted(mem_utils)[len(mem_utils)//2],
                    'p95': sorted(mem_utils)[int(len(mem_utils)*0.95)]
                },
                'temperature': {
                    'mean': sum(temps) / len(temps),
                    'min': min(temps),
                    'max': max(temps)
                },
                'power': {
                    'mean': sum(powers) / len(powers),
                    'min': min(powers),
                    'max': max(powers)
                },
                'samples': len(device_samples)
            }
            
            stats['devices'].append(device_stats)
        
        return stats
    
    def save_results(self, filename=None):
        """Save monitoring results to file"""
        if not filename:
            filename = f'results/gpu_monitor_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        
        os.makedirs('results', exist_ok=True)
        
        results = {
            'start_time': self.metrics_history[0]['timestamp'] if self.metrics_history else None,
            'end_time': self.metrics_history[-1]['timestamp'] if self.metrics_history else None,
            'sample_interval': self.sample_interval,
            'total_samples': len(self.metrics_history),
            'statistics': self.get_statistics(),
            'raw_metrics': self.metrics_history
        }
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nMonitoring results saved to: {filename}")
        
        # Print summary
        stats = results['statistics']
        if stats and 'devices' in stats:
            print(f"\n{'='*60}")
            print("MONITORING SUMMARY")
            print(f"{'='*60}")
            for device_stats in stats['devices']:
                print(f"\nGPU {device_stats['device_id']}:")
                print(f"  GPU Utilization: {device_stats['gpu_utilization']['mean']:.1f}% "
                      f"(min: {device_stats['gpu_utilization']['min']:.1f}%, "
                      f"max: {device_stats['gpu_utilization']['max']:.1f}%)")
                print(f"  Memory Utilization: {device_stats['memory_utilization']['mean']:.1f}% "
                      f"(min: {device_stats['memory_utilization']['min']:.1f}%, "
                      f"max: {device_stats['memory_utilization']['max']:.1f}%)")
                print(f"  Temperature: {device_stats['temperature']['mean']:.1f}°C "
                      f"(min: {device_stats['temperature']['min']:.0f}°C, "
                      f"max: {device_stats['temperature']['max']:.0f}°C)")
                print(f"  Power: {device_stats['power']['mean']:.1f}W "
                      f"(min: {device_stats['power']['min']:.1f}W, "
                      f"max: {device_stats['power']['max']:.1f}W)")
            print(f"{'='*60}")
    
    def cleanup(self):
        """Cleanup NVML"""
        try:
            pynvml.nvmlShutdown()
        except:
            pass


def main():
    parser = argparse.ArgumentParser(description='GPU Monitoring Tool')
    parser.add_argument(
        '--interval',
        type=float,
        default=1.0,
        help='Sampling interval in seconds'
    )
    parser.add_argument(
        '--duration',
        type=int,
        default=None,
        help='Monitoring duration in seconds (default: infinite)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output file path'
    )
    
    args = parser.parse_args()
    
    try:
        monitor = GPUMonitor(sample_interval=args.interval, duration=args.duration)
        monitor.running = True
        monitor.monitor_loop()
        monitor.save_results(args.output)
        
    except KeyboardInterrupt:
        print("\nMonitoring stopped by user")
    except Exception as e:
        print(f"\nError: {e}")
    finally:
        if 'monitor' in locals():
            monitor.cleanup()


if __name__ == '__main__':
    main()

