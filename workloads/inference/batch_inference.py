"""
Batch Inference Workload
Simulates high-throughput batch inference scenarios
"""

import torch
import torchvision.models as models
import time
import argparse
import json
import os
from datetime import datetime
import numpy as np


def run_batch_inference(args):
    """Run batch inference benchmark"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading {args.model}...")
    if args.model == 'resnet50':
        model = models.resnet50(pretrained=True)
    elif args.model == 'resnet18':
        model = resnet18(pretrained=True)
    elif args.model == 'vgg16':
        model = models.vgg16(pretrained=True)
    else:
        raise ValueError(f"Unknown model: {args.model}")
    
    model = model.to(device)
    model.eval()
    
    # Warm-up
    print("Warming up...")
    dummy_input = torch.randn(args.batch_size, 3, 224, 224).to(device)
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)
    
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    
    # Benchmark
    print(f"\nRunning inference with batch size {args.batch_size} for {args.num_batches} batches...")
    
    latencies = []
    throughputs = []
    
    for i in range(args.num_batches):
        batch_input = torch.randn(args.batch_size, 3, 224, 224).to(device)
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.time()
        
        with torch.no_grad():
            outputs = model(batch_input)
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end_time = time.time()
        
        latency = (end_time - start_time) * 1000  # ms
        throughput = args.batch_size / (latency / 1000)  # samples/sec
        
        latencies.append(latency)
        throughputs.append(throughput)
        
        if i % 10 == 0:
            print(f"Batch {i}/{args.num_batches} | Latency: {latency:.2f}ms | "
                  f"Throughput: {throughput:.2f} samples/s")
    
    # Calculate statistics
    stats = {
        'workload_type': 'inference',
        'model': args.model,
        'batch_size': args.batch_size,
        'num_batches': args.num_batches,
        'device': str(device),
        'timestamp': datetime.now().isoformat(),
        'latency_stats': {
            'mean_ms': float(np.mean(latencies)),
            'median_ms': float(np.median(latencies)),
            'std_ms': float(np.std(latencies)),
            'min_ms': float(np.min(latencies)),
            'max_ms': float(np.max(latencies)),
            'p50_ms': float(np.percentile(latencies, 50)),
            'p95_ms': float(np.percentile(latencies, 95)),
            'p99_ms': float(np.percentile(latencies, 99))
        },
        'throughput_stats': {
            'mean_samples_per_sec': float(np.mean(throughputs)),
            'median_samples_per_sec': float(np.median(throughputs)),
            'std_samples_per_sec': float(np.std(throughputs))
        }
    }
    
    # Print summary
    print("\n" + "="*60)
    print("INFERENCE BENCHMARK RESULTS")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Total Batches: {args.num_batches}")
    print(f"\nLatency Statistics:")
    print(f"  Mean: {stats['latency_stats']['mean_ms']:.2f} ms")
    print(f"  Median (P50): {stats['latency_stats']['p50_ms']:.2f} ms")
    print(f"  P95: {stats['latency_stats']['p95_ms']:.2f} ms")
    print(f"  P99: {stats['latency_stats']['p99_ms']:.2f} ms")
    print(f"\nThroughput Statistics:")
    print(f"  Mean: {stats['throughput_stats']['mean_samples_per_sec']:.2f} samples/s")
    print(f"  Median: {stats['throughput_stats']['median_samples_per_sec']:.2f} samples/s")
    print("="*60)
    
    # Save results
    os.makedirs('results', exist_ok=True)
    result_file = f'results/inference_{args.model}_bs{args.batch_size}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(result_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\nResults saved to {result_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Batch Inference Workload')
    parser.add_argument('--model', type=str, default='resnet50', 
                       choices=['resnet50', 'resnet18', 'vgg16'])
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--num-batches', type=int, default=100, help='Number of batches')
    
    args = parser.parse_args()
    run_batch_inference(args)

