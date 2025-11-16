"""
Interactive Jupyter-style Workload
Simulates interactive ML development with periodic GPU access
"""

import torch
import torch.nn as nn
import time
import random
import argparse
import json
import os
from datetime import datetime


class InteractiveSession:
    """Simulates an interactive ML session with bursty GPU usage"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Interactive session started on {self.device}")
        
        self.operations = []
        self.start_time = datetime.now()
    
    def matrix_operation(self, size=1000, duration=2):
        """Simulate matrix operations (data exploration)"""
        print(f"\n[Cell Execution] Running matrix operations ({size}x{size})...")
        start = time.time()
        
        a = torch.randn(size, size).to(self.device)
        b = torch.randn(size, size).to(self.device)
        
        for _ in range(100):
            c = torch.matmul(a, b)
            c = torch.relu(c)
        
        elapsed = time.time() - start
        self.operations.append({
            'type': 'matrix_operation',
            'size': size,
            'duration': elapsed,
            'timestamp': datetime.now().isoformat()
        })
        
        print(f"Completed in {elapsed:.2f}s")
        return c
    
    def model_training_burst(self, epochs=5):
        """Simulate short training burst"""
        print(f"\n[Cell Execution] Quick model training ({epochs} epochs)...")
        start = time.time()
        
        # Simple model
        model = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        ).to(self.device)
        
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()
        
        # Fake training
        for epoch in range(epochs):
            batch = torch.randn(32, 784).to(self.device)
            labels = torch.randint(0, 10, (32,)).to(self.device)
            
            optimizer.zero_grad()
            output = model(batch)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            
            if epoch % 2 == 0:
                print(f"  Epoch {epoch+1}/{epochs} - Loss: {loss.item():.4f}")
        
        elapsed = time.time() - start
        self.operations.append({
            'type': 'training_burst',
            'epochs': epochs,
            'duration': elapsed,
            'timestamp': datetime.now().isoformat()
        })
        
        print(f"Training completed in {elapsed:.2f}s")
    
    def inference_test(self, num_samples=100):
        """Simulate inference testing"""
        print(f"\n[Cell Execution] Testing inference on {num_samples} samples...")
        start = time.time()
        
        model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
        model = model.to(self.device)
        model.eval()
        
        with torch.no_grad():
            for _ in range(num_samples):
                input_data = torch.randn(1, 3, 224, 224).to(self.device)
                output = model(input_data)
        
        elapsed = time.time() - start
        self.operations.append({
            'type': 'inference_test',
            'num_samples': num_samples,
            'duration': elapsed,
            'timestamp': datetime.now().isoformat()
        })
        
        print(f"Inference testing completed in {elapsed:.2f}s")
    
    def idle_period(self, duration=5):
        """Simulate thinking/coding time (no GPU usage)"""
        print(f"\n[Idle] Thinking/Coding for {duration}s...")
        time.sleep(duration)
        
        self.operations.append({
            'type': 'idle',
            'duration': duration,
            'timestamp': datetime.now().isoformat()
        })
    
    def data_preprocessing(self, num_operations=50):
        """Simulate data preprocessing"""
        print(f"\n[Cell Execution] Data preprocessing...")
        start = time.time()
        
        for _ in range(num_operations):
            data = torch.randn(100, 100).to(self.device)
            data = (data - data.mean()) / data.std()
            data = torch.nn.functional.interpolate(
                data.unsqueeze(0).unsqueeze(0), 
                size=(150, 150), 
                mode='bilinear'
            )
        
        elapsed = time.time() - start
        self.operations.append({
            'type': 'data_preprocessing',
            'num_operations': num_operations,
            'duration': elapsed,
            'timestamp': datetime.now().isoformat()
        })
        
        print(f"Preprocessing completed in {elapsed:.2f}s")
    
    def save_stats(self):
        """Save session statistics"""
        stats = {
            'workload_type': 'interactive',
            'start_time': self.start_time.isoformat(),
            'end_time': datetime.now().isoformat(),
            'total_duration': (datetime.now() - self.start_time).total_seconds(),
            'operations': self.operations,
            'summary': {
                'total_operations': len(self.operations),
                'gpu_time': sum(op['duration'] for op in self.operations if op['type'] != 'idle'),
                'idle_time': sum(op['duration'] for op in self.operations if op['type'] == 'idle')
            }
        }
        
        os.makedirs('results', exist_ok=True)
        result_file = f'results/interactive_session_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(result_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"\n{'='*60}")
        print("INTERACTIVE SESSION SUMMARY")
        print(f"{'='*60}")
        print(f"Total Operations: {stats['summary']['total_operations']}")
        print(f"GPU Active Time: {stats['summary']['gpu_time']:.2f}s")
        print(f"Idle Time: {stats['summary']['idle_time']:.2f}s")
        print(f"Total Session Time: {stats['total_duration']:.2f}s")
        print(f"GPU Utilization: {100 * stats['summary']['gpu_time'] / stats['total_duration']:.1f}%")
        print(f"{'='*60}")
        print(f"\nResults saved to {result_file}")


def run_interactive_session(args):
    """Run simulated interactive session"""
    
    session = InteractiveSession()
    
    # Simulate typical interactive workflow
    workflows = [
        ('matrix_operation', {'size': 2000}),
        ('idle_period', {'duration': 3}),
        ('data_preprocessing', {'num_operations': 30}),
        ('idle_period', {'duration': 5}),
        ('model_training_burst', {'epochs': 10}),
        ('idle_period', {'duration': 4}),
        ('inference_test', {'num_samples': 50}),
        ('idle_period', {'duration': 3}),
        ('matrix_operation', {'size': 1500}),
        ('idle_period', {'duration': 2}),
        ('model_training_burst', {'epochs': 5}),
    ]
    
    print("="*60)
    print("SIMULATED INTERACTIVE ML SESSION")
    print("="*60)
    print("This simulates a typical Jupyter notebook workflow with")
    print("periodic GPU usage and idle periods between cell executions.")
    print("="*60)
    
    for operation, kwargs in workflows[:args.num_operations]:
        getattr(session, operation)(**kwargs)
    
    session.save_stats()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Interactive Workload Simulation')
    parser.add_argument('--num-operations', type=int, default=11, 
                       help='Number of operations to execute')
    
    args = parser.parse_args()
    run_interactive_session(args)

