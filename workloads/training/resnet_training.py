"""
ResNet Training Workload
Simulates a typical deep learning training workload with ResNet50 on CIFAR-10
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import time
import argparse
import json
import os
from datetime import datetime


class GPUMetrics:
    """Track GPU metrics during training"""
    
    def __init__(self):
        try:
            import pynvml
            pynvml.nvmlInit()
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self.enabled = True
        except:
            self.enabled = False
            print("Warning: GPU monitoring not available")
    
    def get_metrics(self):
        if not self.enabled:
            return {}
        
        try:
            import pynvml
            info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
            util = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
            
            return {
                'memory_used_mb': info.used / 1024**2,
                'memory_total_mb': info.total / 1024**2,
                'gpu_utilization': util.gpu,
                'memory_utilization': util.memory
            }
        except:
            return {}


def train_resnet(args):
    """Train ResNet50 on CIFAR-10"""
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize metrics tracker
    gpu_metrics = GPUMetrics()
    
    # Data transforms
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    # Load datasets
    print("Loading CIFAR-10 dataset...")
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train
    )
    trainloader = DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True, num_workers=2
    )
    
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test
    )
    testloader = DataLoader(
        testset, batch_size=args.batch_size, shuffle=False, num_workers=2
    )
    
    # Model
    print("Initializing ResNet50...")
    model = torchvision.models.resnet50(pretrained=False, num_classes=10)
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Training metrics
    training_stats = {
        'workload_type': 'training',
        'model': 'resnet50',
        'dataset': 'cifar10',
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'start_time': datetime.now().isoformat(),
        'epoch_stats': []
    }
    
    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        epoch_start = time.time()
        
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            if batch_idx % 100 == 0:
                metrics = gpu_metrics.get_metrics()
                print(f'Epoch: {epoch+1}/{args.epochs} | Batch: {batch_idx}/{len(trainloader)} | '
                      f'Loss: {loss.item():.3f} | Acc: {100.*correct/total:.2f}% | '
                      f'GPU: {metrics.get("gpu_utilization", "N/A")}%')
        
        epoch_time = time.time() - epoch_start
        scheduler.step()
        
        # Validation
        model.eval()
        test_loss = 0
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for inputs, targets in testloader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                test_total += targets.size(0)
                test_correct += predicted.eq(targets).sum().item()
        
        epoch_metrics = gpu_metrics.get_metrics()
        epoch_stats = {
            'epoch': epoch + 1,
            'train_loss': running_loss / len(trainloader),
            'train_acc': 100. * correct / total,
            'test_loss': test_loss / len(testloader),
            'test_acc': 100. * test_correct / test_total,
            'epoch_time': epoch_time,
            'samples_per_sec': len(trainset) / epoch_time,
            'gpu_metrics': epoch_metrics
        }
        training_stats['epoch_stats'].append(epoch_stats)
        
        print(f'\nEpoch {epoch+1} Summary:')
        print(f'  Train Loss: {epoch_stats["train_loss"]:.3f} | Train Acc: {epoch_stats["train_acc"]:.2f}%')
        print(f'  Test Loss: {epoch_stats["test_loss"]:.3f} | Test Acc: {epoch_stats["test_acc"]:.2f}%')
        print(f'  Time: {epoch_time:.2f}s | Throughput: {epoch_stats["samples_per_sec"]:.2f} samples/s\n')
    
    training_stats['end_time'] = datetime.now().isoformat()
    training_stats['total_time'] = sum(e['epoch_time'] for e in training_stats['epoch_stats'])
    
    # Save results
    os.makedirs('results', exist_ok=True)
    result_file = f'results/training_resnet_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(result_file, 'w') as f:
        json.dump(training_stats, f, indent=2)
    
    print(f"\nTraining completed! Results saved to {result_file}")
    
    # Save model
    if args.save_model:
        model_path = f'models/resnet50_cifar10_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pth'
        os.makedirs('models', exist_ok=True)
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ResNet50 Training Workload')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--save-model', action='store_true', help='Save trained model')
    
    args = parser.parse_args()
    train_resnet(args)

