"""
BERT Fine-tuning Workload
Simulates transformer-based model training (NLP task)
"""

import torch
import torch.nn as nn
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import Dataset, DataLoader
import time
import argparse
import json
import os
from datetime import datetime
import numpy as np


class SyntheticTextDataset(Dataset):
    """Synthetic text classification dataset for benchmarking"""
    
    def __init__(self, num_samples=1000, max_length=128):
        self.num_samples = num_samples
        self.max_length = max_length
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
        # Generate synthetic sentences
        templates = [
            "This is a positive sentence about {topic}.",
            "I really enjoyed the {topic} experience.",
            "The {topic} was terrible and disappointing.",
            "Not satisfied with the {topic} at all.",
            "The {topic} exceeded all my expectations.",
        ]
        topics = ["movie", "product", "service", "book", "restaurant"]
        
        self.data = []
        for i in range(num_samples):
            template = templates[i % len(templates)]
            topic = topics[i % len(topics)]
            text = template.format(topic=topic)
            label = 1 if "positive" in template or "enjoyed" in template or "exceeded" in template else 0
            self.data.append((text, label))
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        text, label = self.data[idx]
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(label)
        }


def train_bert(args):
    """Fine-tune BERT for sequence classification"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print("Loading BERT model...")
    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased',
        num_labels=2
    )
    model = model.to(device)
    
    # Prepare dataset
    print("Preparing dataset...")
    train_dataset = SyntheticTextDataset(num_samples=args.num_samples)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    
    # Optimizer
    optimizer = AdamW(model.parameters(), lr=args.lr)
    
    # Training stats
    training_stats = {
        'workload_type': 'training',
        'model': 'bert-base-uncased',
        'task': 'sequence_classification',
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'start_time': datetime.now().isoformat(),
        'epoch_stats': []
    }
    
    # Training loop
    print(f"\nStarting BERT fine-tuning for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        epoch_start = time.time()
        
        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            predictions = torch.argmax(outputs.logits, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            
            if batch_idx % 20 == 0:
                print(f'Epoch: {epoch+1}/{args.epochs} | Batch: {batch_idx}/{len(train_loader)} | '
                      f'Loss: {loss.item():.4f} | Acc: {100.*correct/total:.2f}%')
        
        epoch_time = time.time() - epoch_start
        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total
        
        epoch_stats = {
            'epoch': epoch + 1,
            'loss': avg_loss,
            'accuracy': accuracy,
            'epoch_time': epoch_time,
            'samples_per_sec': len(train_dataset) / epoch_time
        }
        training_stats['epoch_stats'].append(epoch_stats)
        
        print(f'\nEpoch {epoch+1} Summary:')
        print(f'  Loss: {avg_loss:.4f} | Accuracy: {accuracy:.2f}%')
        print(f'  Time: {epoch_time:.2f}s | Throughput: {epoch_stats["samples_per_sec"]:.2f} samples/s\n')
    
    training_stats['end_time'] = datetime.now().isoformat()
    
    # Save results
    os.makedirs('results', exist_ok=True)
    result_file = f'results/training_bert_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(result_file, 'w') as f:
        json.dump(training_stats, f, indent=2)
    
    print(f"\nTraining completed! Results saved to {result_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='BERT Fine-tuning Workload')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--epochs', type=int, default=3, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--num-samples', type=int, default=1000, help='Number of training samples')
    
    args = parser.parse_args()
    train_bert(args)

