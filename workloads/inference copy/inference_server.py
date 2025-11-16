"""
Image Classification Inference Server
High-throughput inference workload using ResNet50
"""

import torch
import torchvision.transforms as transforms
import torchvision.models as models
from flask import Flask, request, jsonify
import io
import time
import json
import threading
from PIL import Image
from datetime import datetime
import argparse
from collections import deque


app = Flask(__name__)

# Global variables
model = None
device = None
inference_stats = {
    'total_requests': 0,
    'total_latency': 0.0,
    'latencies': deque(maxlen=1000),
    'start_time': datetime.now().isoformat()
}
stats_lock = threading.Lock()


def load_model(model_name='resnet50'):
    """Load pre-trained model"""
    global model, device
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading {model_name} on {device}...")
    
    if model_name == 'resnet50':
        model = models.resnet50(pretrained=True)
    elif model_name == 'resnet18':
        model = models.resnet18(pretrained=True)
    elif model_name == 'mobilenet':
        model = models.mobilenet_v2(pretrained=True)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    model = model.to(device)
    model.eval()
    
    # Warm-up
    print("Warming up model...")
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)
    
    print("Model loaded and ready!")


def preprocess_image(image_bytes):
    """Preprocess image for inference"""
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    return transform(image).unsqueeze(0)


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'device': str(device)})


@app.route('/predict', methods=['POST'])
def predict():
    """Inference endpoint"""
    global inference_stats
    
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    start_time = time.time()
    
    try:
        # Read and preprocess image
        image_bytes = request.files['image'].read()
        input_tensor = preprocess_image(image_bytes).to(device)
        
        # Inference
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
        
        # Get top-5 predictions
        top5_prob, top5_idx = torch.topk(probabilities, 5)
        top5_prob = top5_prob.cpu().numpy().tolist()
        top5_idx = top5_idx.cpu().numpy().tolist()
        
        latency = (time.time() - start_time) * 1000  # ms
        
        # Update stats
        with stats_lock:
            inference_stats['total_requests'] += 1
            inference_stats['total_latency'] += latency
            inference_stats['latencies'].append(latency)
        
        return jsonify({
            'predictions': [
                {'class_id': int(idx), 'probability': float(prob)}
                for idx, prob in zip(top5_idx, top5_prob)
            ],
            'latency_ms': latency
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/stats', methods=['GET'])
def stats():
    """Get inference statistics"""
    with stats_lock:
        latencies = list(inference_stats['latencies'])
        return jsonify({
            'total_requests': inference_stats['total_requests'],
            'avg_latency_ms': inference_stats['total_latency'] / max(inference_stats['total_requests'], 1),
            'p50_latency_ms': sorted(latencies)[len(latencies)//2] if latencies else 0,
            'p95_latency_ms': sorted(latencies)[int(len(latencies)*0.95)] if latencies else 0,
            'p99_latency_ms': sorted(latencies)[int(len(latencies)*0.99)] if latencies else 0,
            'throughput_qps': inference_stats['total_requests'] / max((datetime.now() - datetime.fromisoformat(inference_stats['start_time'])).total_seconds(), 1),
            'start_time': inference_stats['start_time']
        })


@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """Batch inference endpoint"""
    if 'batch_size' not in request.json:
        return jsonify({'error': 'batch_size required'}), 400
    
    batch_size = request.json['batch_size']
    start_time = time.time()
    
    try:
        # Create dummy batch for benchmarking
        dummy_batch = torch.randn(batch_size, 3, 224, 224).to(device)
        
        with torch.no_grad():
            outputs = model(dummy_batch)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
        
        latency = (time.time() - start_time) * 1000  # ms
        
        return jsonify({
            'batch_size': batch_size,
            'latency_ms': latency,
            'throughput_samples_per_sec': batch_size / (latency / 1000)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inference Server')
    parser.add_argument('--model', type=str, default='resnet50', choices=['resnet50', 'resnet18', 'mobilenet'])
    parser.add_argument('--port', type=int, default=5000, help='Server port')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Server host')
    
    args = parser.parse_args()
    
    load_model(args.model)
    app.run(host=args.host, port=args.port, threaded=True)

