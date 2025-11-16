"""
Load test for inference server
Generates concurrent requests to measure throughput and latency
"""

import requests
import time
import argparse
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import numpy as np
import io
from PIL import Image


class InferenceLoadTester:
    """Load tester for inference server"""
    
    def __init__(self, server_url, num_clients=10, duration=60):
        self.server_url = server_url
        self.num_clients = num_clients
        self.duration = duration
        self.results = []
        self.lock = threading.Lock()
        self.running = False
    
    def generate_dummy_image(self):
        """Generate dummy image for testing"""
        img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='JPEG')
        img_bytes.seek(0)
        return img_bytes
    
    def send_request(self, client_id):
        """Send single inference request"""
        try:
            start_time = time.time()
            
            img_bytes = self.generate_dummy_image()
            files = {'image': ('test.jpg', img_bytes, 'image/jpeg')}
            
            response = requests.post(
                f'{self.server_url}/predict',
                files=files,
                timeout=30
            )
            
            end_time = time.time()
            latency = (end_time - start_time) * 1000  # ms
            
            result = {
                'client_id': client_id,
                'timestamp': datetime.now().isoformat(),
                'latency_ms': latency,
                'success': response.status_code == 200,
                'status_code': response.status_code
            }
            
            with self.lock:
                self.results.append(result)
            
            return result
            
        except Exception as e:
            result = {
                'client_id': client_id,
                'timestamp': datetime.now().isoformat(),
                'success': False,
                'error': str(e)
            }
            
            with self.lock:
                self.results.append(result)
            
            return result
    
    def client_loop(self, client_id):
        """Continuous request loop for one client"""
        print(f"[Client {client_id}] Started")
        start_time = time.time()
        request_count = 0
        
        while self.running and (time.time() - start_time) < self.duration:
            self.send_request(client_id)
            request_count += 1
            time.sleep(0.1)  # Small delay between requests
        
        print(f"[Client {client_id}] Completed {request_count} requests")
    
    def run(self):
        """Run load test"""
        print(f"\n{'='*60}")
        print("INFERENCE SERVER LOAD TEST")
        print(f"{'='*60}")
        print(f"Server: {self.server_url}")
        print(f"Concurrent clients: {self.num_clients}")
        print(f"Duration: {self.duration}s")
        print(f"{'='*60}\n")
        
        # Check server health
        try:
            response = requests.get(f'{self.server_url}/health', timeout=5)
            if response.status_code != 200:
                print(f"Error: Server health check failed (status: {response.status_code})")
                return
            print("✓ Server is healthy\n")
        except Exception as e:
            print(f"Error: Cannot connect to server - {e}")
            return
        
        # Start load test
        self.running = True
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=self.num_clients) as executor:
            futures = [
                executor.submit(self.client_loop, i)
                for i in range(self.num_clients)
            ]
            
            # Wait for completion
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"Client error: {e}")
        
        end_time = time.time()
        total_duration = end_time - start_time
        
        # Analyze results
        self.analyze_results(total_duration)
    
    def analyze_results(self, total_duration):
        """Analyze and display results"""
        successful = [r for r in self.results if r.get('success', False)]
        failed = [r for r in self.results if not r.get('success', False)]
        
        print(f"\n{'='*60}")
        print("LOAD TEST RESULTS")
        print(f"{'='*60}")
        print(f"Duration: {total_duration:.2f}s")
        print(f"Total requests: {len(self.results)}")
        print(f"Successful: {len(successful)}")
        print(f"Failed: {len(failed)}")
        print(f"Success rate: {len(successful)/len(self.results)*100:.1f}%")
        
        if successful:
            latencies = [r['latency_ms'] for r in successful]
            
            print(f"\nLatency Statistics:")
            print(f"  Mean: {np.mean(latencies):.2f}ms")
            print(f"  Median (P50): {np.percentile(latencies, 50):.2f}ms")
            print(f"  P95: {np.percentile(latencies, 95):.2f}ms")
            print(f"  P99: {np.percentile(latencies, 99):.2f}ms")
            print(f"  Min: {np.min(latencies):.2f}ms")
            print(f"  Max: {np.max(latencies):.2f}ms")
            
            print(f"\nThroughput:")
            qps = len(successful) / total_duration
            print(f"  Queries per second: {qps:.2f}")
            print(f"  Requests per minute: {qps * 60:.2f}")
        
        print(f"{'='*60}\n")
        
        # Save results
        self.save_results(total_duration)
    
    def save_results(self, total_duration):
        """Save results to file"""
        successful = [r for r in self.results if r.get('success', False)]
        latencies = [r['latency_ms'] for r in successful] if successful else []
        
        output = {
            'test_config': {
                'server_url': self.server_url,
                'num_clients': self.num_clients,
                'duration': self.duration
            },
            'summary': {
                'total_duration': total_duration,
                'total_requests': len(self.results),
                'successful_requests': len(successful),
                'failed_requests': len(self.results) - len(successful),
                'success_rate': len(successful) / len(self.results) * 100 if self.results else 0,
                'qps': len(successful) / total_duration if total_duration > 0 else 0
            },
            'latency_stats': {
                'mean_ms': float(np.mean(latencies)) if latencies else 0,
                'median_ms': float(np.median(latencies)) if latencies else 0,
                'p95_ms': float(np.percentile(latencies, 95)) if latencies else 0,
                'p99_ms': float(np.percentile(latencies, 99)) if latencies else 0,
                'min_ms': float(np.min(latencies)) if latencies else 0,
                'max_ms': float(np.max(latencies)) if latencies else 0
            },
            'raw_results': self.results
        }
        
        filename = f'results/load_test_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(filename, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"Results saved to: {filename}\n")


def main():
    parser = argparse.ArgumentParser(description='Inference Server Load Tester')
    parser.add_argument(
        '--server',
        type=str,
        default='http://localhost:5000',
        help='Server URL'
    )
    parser.add_argument(
        '--clients',
        type=int,
        default=10,
        help='Number of concurrent clients'
    )
    parser.add_argument(
        '--duration',
        type=int,
        default=60,
        help='Test duration in seconds'
    )
    
    args = parser.parse_args()
    
    tester = InferenceLoadTester(
        server_url=args.server,
        num_clients=args.clients,
        duration=args.duration
    )
    tester.run()


if __name__ == '__main__':
    main()

