# Kubernetes Deployment Guide

This directory contains Kubernetes manifests for deploying GPU workloads with different sharing strategies.

## Prerequisites

1. **Kubernetes cluster with GPU nodes**
2. **NVIDIA GPU Operator installed**
   ```bash
   kubectl apply -f https://raw.githubusercontent.com/NVIDIA/gpu-operator/master/deployments/gpu-operator/gpu-operator.yaml
   ```

3. **NVIDIA Device Plugin configured**

## Deployment Options

### 1. Basic Pod Deployment (Time Slicing)

Deploy individual workload pods:

```bash
# Create namespace
kubectl create namespace gpu-workloads

# Apply time slicing configuration
kubectl create -n gpu-operator configmap time-slicing-config --from-file=../gpu-configs/dp-config.yaml

# Patch GPU operator to use time slicing
kubectl patch clusterpolicy/cluster-policy \
  -n gpu-operator \
  --type merge \
  -p '{"spec":{"devicePlugin":{"config":{"name":"time-slicing-config"}}}}'

# Deploy workloads
kubectl apply -f gpu-workloads.yaml
```

Check status:
```bash
kubectl get pods -n gpu-workloads
kubectl logs -n gpu-workloads training-workload-1
```

### 2. Inference Server Deployment

Deploy scalable inference service:

```bash
kubectl apply -f inference-deployment.yaml
```

This creates:
- Deployment with 2 replicas
- LoadBalancer service
- Health checks

Access the service:
```bash
# Get service endpoint
kubectl get svc -n gpu-workloads inference-service

# Test inference
curl -X POST http://<EXTERNAL-IP>/predict \
  -F "image=@sample_image.jpg"
```

### 3. Batch Job for Training

Deploy training as a Kubernetes Job:

```bash
kubectl apply -f training-job.yaml
```

Monitor progress:
```bash
kubectl logs -n gpu-workloads -f job/training-job
```

## GPU Sharing Configuration

### Time Slicing
Configured via ConfigMap. Allows multiple pods to share GPU by time-multiplexing.

```yaml
version: v1
sharing:
  timeSlicing:
    replicas: 4  # Number of pods that can share GPU
    failRequestsGreaterThanOne: true
```

### MIG (Multi-Instance GPU)
For A100/H100 GPUs, configure MIG profiles:

```bash
# Label nodes with MIG configuration
kubectl label nodes <node-name> nvidia.com/mig.config=all-1g.5gb

# Update GPU operator
kubectl patch clusterpolicy/cluster-policy \
  -n gpu-operator \
  --type merge \
  -p '{"spec":{"mig":{"strategy":"mixed"}}}'
```

Use MIG in pod spec:
```yaml
resources:
  limits:
    nvidia.com/mig-1g.5gb: 1
```

## Monitoring

Deploy GPU monitoring dashboard:

```bash
kubectl apply -f monitoring/gpu-monitor-daemonset.yaml
```

View metrics:
```bash
kubectl port-forward -n gpu-workloads svc/gpu-monitor 8080:80
```

## Resource Quotas

Limit GPU usage per namespace:

```yaml
apiVersion: v1
kind: ResourceQuota
metadata:
  name: gpu-quota
  namespace: gpu-workloads
spec:
  hard:
    requests.nvidia.com/gpu: "4"
    limits.nvidia.com/gpu: "4"
```

## Troubleshooting

### Pods stuck in Pending
```bash
# Check GPU availability
kubectl describe node <gpu-node-name> | grep nvidia.com/gpu

# Check device plugin
kubectl logs -n gpu-operator -l app=nvidia-device-plugin-daemonset
```

### Time slicing not working
```bash
# Verify ConfigMap
kubectl get cm -n gpu-operator time-slicing-config -o yaml

# Check ClusterPolicy
kubectl get clusterpolicy -n gpu-operator cluster-policy -o yaml

# Restart device plugin
kubectl delete pod -n gpu-operator -l app=nvidia-device-plugin-daemonset
```

### GPU not accessible in pod
```bash
# Check nvidia-smi in pod
kubectl exec -it -n gpu-workloads <pod-name> -- nvidia-smi

# Check environment variables
kubectl exec -it -n gpu-workloads <pod-name> -- env | grep NVIDIA
```

## Best Practices

1. **Use resource limits**: Always specify GPU limits in pod specs
2. **Health checks**: Implement readiness/liveness probes for long-running services
3. **Monitoring**: Deploy monitoring to track GPU utilization
4. **Node affinity**: Use node selectors for specific GPU types
5. **Graceful shutdown**: Handle SIGTERM for proper cleanup

## Cleanup

Remove all resources:

```bash
kubectl delete namespace gpu-workloads

# Remove time slicing config (optional)
kubectl delete cm -n gpu-operator time-slicing-config
```

