# GPU Sharing Configurations

This directory contains scripts to configure different GPU sharing modes for containerized ML workloads.

## Available GPU Sharing Modes

### 1. Time Slicing (Default)
Multiple containers share GPU access through time-multiplexing. Best for mixed workloads.

```bash
./setup_time_slicing.sh [num_replicas]
```

**Parameters:**
- `num_replicas`: Number of concurrent containers (default: 4)

**Pros:**
- Works on all NVIDIA GPUs
- Simple to configure
- Flexible resource allocation

**Cons:**
- Context switching overhead
- No performance isolation guarantees

---

### 2. NVIDIA MPS (Multi-Process Service)
Enables parallel execution of CUDA kernels from multiple processes.

```bash
# Enable MPS
./enable_mps.sh [thread_percentage] [pinned_mem_limit]

# Disable MPS
./disable_mps.sh
```

**Parameters:**
- `thread_percentage`: GPU compute percentage per client (default: 100)
- `pinned_mem_limit`: Pinned memory limit in MB per client (default: 0 = unlimited)

**Pros:**
- Lower overhead than time slicing
- Better for small concurrent tasks
- Improved GPU utilization

**Cons:**
- Limited isolation between processes
- All processes must use same CUDA context
- Requires root privileges

**Environment Setup:**
```bash
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log
```

---

### 3. MIG (Multi-Instance GPU)
Hardware-level GPU partitioning. **Requires A100, A30, or H100 GPUs.**

```bash
# Enable MIG and create instances
sudo ./setup_mig.sh [profile] [num_instances]

# Disable MIG
sudo ./disable_mig.sh
```

**Parameters:**
- `profile`: MIG profile (default: "1g.5gb")
  - `1g.5gb`: 1 slice, 5GB memory
  - `2g.10gb`: 2 slices, 10GB memory
  - `3g.20gb`: 3 slices, 20GB memory
  - `7g.40gb`: Full GPU (A100-40GB)
  
- `num_instances`: Number of instances to create (default: 4)

**Pros:**
- Hardware-level isolation
- Guaranteed performance
- Each instance has dedicated resources

**Cons:**
- Requires specific GPU hardware (A100/A30/H100)
- Static partitioning
- System reboot may be required

---

## Configuration Comparison

| Feature | Time Slicing | MPS | MIG |
|---------|-------------|-----|-----|
| **GPU Support** | All NVIDIA GPUs | All NVIDIA GPUs | A100, A30, H100 only |
| **Isolation** | Process-level | Process-level | Hardware-level |
| **Overhead** | Medium | Low | None |
| **Flexibility** | High | Medium | Low (static) |
| **Setup Complexity** | Low | Medium | High |
| **Root Required** | No | Yes | Yes |

---

## Usage with Docker

### Time Slicing
```bash
# Run multiple containers sharing GPU
docker-compose up

# Or run individual containers
docker run --gpus all gpu-timeslicing/training:latest
```

### MPS
```bash
# Enable MPS
./enable_mps.sh

# Run containers with MPS volumes
docker run --gpus all \
  -v /tmp/nvidia-mps:/tmp/nvidia-mps \
  -v /tmp/nvidia-log:/tmp/nvidia-log \
  -e CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps \
  gpu-timeslicing/training:latest
```

### MIG
```bash
# Setup MIG
sudo ./setup_mig.sh 1g.5gb 4

# List MIG UUIDs
nvidia-smi -L

# Run container with specific MIG instance
docker run --gpus '"device=MIG-<UUID>"' gpu-timeslicing/training:latest
```

---

## Verification

Check GPU configuration:
```bash
# View GPU status
nvidia-smi

# Check MPS status (if enabled)
echo "get_server_list" | nvidia-cuda-mps-control

# Check MIG instances (if enabled)
nvidia-smi mig -lgi
```

---

## Troubleshooting

### Time Slicing Not Working
- Verify NVIDIA Container Toolkit is installed
- Check Docker daemon configuration includes NVIDIA runtime

### MPS Issues
- Ensure MPS daemon is running: `pgrep nvidia-cuda-mps`
- Check MPS logs in `/tmp/nvidia-log`
- Verify environment variables are set correctly

### MIG Not Available
- Confirm GPU model supports MIG (A100/A30/H100)
- Update NVIDIA drivers to latest version
- System reboot may be required after enabling MIG mode

---

## Best Practices

1. **Time Slicing**: Best for development and testing with mixed workloads
2. **MPS**: Use for production inference with many small concurrent requests
3. **MIG**: Use for production training/inference requiring strict isolation

Choose based on:
- Workload characteristics
- Performance requirements
- Hardware availability
- Isolation needs

