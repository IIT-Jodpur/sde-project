```mermaid
flowchart LR
    subgraph Users
        R[Researcher / Engineer]
    end

    subgraph Workloads
        T[Training<br/>ResNet50 / BERT]
        I[Inference<br/>Batch / Server / Load Test]
        J[Interactive<br/>Jupyter Simulation]
    end

    subgraph Orchestration
        D[Docker Compose]
        K[Kubernetes]
    end

    subgraph GPU_Sharing
        TS[Time Slicing<br/>setup_time_slicing.sh]
        MPS[NVIDIA MPS<br/>enable_mps.sh]
        MIG[MIG Profiles<br/>setup_mig.sh]
    end

    subgraph Observability
        Bench[Benchmark Suite<br/>benchmark_suite.py]
        Mon[GPU Monitor<br/>monitoring/gpu_monitor.py]
        Results[Results & Reports<br/>results/*.json, *.png]
    end

    R --> Workloads
    Workloads --> Bench
    Bench --> Orchestration
    Orchestration --> GPU_Sharing
    GPU_Sharing --> GPU[(Physical GPU)]
    Mon --> GPU
    Mon --> Results
    Bench --> Results
```

**Key:** Users submit experiments that trigger the workload suite. Docker or Kubernetes orchestrates containers configured with a specific GPU sharing mode before executing on the physical GPU. Benchmarking and monitoring pipelines collect metrics and persist them under `results/`.

## Workload Lifecycle

This diagram captures the canonical workflow used in documentation (`PROJECT_STRUCTURE.md`, `EXPERIMENTS_GUIDE.md`) for executing experiments end to end.

```mermaid
flowchart LR
    Setup[Setup & Build\nrequirements.txt, docker/build_all.sh]
    Baseline[Run Baseline\nSingle workload]
    Configure[Configure GPU Sharing\nsetup_time_slicing.sh &#124; enable_mps.sh &#124; setup_mig.sh]
    Benchmark[Run Benchmarks\nbenchmark_suite.py, run_experiments.sh]
    Monitor[Monitor GPU\nmonitoring/gpu_monitor.py]
    Analyze[Analyze Results\nbenchmarking/analyze_results.py]
    Document[Document Findings\nRESULTS_TEMPLATE.md]

    Setup --> Baseline --> Configure --> Benchmark --> Monitor --> Analyze --> Document
```

This illustrates the hand-offs between automation scripts: initial environment preparation, baseline runs, GPU mode configuration, concurrent benchmarking, real-time monitoring, post-processing, and documentation.

## GPU Sharing Topology

The final diagram compares how containers reach the GPU across Time Slicing, MPS, and MIG. It highlights isolation vs. concurrency trade-offs that the project studies.

```mermaid
flowchart TB
    subgraph Containers
        C1[Training Container]
        C2[Inference Container]
        C3[Interactive Container]
        C4[Extra Workload]
    end

    C1 --> Switch{Mode Selector}
    C2 --> Switch
    C3 --> Switch
    C4 --> Switch

    Switch -->|Time Slicing| TSFlow[Context Switch Scheduler\nnvidia-smi --compute-mode]
    Switch -->|MPS| MPSFlow[MPS Control Daemon\nCUDA multi-process service]
    Switch -->|MIG| MIGFlow[MIG Partitions\n1g.5gb / 2g.10gb ...]

    TSFlow --> GPU[(Single GPU\nShared timeline)]
    MPSFlow --> GPU
    MIGFlow -->|Isolated Instances| GPU
```

**Trade-offs:**
- **Time Slicing:** Software multiplexing; good utilization, added latency.
- **MPS:** Concurrent kernels; lower overhead, limited isolation.
- **MIG:** Hardware partitions; strict isolation, fixed capacity slices.
