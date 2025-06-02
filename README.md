# GPU-Optimization-Case-Studies

This repository contains case studies demonstrating GPU optimization techniques for machine learning inference workloads.

## Case Study 1: Video Generation Inference Optimization

### Model: Zeroscope v2 576w

Optimizing inference performance for the `cerspense/zeroscope_v2_576w` video generation model.

### Optimization Techniques Applied

#### 1. CUDA Graph Capture
- Increased SM utilization to nearly 100%
- Enabled batch size increase from 4 to 10

#### 2. Torch.compile()
- Achieved 3x reduction in inference latency
- Increased TensorCore utilization
- Enabled batch size increase from 10 to 14

### Results

| Optimization Stage | Batch Size | Latency Improvement | Key Hardware Metric |
|-------------------|------------|-------------------|-------------------|
| Baseline | 4 | - | - |
| + CUDA Graph Capture | 10 | Significant reduction | ~100% SM Utilization |
| + Torch.compile() | 14 | 3x faster | High TensorCore Utilization |

### Files

- `Videogen_Inference_with_CUDA_Graph.ipynb`: Complete implementation with benchmarking and profiling setup

### Next Steps

#### Reduce Memory Fragmentation
Current GPU memory analysis shows high fragmentation that needs optimization:

```
GPU Memory Status (Device 0):
  Total memory          : 15095.06 MB
  Reserved by PyTorch    : 10104.00 MB
  Allocated by tensors   : 3529.53 MB
  Free inside reserved   : 6574.47 MB
  External Fragmentation : 65.07%
  NVML reported used mem : 11911.88 MB
```

**Target**: Reduce the 65.07% external fragmentation to improve memory efficiency and potentially enable larger batch sizes.

---

*This repository demonstrates practical GPU optimization techniques that can be applied to various deep learning inference workloads to achieve significant performance improvements.*