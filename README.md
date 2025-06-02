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

---

*This repository demonstrates practical GPU optimization techniques that can be applied to various deep learning inference workloads to achieve significant performance improvements.*