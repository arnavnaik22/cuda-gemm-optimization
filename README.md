# CUDA-Accelerated GEMM Optimization & Performance Analysis

A C++/CUDA systems engineering project implementing, benchmarking, and profiling General Matrix Multiplication (GEMM) algorithms on NVIDIA Turing architecture (Tesla T4 GPU, Compute Capability 7.5).

---

## 1. Architecture & Memory Hierarchy Implementation

* **CPU Baseline Reference**: Triply nested loop $O(N^3)$ implementation used to validate numerical correctness ($\epsilon \le 10^{-2}$).
* **Naive CUDA Kernel**: Maps thread coordinates $(x, y)$ directly to matrix rows and columns with uncoalesced/redundant global memory (DRAM) accesses.
* **Shared Memory Tiled Kernel**:
  * Partitions matrices into $32 \times 32$ sub-tiles staged into low-latency shared memory (`__shared__`).
  * Allocates **8.19 KB** static shared memory per block ($2 \times 32 \times 32 \times 4\text{ bytes}$).
  * Synchronizes execution across sub-stages using `__syncthreads()` barrier primitives to prevent data hazards.

---

## 2. Empirical Benchmarks (Tesla T4 GPU)

Compiled with: `nvcc -O3 -arch=sm_75 -lineinfo`

| Matrix Dim ($N \times N$) | Naive Latency (ms) | Tiled Latency (ms) | Speedup | Tiled Throughput (GFLOPS) | Correctness Check |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **256 x 256** | 0.13 ms | 0.11 ms | 1.20x | 313.88 GFLOPS | **PASS** |
| **512 x 512** | 0.64 ms | 0.52 ms | 1.25x | 521.23 GFLOPS | **PASS** |
| **1024 x 1024** | 5.25 ms | 4.12 ms | 1.28x | 521.37 GFLOPS | **PASS** |
| **2048 x 2048** | 43.64 ms | 20.16 ms | 2.16x | 852.10 GFLOPS | **PASS** |
| **4096 x 4096** | 231.62 ms | 156.39 ms | 1.48x | 878.85 GFLOPS | **PASS** |

---

## 3. NVIDIA Nsight Compute (`ncu`) Hardware Profiling Analysis

* **Profiling Setup**: Matrix dimension $N = 2048$ profiled on an NVIDIA Tesla T4 GPU (`ncu --set basic profile_runner`).

### Metric Comparison Table

| Metric | Naive GEMM Kernel | Shared Memory Tiled GEMM | Impact / Architectural Takeaway |
| :--- | :--- | :--- | :--- |
| **Duration / Elapsed Cycles** | 52.35 ms (31,409,459 cyc) | **40.97 ms (24,579,230 cyc)** | **21.7% cycle reduction** via data reuse |
| **Registers per Thread** | 49 regs/thread | **40 regs/thread** | Reduced register pressure |
| **Static Shared Memory** | 0 KB/block | **8.19 KB/block** | Staged in on-chip SRAM |
| **Achieved Occupancy** | 98.80% (31.62 warps/SM) | **99.97% (31.99 warps/SM)** | Reaches theoretical limit (32 warps/SM) |
| **Compute / Memory State** | Memory Bound (85.5% unit util) | **Balanced Compute & Memory** | NCU confirms balanced throughput |

---

## 4. Repository Structure

```text
├── gemm.cu               # Benchmark runner & CPU numerical correctness check
├── profile_runner.cu     # Fixed-dimension harness for NCU profiling
├── profile_t4.ncu-rep    # Raw NVIDIA Nsight Compute profile report artifact
└── README.md
