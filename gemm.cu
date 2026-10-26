#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <cuda_runtime.h>

#define TILE_SIZE 32

// 1. CPU Reference Implementation for validation
void cpu_gemm(const float* A, const float* B, float* C, int N) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < N; ++k) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// 2. Naive CUDA Kernel (Global Memory Bound)
__global__ void naive_gemm(const float* __restrict__ A, 
                           const float* __restrict__ B, 
                           float* __restrict__ C, 
                           int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; ++k) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// 3. Shared Memory Tiled CUDA Kernel
__global__ void tiled_gemm(const float* __restrict__ A, 
                           const float* __restrict__ B, 
                           float* __restrict__ C, 
                           int N) {
    __shared__ float tile_A[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_B[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float acc = 0.0f;
    int num_tiles = (N + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < num_tiles; ++t) {
        int a_col = t * TILE_SIZE + threadIdx.x;
        int b_row = t * TILE_SIZE + threadIdx.y;

        tile_A[threadIdx.y][threadIdx.x] = (row < N && a_col < N) ? A[row * N + a_col] : 0.0f;
        tile_B[threadIdx.y][threadIdx.x] = (b_row < N && col < N) ? B[b_row * N + col] : 0.0f;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            acc += tile_A[threadIdx.y][k] * tile_B[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < N && col < N) {
        C[row * N + col] = acc;
    }
}

bool verify_result(const float* ref, const float* target, int N, float eps = 1e-2f) {
    for (int i = 0; i < N * N; ++i) {
        if (std::fabs(ref[i] - target[i]) > eps) {
            std::cerr << "Mismatch at index " << i << ": ref=" << ref[i] << ", got=" << target[i] << std::endl;
            return false;
        }
    }
    return true;
}

void run_benchmark(int N) {
    size_t bytes = N * N * sizeof(float);

    std::vector<float> h_A(N * N);
    std::vector<float> h_B(N * N);
    std::vector<float> h_C_naive(N * N);
    std::vector<float> h_C_tiled(N * N);

    for (int i = 0; i < N * N; ++i) {
        h_A[i] = static_cast<float>(rand()) / RAND_MAX;
        h_B[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    cudaMemcpy(d_A, h_A.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), bytes, cudaMemcpyHostToDevice);

    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Warmup
    naive_gemm<<<blocks, threads>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();

    // 1. Benchmark Naive
    cudaEventRecord(start);
    int iterations = (N <= 1024) ? 10 : 3;
    for (int i = 0; i < iterations; ++i) {
        naive_gemm<<<blocks, threads>>>(d_A, d_B, d_C, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float naive_ms = 0.0f;
    cudaEventElapsedTime(&naive_ms, start, stop);
    naive_ms /= iterations;
    cudaMemcpy(h_C_naive.data(), d_C, bytes, cudaMemcpyDeviceToHost);

    // 2. Benchmark Tiled
    cudaEventRecord(start);
    for (int i = 0; i < iterations; ++i) {
        tiled_gemm<<<blocks, threads>>>(d_A, d_B, d_C, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float tiled_ms = 0.0f;
    cudaEventElapsedTime(&tiled_ms, start, stop);
    tiled_ms /= iterations;
    cudaMemcpy(h_C_tiled.data(), d_C, bytes, cudaMemcpyDeviceToHost);

    // Validate correctness
    bool validated = true;
    if (N <= 512) {
        std::vector<float> h_C_cpu(N * N);
        cpu_gemm(h_A.data(), h_B.data(), h_C_cpu.data(), N);
        validated = verify_result(h_C_cpu.data(), h_C_tiled.data(), N);
    } else {
        validated = verify_result(h_C_naive.data(), h_C_tiled.data(), N);
    }

    float speedup = naive_ms / tiled_ms;
    double gflops = (2.0 * std::pow(N, 3) * 1e-9) / (tiled_ms * 1e-3);

    std::cout << "| " << std::setw(9) << N 
              << " | " << std::setw(15) << std::fixed << std::setprecision(2) << naive_ms
              << " | " << std::setw(15) << std::fixed << std::setprecision(2) << tiled_ms
              << " | " << std::setw(11) << std::fixed << std::setprecision(2) << speedup << "x"
              << " | " << std::setw(13) << std::fixed << std::setprecision(2) << gflops
              << " | " << std::setw(11) << (validated ? "PASS" : "FAIL") << " |" << std::endl;

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main() {
    std::cout << "| Matrix Dim | Naive Latency (ms) | Tiled Latency (ms) | Speedup    | Tiled GFLOPS  | Correctness |" << std::endl;
    std::cout << "|:-----------|:-------------------|:-------------------|:-----------|:--------------|:------------|" << std::endl;

    std::vector<int> sizes = {256, 512, 1024, 2048, 4096};
    for (int n : sizes) {
        run_benchmark(n);
    }
    return 0;
}
