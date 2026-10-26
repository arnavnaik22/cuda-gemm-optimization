#include <cuda_runtime.h>
#define TILE_SIZE 32

__global__ void naive_gemm(const float* A, const float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; ++k) sum += A[row * N + k] * B[k * N + col];
        C[row * N + col] = sum;
    }
}

__global__ void tiled_gemm(const float* A, const float* B, float* C, int N) {
    __shared__ float sA[TILE_SIZE][TILE_SIZE];
    __shared__ float sB[TILE_SIZE][TILE_SIZE];
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    float sum = 0.0f;
    for (int t = 0; t < (N + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        sA[threadIdx.y][threadIdx.x] = (row < N && t * TILE_SIZE + threadIdx.x < N) ? A[row * N + t * TILE_SIZE + threadIdx.x] : 0.0f;
        sB[threadIdx.y][threadIdx.x] = (t * TILE_SIZE + threadIdx.y < N && col < N) ? B[(t * TILE_SIZE + threadIdx.y) * N + col] : 0.0f;
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) sum += sA[threadIdx.y][k] * sB[k][threadIdx.x];
        __syncthreads();
    }
    if (row < N && col < N) C[row * N + col] = sum;
}

int main() {
    int N = 2048;
    size_t sz = N * N * sizeof(float);
    float *dA, *dB, *dC;
    cudaMalloc(&dA, sz); cudaMalloc(&dB, sz); cudaMalloc(&dC, sz);
    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);
    
    // Warmup & execution
    naive_gemm<<<blocks, threads>>>(dA, dB, dC, N);
    tiled_gemm<<<blocks, threads>>>(dA, dB, dC, N);
    cudaDeviceSynchronize();
    
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    return 0;
}
