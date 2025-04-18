#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include <cmath>

// Kernel code
// Generalise such that we can accept non-square matrices 
// Assume A=MxN and B=NxK -> C=MxK
// Work on tiles to reduce global memory access
// Idea is simple, fetch tile once and store in shared memory
// Use values in shared memory to compute partial product and accumulate over tiles
// Similar to FlashAttention idea

__global__ void tiledMatMul(float *A, float *B, float *C, int M, int N, int K) {
    
    const int TILE_SIZE = 16;
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];

    int tx = threadIdx.x, ty = threadIdx.y;
    
    int col = blockIdx.x * TILE_SIZE + tx; 
    int row = blockIdx.y * TILE_SIZE + ty;
    
    float acc = 0.0f;
    // Loop over tiles
    for (int t = 0; t < (N + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // First fetch elements from global memory and sync
        // Row-major order
        if (row < M && (t * TILE_SIZE + tx) < N)
            tileA[ty][tx] = A[row * N + t * TILE_SIZE + tx];
        else
            tileA[ty][tx] = 0.0f;

        // Load B into shared memory if within bounds
        if (col < K && (t * TILE_SIZE + ty) < N)
            tileB[ty][tx] = B[(t * TILE_SIZE + ty) * K + col];
        else
            tileB[ty][tx] = 0.0f;
        __syncthreads();
        
        // Perform local multiplication and sync threads
        for (int i = 0; i < TILE_SIZE; i++) {
            acc += tileA[ty][i] * tileB[i][tx];
        }
        __syncthreads();

    }
    // Write the result to global memory if within bounds
    if (row < M && col < K) {
        C[row * K + col] = acc;
    }
}
// CPU matrix multiplication for verification
void cpuMatMul(float *A, float *B, float *C, int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K; j++) {
            float acc = 0.0f;
            for (int k = 0; k < N; k++) {
                acc += A[i * N + k] * B[k * K + j];
            }
            C[i * K + j] = acc;
        }
    }
}

// Verification function
bool verifyResults(float *gpu_result, float *cpu_result, int M, int K, float tolerance = 1e-5) {
    for (int i = 0; i < M * K; i++) {
        if (std::fabs(gpu_result[i] - cpu_result[i]) > tolerance) {
            std::cout << "Mismatch at position " << i << ": GPU = " << gpu_result[i] 
                      << ", CPU = " << cpu_result[i] << std::endl;
            return false;
        }
    }
    return true;
}

int main() {
    // Create data on host
    // Create two matrices and fill with random floats [0.0, 1.0)
    const int M = 100, N = 100, K = 100; 
    float *h_A = new float[M * N];
    float *h_B = new float[N * K];
    float *h_C = new float[M * K];
    float *cpu_C = new float[M * K];  // For CPU verification

    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dis(0.0, 1.0); 
    
    for (int i = 0; i < M * N; ++i) {
        h_A[i] = dis(gen);
    }
    for (int i = 0; i < N * K; ++i) {
        h_B[i] = dis(gen);
    }

    // Calculate sizes for memory allocation
    size_t bytes_A = M * N * sizeof(float);
    size_t bytes_B = N * K * sizeof(float);
    size_t bytes_C = M * K * sizeof(float);

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, bytes_A);
    cudaMalloc((void **)&d_B, bytes_B);
    cudaMalloc((void **)&d_C, bytes_C);

    // Copy matrices from host to device
    cudaMemcpy(d_A, h_A, bytes_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes_B, cudaMemcpyHostToDevice);
    
    // Launch kernel
    const int blockDim = 16;
    dim3 blockSize(blockDim, blockDim);
    dim3 gridSize((K + blockDim - 1) / blockDim, (M + blockDim - 1) / blockDim);
    tiledMatMul<<<gridSize, blockSize>>>(d_A, d_B, d_C, M, N, K);

    // Return result from device to host
    cudaMemcpy(h_C, d_C, bytes_C, cudaMemcpyDeviceToHost);

    // Compute CPU result
    cpuMatMul(h_A, h_B, cpu_C, M, N, K);

    // Verify results
    bool passed = verifyResults(h_C, cpu_C, M, K);
    if (passed) {
        std::cout << "Verification PASSED!" << std::endl;
    } else {
        std::cout << "Verification FAILED!" << std::endl;
    }

    // Clean up
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    delete[] cpu_C;
    
    return 0;
}
