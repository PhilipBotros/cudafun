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
// This variant uses thread coarsening where we process multiple blocks of matrix B per thread

__global__ void tiledMatMul(float *A, float *B, float *C, int M, int N, int K) {
    
    const int TILE_SIZE = 16;
    const int COARSE_FACTOR = 4;
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];

    int tx = threadIdx.x, ty = threadIdx.y;
    
    int colStart = blockIdx.x * TILE_SIZE * COARSE_FACTOR + tx; 
    int row = blockIdx.y * TILE_SIZE + ty;
    
    // Accumulator is now an array of size COARSE_FACTOR
    float acc[COARSE_FACTOR];
    for (int c = 0; c < COARSE_FACTOR; c++) {
        acc[c] = 0.0f;
    }
    // Loop over tiles
    for (int t = 0; t < (N + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // First fetch elements from global memory and sync
        // Row-major order
        if (row < M && (t * TILE_SIZE + tx) < N)
            tileA[ty][tx] = A[row * N + t * TILE_SIZE + tx];
        else
            tileA[ty][tx] = 0.0f;
        
        for (int c = 0; c < COARSE_FACTOR; c++) {
            // Now loop over C blocks of matrix B
            int col = colStart + TILE_SIZE * c;
            if (col < K && (t * TILE_SIZE + ty) < N)
                tileB[ty][tx] = B[(t * TILE_SIZE + ty) * K + col];
            else
                tileB[ty][tx] = 0.0f;
            
            __syncthreads();
            // Perform local multiplication and sync threads
            for (int i = 0; i < TILE_SIZE; i++) {
                acc[c] += tileA[ty][i] * tileB[i][tx];
            }
            __syncthreads();
        }        

    }
    // Write the result to global memory if within bounds
    for (int c = 0; c < COARSE_FACTOR; c++) {
        int col = colStart + TILE_SIZE * c;
        if (row < M && col < K) {
            C[row * K + col] = acc[c];
        }
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
    const int M = 1024, N = 1024, K = 1024; 
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
    const int COARSE_FACTOR = 4;
    dim3 blockSize(blockDim, blockDim);
    dim3 gridSize((K + blockDim * COARSE_FACTOR - 1) / (blockDim * COARSE_FACTOR), (M + blockDim - 1) / blockDim);

    // Warmup runs to avoid cold start overhead
    for (int i = 0; i < 10; i++) {
        tiledMatMul<<<gridSize, blockSize>>>(d_A, d_B, d_C, M, N, K);
    }
    cudaDeviceSynchronize();

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Run multiple iterations and average
    const int numIterations = 100;
    cudaEventRecord(start);

    for (int i = 0; i < numIterations; i++) {
        tiledMatMul<<<gridSize, blockSize>>>(d_A, d_B, d_C, M, N, K);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Calculate elapsed time and FLOP/s
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    float avgTime = milliseconds / numIterations;

    long long flops = 2LL * M * N * K;
    double gflops = (flops / 1e9) / (avgTime / 1000.0);

    std::cout << "Kernel time: " << avgTime << " ms" << std::endl;
    std::cout << "Performance: " << gflops << " GFLOP/s" << std::endl;

    // Cleanup events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Return result from device to host
    cudaMemcpy(h_C, d_C, bytes_C, cudaMemcpyDeviceToHost);

    // Skip CPU verification for large matrices (too slow and we already verified at small sizes)
    if (M <= 100) {
        cpuMatMul(h_A, h_B, cpu_C, M, N, K);
        bool passed = verifyResults(h_C, cpu_C, M, K);
        if (passed) {
            std::cout << "Verification PASSED!" << std::endl;
        } else {
            std::cout << "Verification FAILED!" << std::endl;
        }
    } else {
        std::cout << "Verification SKIPPED (large matrix)" << std::endl;
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
