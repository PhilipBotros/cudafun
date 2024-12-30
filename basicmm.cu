#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include <cmath>

// Kernel code
// Generalise such that we can accept non-square matrices 
// Assume A=MxN and B=NxK -> C=MxK
__global__ void matMul(float *A, float *B, float *C, int M, int N, int K) {
    int col = blockIdx.x * blockDim.x + threadIdx.x; 
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col < K && row < M) {
        float acc = 0.0f;
        // Row-major order flattened array
        for (int i = 0; i < N; i++) {
            acc += A[row * N + i] * B[col + K * i];
        }
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
    matMul<<<gridSize, blockSize>>>(d_A, d_B, d_C, M, N, K);

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