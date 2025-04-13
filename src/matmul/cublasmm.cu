#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>
#include <random>
#include <cmath>

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
    
    // Create cuBLAS handle
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Perform matrix multiplication using cuBLAS
    // cublasSgemm performs C = alpha * op(A) * op(B) + beta * C
    // where op(X) = X or X^T
    float alpha = 1.0f;
    float beta = 0.0f;
    
    // Note: cuBLAS assumes column-major order, so we need to transpose the operation
    // to match our row-major data layout
    cublasSgemm(handle,
                CUBLAS_OP_N, CUBLAS_OP_N,  // No transpose for A and B
                K, M, N,                   // Dimensions
                &alpha,                    // alpha
                d_B, K,                    // B matrix, leading dimension K
                d_A, N,                    // A matrix, leading dimension N
                &beta,                     // beta
                d_C, K);                   // C matrix, leading dimension K

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
    cublasDestroy(handle);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    delete[] cpu_C;
    
    return 0;
}