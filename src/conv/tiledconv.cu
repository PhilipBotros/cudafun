#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include <cmath>

// Define constant memory for conv kernel
#define KERNEL_SIZE 5
__constant__ float d_K[KERNEL_SIZE][KERNEL_SIZE];

// Kernel code
// A is input matrix, K is conv kernel, C is output matrix
// Assume A=MxN and B=KxK -> C=MxN
// Using shared memory to load input blocks
__global__ void tiledConv(float *d_A, float *d_C, int M, int N, int K, int outDim) {
    int R = K / 2;

    // Index based on the output tile to simplify code
    int col = blockIdx.x * outDim + threadIdx.x - R; 
    int row = blockIdx.y * outDim + threadIdx.y - R;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Declare shared memory
    const int TILE_SIZE = 32;
    __shared__ float tileIn[TILE_SIZE][TILE_SIZE];

    // Load input data into shared memory with bounds checking
    if (col >= 0 && col < N && row >= 0 && row < M) {
        tileIn[ty][tx] = d_A[row * N + col];
    } else {
        tileIn[ty][tx] = 0.0f;
    }

    __syncthreads();

    // Compute convolution only for valid output elements
    if (tx >= R && tx < outDim + R && ty >= R && ty < outDim + R) {
        float acc = 0.0f;
        for (int i = -R; i <= R; i++) {
            for (int j = -R; j <= R; j++) {
                int sharedRow = ty + i;
                int sharedCol = tx + j;
                acc += tileIn[sharedRow][sharedCol] * d_K[i + R][j + R];
            }
        }
        int outputRow = row + R;
        int outputCol = col + R;
        if (outputRow < M && outputCol < N) {
            d_C[outputRow * N + outputCol] = acc;
        }
    }
}

// CPU conv for verification
void cpuConv(float *A, float *K, float *C, int M, int N, int kernel_size) {
    int start_idx = -kernel_size / 2;
    int end_idx = kernel_size / 2;

    for (int row = 0; row < M; row++) {
        for (int col = 0; col < N; col++) {
            float acc = 0.0f;
            for (int i = start_idx; i <= end_idx; i++) {
                for (int j = start_idx; j <= end_idx; j++) {
                    int A_row = row + i; 
                    int A_col = col + j; 

                    if (A_row >= 0 && A_row < M && A_col >= 0 && A_col < N) {
                        acc += A[A_row * N + A_col] * K[(i + end_idx) * kernel_size + (j + end_idx)];
                    }
                }
            }
            C[row * N + col] = acc;
        }
    }
}

// Verification function
bool verifyResults(float *gpu_result, float *cpu_result, int M, int N, float tolerance = 1e-5) {
    for (int i = 0; i < M * N; i++) {
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
    const int M = 100, N = 100, K = 5; 
    float *h_A = new float[M * N];
    float *h_K = new float[K * K];
    float *h_C = new float[M * N];
    float *cpu_C = new float[M * N];  // For CPU verification

    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dis(0.0, 1.0); 
    
    for (int i = 0; i < M * N; ++i) {
        h_A[i] = dis(gen);
    }
    for (int i = 0; i < K * K; ++i) {
        h_K[i] = dis(gen);
    }

    // Calculate sizes for memory allocation
    size_t bytes_A = M * N * sizeof(float);
    size_t bytes_K = K * K * sizeof(float);
    size_t bytes_C = M * N * sizeof(float);

    // Allocate device memory
    float *d_A, *d_C;
    cudaMalloc((void **)&d_A, bytes_A);
    cudaMalloc((void **)&d_C, bytes_C);

    // Copy matrices from host to device
    cudaMemcpy(d_A, h_A, bytes_A, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(d_K, h_K, bytes_K);
    
    // Launch kernel
    const int blockDim = 32;
    const int outDim = blockDim - 2 * KERNEL_SIZE / 2;
    dim3 blockSize(blockDim, blockDim);
    dim3 gridSize((N + outDim - 1) / outDim, (M + outDim - 1) / outDim);
    tiledConv<<<gridSize, blockSize>>>(d_A, d_C, M, N, K, outDim);

    // Return result from device to host
    cudaMemcpy(h_C, d_C, bytes_C, cudaMemcpyDeviceToHost);

    // Compute CPU result
    cpuConv(h_A, h_K, cpu_C, M, N, K);

    // Verify results
    bool passed = verifyResults(h_C, cpu_C, M, N);
    if (passed) {
        std::cout << "Verification PASSED!" << std::endl;
    } else {
        std::cout << "Verification FAILED!" << std::endl;
    }

    // Clean up
    cudaFree(d_A);
    cudaFree(d_C);
    delete[] h_A;
    delete[] h_K;
    delete[] h_C;
    delete[] cpu_C;
    
    return 0;
}