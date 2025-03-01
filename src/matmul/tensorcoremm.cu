#include <cuda_fp16.h>
#include <mma.h>
#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include <cmath>

using namespace nvcuda;

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

// Each block (with one warp of 32 threads) computes one 16x16 tile of C using tensor cores.
__global__ void wmmaGemm(const __half *A, const __half *B, float *C, int M, int N, int K) {
    // Each block is one warp: grid dimensions determine the tile location.
    int warpM = blockIdx.y;  
    int warpN = blockIdx.x;  

    // Declare WMMA fragments for a tile of A, B, and the accumulator tile of C.
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> aFrag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::col_major> bFrag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> cFrag;

    // Initialize the output fragment to zero.
    wmma::fill_fragment(cFrag, 0.0f);

    // Loop over the K dimension in steps of WMMA_K.
    for (int i = 0; i < N; i += WMMA_K) {
        // Compute pointers to the current tile in A and B.
        int aRow = warpM * WMMA_M;
        int aCol = i;
        int bRow = i;
        int bCol = warpN * WMMA_N;

        const __half *tileA = A + aRow * N + aCol;
        const __half *tileB = B + bRow * K + bCol;

        // Load the A and B tiles into WMMA fragments.
        wmma::load_matrix_sync(aFrag, tileA, N);
        wmma::load_matrix_sync(bFrag, tileB, K);

        // Multiply and accumulate the fragments.
        wmma::mma_sync(cFrag, aFrag, bFrag, cFrag);
    }

    // Compute the pointer to the output tile in C.
    int cRow = warpM * WMMA_M;
    int cCol = warpN * WMMA_N;
    float *tileC = C + cRow * K + cCol;
    // Store the computed tile back to global memory.
    wmma::store_matrix_sync(tileC, cFrag, K, wmma::mem_row_major);
}

// CPU matrix multiplication for verification (converts __half to float)
void cpuGemm(const __half *A, const __half *B, float *C, int M, int N, int K) {
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < K; n++) {
            float sum = 0.0f;
            for (int k = 0; k < N; k++) {
                float a = __half2float(A[m * N + k]);
                float b = __half2float(B[k * K + n]);
                sum += a * b;
            }
            C[m * K + n] = sum;
        }
    }
}

bool verifyResults(const float *gpu, const float *cpu, int M, int K, float tol = 1e-3f) {
    for (int i = 0; i < M * K; i++) {
        if (std::fabs(gpu[i] - cpu[i]) > tol) {
            std::cout << "Mismatch at index " << i << ": GPU = " << gpu[i] << ", CPU = " << cpu[i] << "\n";
            return false;
        }
    }
    return true;
}

int main() {
    // Matrix dimensions (must be multiples of 16)
    int M = 128, N = 128, K = 128;
    size_t sizeA = M * N;
    size_t sizeB = N * K;
    size_t sizeC = M * K;

    // Allocate host memory for matrices A and B (in half precision) and C (in float)
    __half *h_A = new __half[sizeA];
    __half *h_B = new __half[sizeB];
    float *h_C = new float[sizeC];
    float *h_C_cpu = new float[sizeC];

    // Initialize matrices with random numbers and convert to half precision.
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);
    for (size_t i = 0; i < sizeA; i++) {
        float val = dis(gen);
        h_A[i] = __float2half(val);
    }
    for (size_t i = 0; i < sizeB; i++) {
        float val = dis(gen);
        h_B[i] = __float2half(val);
    }

    // Allocate device memory.
    __half *d_A, *d_B;
    float *d_C;
    cudaMalloc((void**)&d_A, sizeA * sizeof(__half));
    cudaMalloc((void**)&d_B, sizeB * sizeof(__half));
    cudaMalloc((void**)&d_C, sizeC * sizeof(float));

    // Copy data from host to device.
    cudaMemcpy(d_A, h_A, sizeA * sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeB * sizeof(__half), cudaMemcpyHostToDevice);

    // Launch the WMMA kernel.
    // Grid dimensions: one block per output tile. For M=128 and K=128, we have (128/16)x(128/16) = 8x8 blocks.
    dim3 gridDim(K / WMMA_N, M / WMMA_M);
    // One warp (32 threads) per block.
    dim3 blockDim(32, 1, 1);
    wmmaGemm<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
    cudaDeviceSynchronize();

    // Copy the result from device to host.
    cudaMemcpy(h_C, d_C, sizeC * sizeof(float), cudaMemcpyDeviceToHost);

    // Compute CPU result for verification.
    cpuGemm(h_A, h_B, h_C_cpu, M, N, K);

    // Verify the GPU result.
    if (verifyResults(h_C, h_C_cpu, M, K)) {
        std::cout << "Verification PASSED!\n";
    } else {
        std::cout << "Verification FAILED!\n";
    }

    // Free device and host memory.
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    delete[] h_C_cpu;

    return 0;
}