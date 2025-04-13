#include <cudnn.h>
#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include <cmath>

// CPU convolution for verification
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

bool verifyResults(float *gpu_result, float *cpu_result, int M, int N, float tolerance = 1e-5) {
    for (int i = 0; i < M * N; i++) {
        if (std::fabs(gpu_result[i] - cpu_result[i]) > tolerance) {
            std::cout << "Mismatch at position " << i 
                      << ": GPU = " << gpu_result[i] 
                      << ", CPU = " << cpu_result[i] 
                      << std::endl;
            return false;
        }
    }
    return true;
}

int main() {
    // Matrix dimensions.
    const int M = 100, N = 100, K = 5; // Input: 100x100, Kernel: 5x5.
    const int input_height = M;
    const int input_width = N;
    const int channels = 1;       // Single channel.
    const int batch_size = 1;     // Single image.

    // Allocate and initialize host memory.
    float *h_A = new float[M * N];
    float *h_K = new float[K * K];
    float *h_C = new float[M * N];
    float *cpu_C = new float[M * N];

    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dis(0.0, 1.0);

    for (int i = 0; i < M * N; ++i) {
        h_A[i] = dis(gen);
    }
    for (int i = 0; i < K * K; ++i) {
        h_K[i] = dis(gen);
    }

    // Allocate device memory.
    float *d_A, *d_K, *d_C;
    size_t bytes_A = M * N * sizeof(float);
    size_t bytes_K = K * K * sizeof(float);
    size_t bytes_C = M * N * sizeof(float);

    cudaMalloc(&d_A, bytes_A);
    cudaMalloc(&d_K, bytes_K);
    cudaMalloc(&d_C, bytes_C);

    // Copy input data to the device.
    cudaMemcpy(d_A, h_A, bytes_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K, bytes_K, cudaMemcpyHostToDevice);

    // Create the cuDNN handle.
    cudnnHandle_t cudnn;
    cudnnCreate(&cudnn);

    // Create and set the input tensor descriptor. Shape: [batch_size, channels, height, width].
    cudnnTensorDescriptor_t inputDesc;
    cudnnCreateTensorDescriptor(&inputDesc);
    cudnnSetTensor4dDescriptor(
        inputDesc,
        CUDNN_TENSOR_NCHW,      // Format
        CUDNN_DATA_FLOAT,       // Data type
        batch_size, channels, input_height, input_width);

    // Create and set the filter descriptor. Shape: [out_channels, in_channels, height, width].
    cudnnFilterDescriptor_t filterDesc;
    cudnnCreateFilterDescriptor(&filterDesc);
    cudnnSetFilter4dDescriptor(
        filterDesc,
        CUDNN_DATA_FLOAT,       
        CUDNN_TENSOR_NCHW,
        1,  // out_channels (single filter)
        1,  // in_channels
        K, K);

    // Create and set the convolution descriptor.
    cudnnConvolutionDescriptor_t convDesc;
    cudnnCreateConvolutionDescriptor(&convDesc);
    int pad_h = K / 2; // for K=5, pad_h = 2
    int pad_w = K / 2;
    int stride_h = 1;
    int stride_w = 1;
    int dilation_h = 1;
    int dilation_w = 1;
    cudnnSetConvolution2dDescriptor(
        convDesc,
        pad_h, pad_w,
        stride_h, stride_w,
        dilation_h, dilation_w,
        CUDNN_CROSS_CORRELATION, // Use cross-correlation (default for cuDNN)
        CUDNN_DATA_FLOAT);

    // Get the output dimensions.
    int out_n, out_c, out_h, out_w;
    cudnnGetConvolution2dForwardOutputDim(
        convDesc,
        inputDesc,
        filterDesc,
        &out_n, &out_c, &out_h, &out_w);

    // Create and set the output tensor descriptor.
    cudnnTensorDescriptor_t outputDesc;
    cudnnCreateTensorDescriptor(&outputDesc);
    cudnnSetTensor4dDescriptor(
        outputDesc,
        CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT,
        out_n, out_c, out_h, out_w);

    // Choose the forward convolution algorithm.
    cudnnConvolutionFwdAlgo_t algo;
    cudnnGetConvolutionForwardAlgorithm(
        cudnn,
        inputDesc,
        filterDesc,
        convDesc,
        outputDesc,
        CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
        0,
        &algo);

    // Get the workspace size required.
    size_t workspaceSize = 0;
    cudnnGetConvolutionForwardWorkspaceSize(
        cudnn,
        inputDesc,
        filterDesc,
        convDesc,
        outputDesc,
        algo,
        &workspaceSize);

    // Allocate workspace memory if needed.
    void* d_workspace = nullptr;
    if (workspaceSize > 0) {
        cudaMalloc(&d_workspace, workspaceSize);
    }

    // Run cuDNN convolution.
    float alpha = 1.0f, beta = 0.0f;
    cudnnConvolutionForward(
        cudnn,
        &alpha,
        inputDesc,
        d_A,
        filterDesc,
        d_K,
        convDesc,
        algo,
        d_workspace,
        workspaceSize,
        &beta,
        outputDesc,
        d_C);

    // Copy the result from device to host.
    cudaMemcpy(h_C, d_C, bytes_C, cudaMemcpyDeviceToHost);

    // Compute the reference result on the CPU.
    cpuConv(h_A, h_K, cpu_C, M, N, K);

    // Verify the results.
    bool passed = verifyResults(h_C, cpu_C, M, N);
    if (passed) {
        std::cout << "Verification PASSED!" << std::endl;
    } else {
        std::cout << "Verification FAILED!" << std::endl;
    }

    // Clean up allocated resources.
    if (d_workspace) {
        cudaFree(d_workspace);
    }
    cudaFree(d_A);
    cudaFree(d_K);
    cudaFree(d_C);
    cudnnDestroyTensorDescriptor(inputDesc);
    cudnnDestroyTensorDescriptor(outputDesc);
    cudnnDestroyFilterDescriptor(filterDesc);
    cudnnDestroyConvolutionDescriptor(convDesc);
    cudnnDestroy(cudnn);
    
    delete[] h_A;
    delete[] h_K;
    delete[] h_C;
    delete[] cpu_C;

    return 0;
}