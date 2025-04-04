extern "C" {
#include <cuda_runtime.h>

__global__ void add_arrays(const float* a, const float* b, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        out[idx] = a[idx] + b[idx];
}

void launch_sum_kernel(const float* a, const float* b, float* out, int n) {
    float *d_a, *d_b, *d_out;
    size_t size = n * sizeof(float);

    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_out, size);

    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    add_arrays<<<blocks, threads>>>(d_a, d_b, d_out, n);

    cudaMemcpy(out, d_out, size, cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);
}
}