#include <cuda_runtime.h>

__global__ void add_arrays(const float* a, const float* b, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        out[idx] = a[idx] + b[idx];
}

void launch_sum_kernel(const float* a, const float* b, float* out, int n, cudaStream_t stream) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    add_arrays<<<blocks, threads, 0, stream>>>(a, b, out, n);
}