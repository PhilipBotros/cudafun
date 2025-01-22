#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <iostream>
#include <random>
#include <cmath>

namespace cg = cooperative_groups;

__global__ void softmax1D(float *in, float *out, int dim){
    cg::grid_group grid = cg::this_grid();
    extern __shared__ float smem[];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tx = threadIdx.x;
    if (idx >= dim) return;

    float e_x = exp(in[idx]);
    smem[tx] = e_x;
    __syncthreads();
    
    // Sum elements in shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tx < stride) {
            smem[tx] += smem[tx + stride];
        }
        __syncthreads(); 
    }
    // Local thread 0 writes the sum of this block to HBM
    if (tx == 0) {
        out[blockIdx.x] = smem[0];
    }

    // Sync the grid to ensure that block sums have been written to HBM
    grid.sync();

    // Global thread 0 now sums the block sums
    if (grid.thread_rank() == 0) {
        float total_sum = 0.0f;
        for (int i = 0; i < gridDim.x; ++i) {
            total_sum += out[i];
        }
        out[0] = total_sum; 
    }
    
    // Sync sum and calculate softmax value for out[idx]
    grid.sync();
    out[idx] = e_x / out[0];
}

int main() {
    
    int dim = 100;
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dis(0.0, 1.0); 
    
    float *h_in = new float[dim];
    size_t bytes_in = dim * sizeof(float);
    size_t bytes_out = dim * sizeof(float);

    for (int i = 0; i < dim; ++i) {
        h_in[i] = dis(gen);
    }
    float *h_out = new float[dim];
    
    float *d_in, *d_out;
    cudaMalloc((void **)&d_in, bytes_in);
    cudaMalloc((void **)&d_out, bytes_out);

    cudaMemcpy(d_in, h_in, bytes_in, cudaMemcpyHostToDevice);
    int blockDim = 16;
    int gridDim = (dim + blockDim - 1) / blockDim;

    // Ensure cooperative groups support
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    if (!deviceProp.cooperativeLaunch) {
        std::cerr << "Device does not support cooperative launch!" << std::endl;
        return -1;
    }

    void *kernel_args[] = {&d_in, &d_out, &dim};
    cudaLaunchCooperativeKernel(
        (void *)softmax1D,
        gridDim,
        blockDim,
        kernel_args,
        blockDim * sizeof(float)
    );
    cudaMemcpy(h_out, d_out, bytes_out, cudaMemcpyDeviceToHost);
    
    std::cout << "Input:\n";
    for (int i = 0; i < dim; ++i) {
        std::cout << h_in[i] << " ";
    }
    std::cout << "\n\nSoftmax Output:\n";
    for (int i = 0; i < dim; ++i) {
        std::cout << h_out[i] << " ";
    }
    std::cout << std::endl;

    delete[] h_in;
    delete[] h_out;
    cudaFree(d_in);
    cudaFree(d_out);

    return 0;
}