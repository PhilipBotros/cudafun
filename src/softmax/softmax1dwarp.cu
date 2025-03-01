#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <iostream>
#include <random>
#include <cmath>

namespace cg = cooperative_groups;

__global__ void softmax1D(float *in, float *out, int dim) {
    cg::grid_group grid = cg::this_grid();
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Compute exponential only for valid indices.
    float e_x = 0.0f;
    if (idx < dim) {
        e_x = expf(in[idx]);
    }
    
    // Each thread starts with its own value.
    float sum = e_x;
    // Use full mask for active threads.
    unsigned int mask = 0xffffffff;
    // Warp-level reduction using shuffles.
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(mask, sum, offset);
    }
    
    // Each warp's first thread (lane 0) stores the partial sum.
    __shared__ float blockSums[32];
    int warpId = threadIdx.x / warpSize;
    if ((threadIdx.x & (warpSize - 1)) == 0) {
        blockSums[warpId] = sum;
    }
    __syncthreads();
    
    // Let the first warp reduce the partial warp sums.
    float blockSum = 0.0f;
    int numWarps = (blockDim.x + warpSize - 1) / warpSize;
    if (threadIdx.x < numWarps) {
        blockSum = blockSums[threadIdx.x];
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            blockSum += __shfl_down_sync(mask, blockSum, offset);
        }
        // First thread of the block writes the block sum.
        if (threadIdx.x == 0) {
            out[blockIdx.x] = blockSum;
        }
    }
    
    // Grid-wide sync: all blocks have written their sums.
    grid.sync();
    
    // Global reduction: thread 0 of the entire grid sums all block sums.
    if (grid.thread_rank() == 0) {
        float total_sum = 0.0f;
        for (int i = 0; i < gridDim.x; ++i) {
            total_sum += out[i];
        }
        // Save the total sum to out[0].
        out[0] = total_sum;
    }
    
    // Ensure that the total sum is available to all threads.
    grid.sync();
    
    // Compute the final softmax value for valid indices.
    if (idx < dim) {
        out[idx] = e_x / out[0];
    }
}

int main() {
    int dim = 100;
    size_t bytes_in = dim * sizeof(float);
    size_t bytes_out = dim * sizeof(float);
    
    // Allocate and initialize host memory.
    float *h_in = new float[dim];
    float *h_out = new float[dim];
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);
    for (int i = 0; i < dim; ++i) {
        h_in[i] = dis(gen);
    }
    
    // Allocate device memory.
    float *d_in, *d_out;
    cudaMalloc((void**)&d_in, bytes_in);
    cudaMalloc((void**)&d_out, bytes_out);
    
    cudaMemcpy(d_in, h_in, bytes_in, cudaMemcpyHostToDevice);
    
    // Set block and grid dimensions.
    int blockDim = 16;
    int gridDim = (dim + blockDim - 1) / blockDim;
    
    // Check for cooperative launch support.
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    if (!deviceProp.cooperativeLaunch) {
        std::cerr << "Device does not support cooperative launch!" << std::endl;
        cudaFree(d_in);
        cudaFree(d_out);
        delete[] h_in;
        delete[] h_out;
        return -1;
    }
    
    // Set up kernel arguments.
    void *kernelArgs[] = { &d_in, &d_out, &dim };
    
    // Launch the kernel cooperatively. No extra shared memory size is needed here.
    cudaLaunchCooperativeKernel(
        (void*)softmax1D,
        gridDim,
        blockDim,
        kernelArgs
    );
    
    // Copy the results back to host.
    cudaMemcpy(h_out, d_out, bytes_out, cudaMemcpyDeviceToHost);
    
    // Output the input and softmax result.
    std::cout << "Input:\n";
    for (int i = 0; i < dim; ++i) {
        std::cout << h_in[i] << " ";
    }
    std::cout << "\n\nSoftmax Output:\n";
    for (int i = 0; i < dim; ++i) {
        std::cout << h_out[i] << " ";
    }
    std::cout << std::endl;
    
    // Clean up.
    cudaFree(d_in);
    cudaFree(d_out);
    delete[] h_in;
    delete[] h_out;
    
    return 0;
}