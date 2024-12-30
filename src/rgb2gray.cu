#include <cuda_runtime.h>
#include <iostream>
#include <opencv2/opencv.hpp>

// Kernel code
__global__ void rgb2Gray(unsigned char *rgbImage, unsigned char *grayImage, int rows, int cols) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    if (x < cols && y < rows) {
        // Row-major order with 3 channels (RGB)
        int idx = (y * cols + x) * 3;
        // RGB2Gray: 0.299 * R + 0.587 * G + 0.114 * B
        unsigned char R = rgbImage[idx];
        unsigned char G = rgbImage[idx + 1];
        unsigned char B = rgbImage[idx + 2];
        grayImage[y * cols + x] = (unsigned char)(0.299f * R + 0.587f * G + 0.114f * B);
    }
}

cv::Mat loadImage(const std::string& imagePath) {
    cv::Mat imageBGR = cv::imread(imagePath, cv::IMREAD_COLOR);
    if (imageBGR.empty()) {
        throw std::runtime_error("Failed to load image from: " + imagePath);
    }
    cv::Mat imageRGB;
    cv::cvtColor(imageBGR, imageRGB, cv::COLOR_BGR2RGB);
    return imageRGB;
}

int main() {
    // Load image on host
    std::string imagePath = "data/color_image.jpg";
    cv::Mat h_rgbImage = loadImage(imagePath);

    // 3 channels for RGB, 1 channel for greyscale
    size_t totalBytesIn = h_rgbImage.total() * h_rgbImage.elemSize();
    size_t totalBytesOut = h_rgbImage.total();

    cv::Mat h_grayImage(h_rgbImage.rows, h_rgbImage.cols, CV_8UC1); 

    // Allocate device memory
    unsigned char *d_rgbImage, *d_grayImage;
    cudaMalloc((void **)&d_rgbImage, totalBytesIn);
    cudaMalloc((void **)&d_grayImage, totalBytesOut);

    // Copy image from host to device
    // Image is copied as flat memory using row-major order
    cudaMemcpy(d_rgbImage, h_rgbImage.data, totalBytesIn, cudaMemcpyHostToDevice);

    // Launch kernel
    // 16x16 decent default for images due to fully utilizing warps (16x16%32 = 0),
    // providing enough threads while leaving shared memory for intermediate computations 
    // and usually utilising coalesced memory access patterns    
    dim3 blockSize(16, 16);
    dim3 gridSize((h_rgbImage.cols + blockSize.x - 1) / blockSize.x, 
              (h_rgbImage.rows + blockSize.y - 1) / blockSize.y);
    rgb2Gray<<<gridSize, blockSize>>>(d_rgbImage, d_grayImage, h_rgbImage.rows, h_rgbImage.cols);

    // Return result from device to host
    cudaMemcpy(h_grayImage.data, d_grayImage, totalBytesOut, cudaMemcpyDeviceToHost);
    cudaFree(d_rgbImage);
    cudaFree(d_grayImage);
    
    // Save image to disk
    cv::imwrite("data/grayscale_output.jpg", h_grayImage);

    return 0;
}

// nvcc -I/usr/include/opencv4 -o rgb2gray rgb2gray.cu -L/usr/lib/x86_64-linux-gnu -lopencv_core -lopencv_imgproc -lopencv_imgcodecs