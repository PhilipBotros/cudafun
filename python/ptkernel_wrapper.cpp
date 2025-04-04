#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

void launch_sum_kernel(const float* a, const float* b, float* out, int n, cudaStream_t stream);

torch::Tensor sum_cuda(torch::Tensor a, torch::Tensor b) {
    TORCH_CHECK(a.is_cuda(), "a must be a CUDA tensor");
    TORCH_CHECK(b.is_cuda(), "b must be a CUDA tensor");
    TORCH_CHECK(a.dtype() == torch::kFloat32, "inputs must be float32");
    TORCH_CHECK(a.sizes() == b.sizes(), "input sizes must match");

    auto out = torch::empty_like(a);
    int n = a.numel();

    launch_sum_kernel(
        a.data_ptr<float>(),
        b.data_ptr<float>(),
        out.data_ptr<float>(),
        n,
        c10::cuda::getCurrentCUDAStream()
    );

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sum_cuda", &sum_cuda, "CUDA elementwise sum");
}