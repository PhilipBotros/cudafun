import torch
from torch.utils.cpp_extension import load

sum_ext = load(
    name="sum_cuda_ext",
    sources=["ptkernel_wrapper.cpp", "ptkernel_sum.cu"],
    verbose=True,
)

a = torch.randn(1024, device='cuda', dtype=torch.float32)
b = torch.randn(1024, device='cuda', dtype=torch.float32)
out = sum_ext.sum_cuda(a, b)

torch.testing.assert_close(out, a + b)
print("Success, same result!")