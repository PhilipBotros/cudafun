import ctypes
import numpy as np
from numpy.ctypeslib import ndpointer

# Load shared library created using:
# nvcc -Xcompiler -fPIC -shared -o libsum.so pykernel_sum.cu
lib = ctypes.CDLL('./libsum.so')

# Configure function signature
lib.launch_sum_kernel.argtypes = [
    ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
    ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
    ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
    ctypes.c_int,
]
lib.launch_sum_kernel.restype = None

# Allocate arrays; NumPy arrays are wrappers around C arrays and are contiguous 
# and we can use them directly (provided we did not use any reshaping or indexing 
# that would change the memory layout to be non-contiguous)
N = 1024
a = np.random.rand(N).astype(np.float32)
b = np.random.rand(N).astype(np.float32)
out = np.zeros_like(a)

# Call the kernel and verify the result
lib.launch_sum_kernel(a, b, out, N)

np.testing.assert_allclose(out, a + b, rtol=1e-5)
print("Success, same result!")