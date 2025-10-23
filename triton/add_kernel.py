import torch
import triton
import triton.language as tl


@triton.jit
def add_kernel(
    x_ptr,  # Pointer to first input vector
    y_ptr,  # Pointer to second input vector
    output_ptr,  # Pointer to output vector
    n_elements,  # Size of the vectors
    BLOCK_SIZE: tl.constexpr,  # Number of elements each program should process
):
    """
    Triton kernel for element-wise addition of two vectors.
    Each program instance handles a block of elements.
    """
    # Get the program ID (which block this instance is processing)
    pid = tl.program_id(axis=0)

    # Calculate the starting index for this block
    block_start = pid * BLOCK_SIZE

    # Generate offsets for the elements this program will process
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Create a mask to handle cases where n_elements is not divisible by BLOCK_SIZE
    mask = offsets < n_elements

    # Load data from global memory (with mask to avoid out-of-bounds access)
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)

    # Perform the addition
    output = x + y

    # Write the result back to global memory
    tl.store(output_ptr + offsets, output, mask=mask)


def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Wrapper function to perform element-wise addition using the Triton kernel.

    Args:
        x: First input tensor
        y: Second input tensor

    Returns:
        Result of x + y
    """
    # Ensure inputs are on GPU and have the same shape
    assert x.is_cuda and y.is_cuda, "Tensors must be on CUDA"
    assert x.shape == y.shape, "Input tensors must have the same shape"

    # Allocate output tensor
    output = torch.empty_like(x)

    # Get total number of elements
    n_elements = x.numel()

    # Choose block size (power of 2, typically 256, 512, or 1024)
    BLOCK_SIZE = 1024

    # Calculate grid size (number of blocks needed)
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)

    # Launch the kernel
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE)

    return output


if __name__ == "__main__":
    # Simple test
    print("Testing Triton addition kernel...")

    # Create test tensors
    size = 98432  # Non-power-of-2 size to test masking
    x = torch.randn(size, device='cuda')
    y = torch.randn(size, device='cuda')

    # Run our Triton kernel
    triton_output = add(x, y)

    # Compare with PyTorch's native addition
    torch_output = x + y

    # Check if results match
    if torch.allclose(triton_output, torch_output):
        print(f"✓ Success! Results match for {size} elements")
        print(f"  Max difference: {(triton_output - torch_output).abs().max().item():.2e}")
    else:
        print("✗ Error! Results don't match")
        print(f"  Max difference: {(triton_output - torch_output).abs().max().item():.2e}")
