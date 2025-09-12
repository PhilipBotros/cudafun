#include <cuda_runtime.h>
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/layout/matrix.h"

int main() {
  using Element = float;
  using Layout  = cutlass::layout::RowMajor;

  // 1) Pick a ready-made device GEMM
  using Gemm = cutlass::gemm::device::Gemm<
      Element, Layout,   // A
      Element, Layout,   // B
      Element, Layout,   // C/D
      Element            // Accumulator type
  >;

  // 2) Problem sizes (M x K) * (K x N) = (M x N)
  int M = 1024, N = 1024, K = 1024;

  // 3) Allocate device memory (row-major, leading dims = columns for row-major)
  Element *d_A, *d_B, *d_C;
  cudaMalloc(&d_A, sizeof(Element)*M*K);
  cudaMalloc(&d_B, sizeof(Element)*K*N);
  cudaMalloc(&d_C, sizeof(Element)*M*N);

  // (Optionally initialize A/B/C here...)

  // 4) Set scalars
  Element alpha = 1.f, beta = 0.f;

  // 5) Create arguments: problem size, tensor refs, leading dimensions (lda/ldb/ldc)
  Gemm::Arguments args(
    {M, N, K},                   // GemmCoord
    {d_A, K},                    // A pointer, lda
    {d_B, N},                    // B pointer, ldb
    {d_C, N},                    // C pointer, ldc
    {d_C, N},                    // D pointer, ldd (output)
    {alpha, beta}
  );

  // 6) Launch
  Gemm gemm_op;
  size_t ws = gemm_op.get_workspace_size(args);
  void* d_ws = nullptr; if (ws) cudaMalloc(&d_ws, ws);

  auto status = gemm_op.initialize(args, d_ws);
  if (status != cutlass::Status::kSuccess) return -1;

  status = gemm_op();
  if (status != cutlass::Status::kSuccess) return -2;

  cudaDeviceSynchronize();

  cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
  if (d_ws) cudaFree(d_ws);
  return 0;
}