## CUDA Communication & Synchronization Primitives

Modern NVIDIA GPUs provide a hierarchy of mechanisms for threads to exchange data and synchronize execution efficiently.  
Each operates at a different **scope** and uses specialized hardware resources.

| **Scope** | **Threads** | **Resource / Mechanism** | **Common Primitives** | **Notes (Modern GPUs)** |
|------------|--------------|---------------------------|------------------------|--------------------------|
| **Warp** | 32 | Register crossbar | `__shfl_sync`, `__shfl_down_sync`, `__shfl_up_sync`, `__ballot_sync`, `__any_sync`, `__all_sync`, `__syncwarp` | Fast intra-warp data exchange and vote operations. Operate entirely in registers (1–2 cycles). |
| **Warpgroup** *(Hopper+)* | 128 (4 warps) | Warpgroup shared memory (WSM) | `WGMMA` (tensor ops), warpgroup barriers, `TMA` async tile copies | Hardware unit for multi-warp tensor operations and asynchronous DMA via the Tensor Memory Accelerator. |
| **Thread Block** | ≤1024 | Shared memory (SMEM) | `__syncthreads`, `__syncthreads_count`, `__syncthreads_or`, `__syncthreads_and`, block atomics | Classic shared-memory synchronization. Basis for most block-wide reductions and staging buffers. |
| **Block (Async)** *(Ampere+)* | ≤1024 | Shared memory + async copy engine | `cp.async`, `cp.async.commit_group`, `cp.async.wait_group`, `cuda::pipeline` | Asynchronous per-thread or per-group copies; overlap data movement with compute. |
| **Cluster** *(Hopper+)* | Multiple blocks on one GPC | Cluster shared memory | `cg::this_cluster()`, `cluster.sync()`, TMA multicast | Enables inter-block cooperation via cluster-wide shared memory and barriers. |
| **Grid** | All blocks in kernel | Global memory / scheduler | `cg::this_grid().sync()`, global atomics, cooperative kernel launch | Grid-wide synchronization for cooperative kernels (`cudaLaunchCooperativeKernel`). |
| **Global Asynchronous DMA** *(Hopper+)* | SM-level hardware | Tensor Memory Accelerator (TMA) | `tma.load_async`, `tma.store_async`, async barriers | Multi-dimensional, hardware-accelerated tile copies between global and shared memory. |

### Key Concepts

- **Asynchronous copies** (`cp.async`, `TMA`) allow overlapping memory transfers and computation.
- **Cooperative Groups (`cg::...`)** provide a portable API for forming and synchronizing logical thread groups at any scope.
- **Warpgroup operations** (Hopper+) enable tightly coupled 128-thread tensor-core compute.
- **Thread clusters** (Hopper+) extend shared memory and synchronization beyond a single block.
- **Grid synchronization** requires cooperative kernel launches.

---

### Hierarchy Summary

```text
Warp (32) → Warpgroup (128) → Block (≤1024) → Cluster (multi-block) → Grid (all blocks)
registers       WSM              shared mem        cluster SMEM          global mem