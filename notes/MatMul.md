A square matrix multiplication takes O(N^3) operations, for every N^2 output elements we perform 2N operations.
A naive implementation has O(N^3) global memory accesses, again for every N^2 output elements we fetch 2N elements.

OP/B = O(N^3) / O(N^3) / 4 bytes = 1/4 for FP32

Given that fetching from global memory takes ~300-600 cycles and a fused multiply add is ~4 cycles we are heavy memory bound.

Tiling reduces the number of global memory fetches by a factor of N/T, where T is the tile size.

OP/B = T/4 for FP32

Thread coarsening further reduces the number of global memory fetches by a factor of C, where C is the coarsening factor. This is at the potential cost of decreasing the parallelism of the kernel so has to be tuned accordingly.

OP/B = TxC / 4 for FP32

