## **Matrix Multiplication Computational Complexity**
For a square matrix multiplication of size \(N \times N\):

- It requires **\(O(N^3)\) operations**, since for every \(N^2\) output elements, we perform **\(2N\)** operations.
- A naive implementation has **\(O(N^3)\) global memory accesses**, where for every \(N^2\) output elements, we fetch **\(2N\)** elements from global memory.

## **Operational Intensity (OP/B)**
Operational intensity (OP/B) is defined as the number of operations per byte of memory access.

For FP32:
```
OP/B = O(N^3) / (O(N^3) * 4 bytes) = 1/4
```

Given that:
- **Fetching from global memory** takes **~300-600 cycles**.
- **A fused multiply-add (FMA) operation** takes **~4 cycles**.

Since memory fetches are significantly more expensive than arithmetic operations, **this makes matrix multiplication heavily memory-bound**.

---

## **Tiling for Memory Optimization**
Tiling reduces the number of global memory fetches by a factor of \(N/T\), where \(T\) is the **tile size**. This increases **data reuse** and reduces redundant global memory accesses.

For FP32:
```
OP/B = T / 4
```
---

## **Thread Coarsening for Further Optimization**
Thread coarsening reduces **global memory fetches** by a factor of **\(C\)** (coarsening factor). This means each thread computes **\(C\)** output elements instead of just one.

However, **coarsening too much reduces parallelism**, so it needs **tuning**.

For FP32:
```
OP/B = (T * C) / 4
```

