# ECE 408 / CS 483 / CSE 408
## Lecture 3: Multidimensional Grids & Basic Matrix Multiplication

**Instructor:** Volodymyr Kindratenko  
**학기:** Spring 2026

---

## 1. Lecture Overview

### Lecture Goals

- CUDA에서 **2D / multidimensional grid & block** 개념 이해
- 행렬 연산을 CUDA thread 구조로 매핑하는 방법 학습
- **Matrix Multiplication의 병렬화 기본 형태** 이해
- 성능 문제의 원인이 되는 **Global Memory Access 패턴** 인식

---

## 2. From CPU to GPU: Matrix Multiplication

### Matrix Multiplication Definition

For square matrices:

\[
P_{ij} = \sum_{k=0}^{W-1} M_{ik} \cdot N_{kj}
\]

- M: input matrix
- N: input matrix
- P: output matrix
- Width = W

### CPU Implementation (Baseline)

```c
void MatrixMul(float *M, float *N, float *P, int Width) {
    for (int i = 0; i < Width; ++i) {
        for (int j = 0; j < Width; ++j) {
            float sum = 0;
            for (int k = 0; k < Width; ++k) {
                sum += M[i * Width + k] * N[k * Width + j];
            }
            P[i * Width + j] = sum;
        }
    }
}
```

- Outer loops: independent → parallelizable
- Inner loop (k-loop): reduction → handled later

---

## 3. Parallelization Strategy

### Key Idea

- Each thread computes exactly one element of P
- Parallelize the i, j loops
- One thread ↔ one (row, col) element of P

---

## 4. 2D Grids and 2D Blocks

### Why 2D?

- Output matrix P is 2D
- Natural mapping:
  - `threadIdx.x` → column
  - `threadIdx.y` → row

### Grid and Block Configuration

```c
dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);

dim3 dimGrid(
    ceil((float)Width / TILE_WIDTH),
    ceil((float)Width / TILE_WIDTH),
    1
);
```

- Each block computes a TILE_WIDTH × TILE_WIDTH tile
- Grid covers entire output matrix

---

## 5. CUDA Kernel for Matrix Multiplication

### Thread Index Computation

```c
int Row = blockIdx.y * blockDim.y + threadIdx.y;
int Col = blockIdx.x * blockDim.x + threadIdx.x;
```

### Naive CUDA Kernel

```c
__global__
void MatrixMulKernel(float* d_M, float* d_N, float* d_P, int Width) {

    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    int Col = blockIdx.x * blockDim.x + threadIdx.x;

    if (Row < Width && Col < Width) {
        float Pvalue = 0;
        for (int k = 0; k < Width; ++k) {
            Pvalue += d_M[Row * Width + k] * d_N[k * Width + Col];
        }
        d_P[Row * Width + Col] = Pvalue;
    }
}
```

---

## 6. What This Kernel Is Doing

- Replaces CPU outer loops with:
  - `blockIdx`
  - `threadIdx`
- Functionally correct
- Performance is very poor

---

## 7. Why Is Performance Bad?

### Global Memory Bottleneck

- Each thread:
  - Loads M[row][k] → 4B
  - Loads N[k][col] → 4B
  - Total: 8B per iteration
- Performs:
  - 1 multiply
  - 1 add → 2 FLOPs

### Arithmetic Intensity

\[
\frac{8\,\text{B}}{2\,\text{FLOPs}} = 4\,\text{B/FLOP}
\]

### Hardware Example

- Peak compute: 1000 GFLOP/s
- Memory bandwidth: 150 GB/s
- Maximum achievable FLOPS: \(\frac{150}{4} = 37.5\) GFLOP/s
- ➡ Memory-bound kernel

---

## 8. Compute-bound vs Memory-bound

### Definitions

- **Compute-bound**
  - Limited by FLOPS rate
  - Cores always busy
- **Memory-bound**
  - Limited by memory bandwidth
  - Cores stall waiting for data

### This kernel

- Memory-bound
- Cannot utilize GPU compute capability

---

## 9. Performance Metrics

### Key Metrics

- **FLOPS Rate:** operations per second
- **Memory Bandwidth:** bytes per second
- These define upper bounds, not guarantees.

---

## 10. Roofline Model (Conceptual)

- **X-axis:** Operational Intensity (OP/B)
- **Y-axis:** Performance (OP/s)
- Two limits:
  - Memory bandwidth ceiling
  - Compute ceiling
- Kernel performance is limited by whichever is lower

---

## 11. Vector Addition vs Matrix Multiplication

### Vector Addition

- 1 add per 12B accessed
- OP/B ≈ 0.083
- Strongly memory-bound

### Matrix Multiplication (Ideal)

- Data: 12N² bytes
- Ops: 2N³ FLOPs
- OP/B ≈ 0.167N
- ➡ For large N → compute-bound

**BUT naive kernel:**

- Reloads same data repeatedly
- Low OP/B in practice

---

## 12. Key Takeaways from Lecture 3

- 2D grids and blocks map naturally to matrices
- Naive CUDA kernels are often memory-bound
- Global memory access dominates runtime
- Correctness ≠ performance
- Data reuse is critical → next lecture (tiling & shared memory)

---

## 13. Exam / Quiz Pitfalls

- **Thread indexing:**

```c
row = blockIdx.y * blockDim.y + threadIdx.y
col = blockIdx.x * blockDim.x + threadIdx.x
```

- Boundary checks are mandatory
- Memory bandwidth can dominate even simple kernels
- Peak FLOPS ≠ achievable FLOPS

---

## 14. What Comes Next

- Shared memory
- Tiling
- Reusing data across threads
- Synchronization (`__syncthreads()`)
