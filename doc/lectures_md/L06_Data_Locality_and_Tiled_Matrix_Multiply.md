# ECE 408 / CS 483 / CSE 408
## Lecture 6: Data Locality and Tiled Matrix Multiply

**Instructor:** Volodymyr Kindratenko  
**학기:** Spring 2026

---

## 1. Lecture Overview

### Lecture Goals

- **Data locality** 개념 이해
- Global memory 병목의 원인 재확인
- **Shared memory tiling** 전략 학습
- Tiled matrix multiplication의
  - 구조
  - 동기화
  - 경계 조건 처리
- Tiling이 성능을 어떻게 바꾸는지 정량적으로 이해

---

## 2. Motivation: Why Tiling?

### GPU Example (from lecture)

- Peak compute: **1000 GFLOP/s**
- Memory bandwidth: **150 GB/s**

If one FLOP needs 4 Bytes:

\[
\frac{150}{4} = 37.5 \text{ GFLOP/s}
\]

➡ **Memory bandwidth limits performance**

---

## 3. Matrix Multiplication Review

Given matrices A, B, C (N×N):

\[
C_{ij} = \sum_{k=0}^{N-1} A_{ik} B_{kj}
\]

### Data Reuse Opportunity

- Each element of A is reused **N times**
- Each element of B is reused **N times**

➡ But naive CUDA kernel **does not exploit reuse**

---

## 4. Memory Bottleneck in Naive Kernel

### Naive Kernel Behavior

- Each thread:
  - Loads A[i][k] from global memory
  - Loads B[k][j] from global memory
- Same elements loaded repeatedly by different threads

➡ **Wasted bandwidth**

---

## 5. Data Locality

### Types of Locality

- **Temporal locality**: reuse the same data
- **Spatial locality**: nearby data accessed together

Matrix multiplication has **excellent temporal locality**  
→ must be exploited manually on GPU

---

## 6. Tiling: Core Idea

### Tiling Strategy

- Partition matrices into **small tiles**
- Each tile:
  - Loaded once from global memory
  - Stored in **shared memory**
  - Reused by all threads in a block

➡ Dramatically reduces global memory traffic

---

## 7. Tiled Matrix Multiplication Structure

Each block computes a tile of C:

1. Load a tile of A and B into shared memory
2. Synchronize threads
3. Compute partial dot products
4. Synchronize
5. Move to next tile
6. Accumulate result

---

## 8. Shared Memory Declarations

```c
__shared__ float A_s[TILE_DIM][TILE_DIM];
__shared__ float B_s[TILE_DIM][TILE_DIM];
```

- **A_s:** tile of A
- **B_s:** tile of B

---

## 9. Thread Indexing (2D)

```c
unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
```

- Each thread computes one C[row][col]

---

## 10. Tiled Kernel (Core Structure)

```c
float sum = 0.0f;

for (unsigned int tile = 0; tile < N / TILE_DIM; ++tile) {

    // Load tiles
    A_s[threadIdx.y][threadIdx.x] =
        A[row * N + tile * TILE_DIM + threadIdx.x];

    B_s[threadIdx.y][threadIdx.x] =
        B[(tile * TILE_DIM + threadIdx.y) * N + col];

    __syncthreads();

    // Compute partial product
    for (unsigned int k = 0; k < TILE_DIM; ++k) {
        sum += A_s[threadIdx.y][k] * B_s[k][threadIdx.x];
    }

    __syncthreads();
}

C[row * N + col] = sum;
```

---

## 11. Why Synchronization Is Required

### After Loading

- Ensure all threads finished loading
- Prevent using incomplete tiles

### After Computing

- Prevent overwriting shared memory
- Before loading next tile

```c
__syncthreads();
```

❗ All threads in block must reach it

---

## 12. Handling Boundary Conditions

### Problem

- N may not be a multiple of TILE_DIM
- Threads may:
  - Access out-of-bounds A or B
  - Compute invalid C elements

---

## 13. Tile Count Fix

**Original:**

```c
tile < N / TILE_DIM
```

**Correct (round up):**

```c
tile < (N - 1) / TILE_DIM + 1
```

---

## 14. Boundary Handling During Tile Load

### Load A Tile

```c
if (row < N && tile*TILE_DIM + threadIdx.x < N)
    A_s[threadIdx.y][threadIdx.x] = A[row*N + tile*TILE_DIM + threadIdx.x];
else
    A_s[threadIdx.y][threadIdx.x] = 0;
```

### Load B Tile

```c
if (tile*TILE_DIM + threadIdx.y < N && col < N)
    B_s[threadIdx.y][threadIdx.x] = B[(tile*TILE_DIM + threadIdx.y)*N + col];
else
    B_s[threadIdx.y][threadIdx.x] = 0;
```

---

## 15. Why Writing Zero Works

- Threads outside valid range:
  - Multiply by 0
  - Do no harm to sum
- Avoids branching inside compute loop
- Keeps kernel simple and uniform

---

## 16. Boundary Handling for Output

```c
if (row < N && col < N) {
    C[row * N + col] = sum;
}
```

- Threads outside C compute garbage but do not store

---

## 17. Performance Impact of Tiling

**Example: TILE_DIM = 16**

- Each element reused 16 times
- Global memory traffic reduced by 16×

**Effective bandwidth:**

\[
(150/4) \times 16 = 600\ \text{GFLOP/s}
\]

---

## 18. TILE_DIM Trade-offs

**TILE_DIM = 16**

- 256 threads/block
- Lower shared memory usage
- More blocks per SM

**TILE_DIM = 32**

- 1024 threads/block
- Higher data reuse
- Fewer blocks per SM

➡ Both can expose same memory-level parallelism

---

## 19. Occupancy vs Bandwidth

- **Large tiles:**
  - Reduce memory traffic
  - May reduce occupancy
- **But:**
  - Memory bottleneck removed
  - Compute becomes limiting

➡ Lower occupancy can still be faster

---

## 20. Branch Divergence Impact

- Boundary checks cause divergence
- Only affects boundary blocks
- Negligible for large matrices

---

## 21. Key Lessons from Tiling

- Shared memory = programmer-managed cache
- Synchronization is mandatory
- Boundary conditions must be handled carefully
- Zero-padding simplifies computation
- Performance gains can be orders of magnitude

---

## 22. Tiling on CPU (Conceptual)

- CPU uses caches instead of shared memory
- Fewer threads, larger caches
- Same idea:
  - Block computation
  - Reuse data before eviction

---

## 23. Exam / Quiz Traps

- Forgetting `__syncthreads()`
- Incorrect tile loop bound
- Mixing up A/B boundary conditions
- Writing out-of-bounds C
- Assuming occupancy always dominates

---

## 24. One-Liners for Exams

- Tiling exploits data reuse
- Shared memory reduces global memory traffic
- Zero-padding avoids complex branching
- Synchronization is required between phases
- Lower occupancy can still win
