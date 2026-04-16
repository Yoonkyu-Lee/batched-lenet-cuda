# ECE 408 / CS 483 / CSE 408
## Lecture 5: CUDA Memory Model

**Instructor:** Volodymyr Kindratenko  
**학기:** Spring 2026

---

## 1. Lecture Overview

### Lecture Goals

- CUDA의 **메모리 계층 구조** 이해
- 각 메모리 타입의
  - 위치
  - 접근 속도
  - 가시성(scope)
  - 용도
- **성능 관점에서 왜 메모리가 중요한지** 이해
- 이후 강의:
  - Tiling
  - Shared memory 최적화
  - Coalescing
  를 위한 기초 다지기

---

## 2. Why Memory Matters

### 현실적인 GPU 한계

- 연산 성능 (FLOPS)은 매우 큼
- 하지만:
  - **Global memory latency**: 수백 사이클
  - **Memory bandwidth**가 병목이 되기 쉬움

➡ 많은 CUDA 커널은 **memory-bound**

---

## 3. CUDA Memory Hierarchy (Overview)

```
Registers (per-thread, fastest)
Shared Memory (per-block, very fast)
L1 Cache
L2 Cache (device-wide)
Global Memory (slow, large)
```

**추가 메모리:**

- Constant memory
- Texture memory
- Local memory (이름과 달리 off-chip)

---

## 4. Registers

### Registers

- 위치: **SM 내부**
- 가시성: **thread-private**
- 속도: 가장 빠름

### 특징

- 자동 할당 (프로그래머가 직접 지정 X)
- 레지스터 부족 시 → **local memory spill**

➡ Register usage는 **occupancy에 직접 영향**

---

## 5. Local Memory (Spill Memory)

### Local Memory란?

- Thread-private
- 하지만 **global memory에 위치**

### 언제 발생?

- 레지스터 부족
- Large arrays declared inside kernel
- 복잡한 indexing

### 성능

- 매우 느림 (global memory latency)

➡ 이름에 속지 말 것: **local ≠ fast**

---

## 6. Shared Memory

### Shared Memory

- 위치: **SM 내부**
- 가시성: **block-wide**
- 속도: registers 다음으로 빠름

### 용도

- Thread 간 데이터 공유
- Global memory 접근 최소화
- Tiling 알고리즘의 핵심

```c
__shared__ float tile[TILE_DIM][TILE_DIM];
```

---

## 7. Shared Memory Characteristics

- **Lifetime:** block execution 동안
- **Banked memory** 구조
- **Synchronization 필요:**

```c
__syncthreads();
```

➡ 올바른 사용 시 dramatic speedup

---

## 8. Global Memory

### Global Memory

- 위치: off-chip DRAM
- 가시성: 모든 threads
- 용량: GB 단위
- 속도: 가장 느림

### 특징

- High latency
- High bandwidth (if accesses are coalesced)

---

## 9. Memory Coalescing

### Coalesced Access

- Warp의 threads가 연속적인 주소에 접근

```c
A[base + threadIdx.x]
```

### Non-Coalesced Access

```c
A[base + threadIdx.x * stride]
```

➡ 같은 데이터량이라도 transaction 수 차이 큼

---

## 10. L1 / L2 Cache

### L1 Cache

- SM-local
- 작음
- Shared memory와 리소스 공유 (architecture dependent)

### L2 Cache

- Device-wide
- 모든 SM이 공유
- Global memory traffic 완화

➡ 캐시는 성능 힌트, 성능 보장은 아님

---

## 11. Constant Memory

### Constant Memory

- Read-only
- Device-wide
- Cache optimized for: 모든 thread가 같은 주소를 읽을 때

```c
__constant__ float coeff[64];
```

### 특징

- Broadcast 가능
- Access pattern 나쁘면 global memory처럼 느림

---

## 12. Texture Memory (개념 소개)

### Texture Memory

- Read-only
- Cache optimized for:
  - 2D spatial locality
  - Addressing modes 지원

➡ 이미지 처리에서 유용  
➡ 자세한 내용은 후반부

---

## 13. Memory Scope Summary

| Memory   | Scope | Location | Speed              |
|----------|-------|----------|--------------------|
| Register | Thread| On-chip  | Fastest            |
| Shared   | Block | On-chip  | Very fast          |
| Local    | Thread| Off-chip | Slow               |
| Global   | Grid  | Off-chip | Slow               |
| Constant | Grid  | Cached   | Fast (broadcast)   |

---

## 14. Example: Why Tiling Helps

### Without Tiling

- Same data loaded repeatedly from global memory
- Low arithmetic intensity

### With Tiling

- Load once into shared memory
- Reuse many times
- Reduce global memory traffic

➡ Memory-bound → compute-bound 가능

---

## 15. Memory Bandwidth Example

**Given:**

- 150 GB/s bandwidth
- float = 4 Bytes

**Max loads:**

\[
\frac{150}{4} = 37.5\ \text{G elements/s}
\]

**If reused 16 times:**

\[
37.5 \times 16 = 600\ \text{GFLOP/s}
\]

---

## 16. Synchronization and Memory

### `__syncthreads()`

- Block-level barrier
- Ensures:
  - All loads complete
  - All writes visible
- ❗ All threads in block must reach it

---

## 17. Memory Performance Pitfalls

- Excessive register usage → spill
- Large shared memory usage → low occupancy
- Non-coalesced global access
- Branch divergence during memory access

---

## 18. Programming Guidelines

- Minimize global memory accesses
- Maximize data reuse
- Prefer shared memory for reuse
- Check register usage (`nvcc --ptxas-options=-v`)
- Measure, don't guess

---

## 19. Exam / Quiz Traps

- Local memory is slow
- Cache ≠ guaranteed reuse
- Coalescing happens at warp level
- Shared memory requires synchronization
- Memory scope questions are common

---

## 20. Key Takeaways

- CUDA performance is often memory-limited
- Memory hierarchy must be exploited explicitly
- Shared memory is programmer-managed cache
- Correctness ≠ performance
- Memory access pattern matters as much as algorithm

---

## 21. Looking Ahead

### Next Lecture

- Lecture 6: Data Locality & Tiled Matrix Multiply
- Full optimization of matrix multiplication
- Shared memory + synchronization in practice

---

## 22. One-Liners for Exams

- Registers fastest, global slowest
- Local memory lives in global memory
- Coalescing happens per warp
- Shared memory enables reuse
- Synchronization is mandatory
