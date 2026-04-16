# ECE 408 / CS 483 / CSE 408
## Lecture 4: GPU Architecture

**Instructor:** Volodymyr Kindratenko  
**학기:** Spring 2026

---

## 1. Lecture Overview

### Lecture Goals

- GPU가 왜 CPU와 다른 구조를 가지는지 이해
- GPU의 **계층적 구조** (Grid → Block → Warp → Thread)
- **SIMT (Single Instruction, Multiple Threads)** 실행 모델 이해
- 성능과 직접 연결되는 개념:
  - Warp
  - Occupancy
  - Latency hiding

---

## 2. CPU vs GPU: Architectural Philosophy

### CPU

- 목표: **Low latency**
- 특징:
  - 소수의 강력한 코어
  - 큰 캐시
  - 복잡한 제어 로직
  - Branch prediction, OoO execution

### GPU

- 목표: **High throughput**
- 특징:
  - 수천 개의 단순한 코어
  - 작은 캐시
  - 단순한 제어
  - Massive multithreading

---

## 3. Throughput-Oriented Design

### Key Observation

- Many applications tolerate latency
- GPUs hide latency with **parallelism**, not speculation

### Strategy

- When one thread stalls → run another
- Context switch cost ≈ 0

---

## 4. GPU Execution Hierarchy

### Software View

```
Grid
└── Block
    └── Thread
```

### Hardware View

```
GPU
└── SM (Streaming Multiprocessor)
    └── Warp
        └── Thread
```

---

## 5. Streaming Multiprocessor (SM)

### SM Responsibilities

- Execute warps
- Manage:
  - Registers
  - Shared memory
  - Warp scheduling

### Typical Resources (varies by architecture)

- Max threads per SM: 2048
- Max warps per SM: 64
- Shared memory: tens to hundreds of KB
- Registers: ~64K per SM

---

## 6. What Is a Warp?

### Definition

- **Warp = 32 threads**
- Basic scheduling unit in NVIDIA GPUs

### Important Properties

- All threads in a warp:
  - Execute the **same instruction**
  - On **different data**

➡ **SIMT model**

---

## 7. SIMT (Single Instruction, Multiple Threads)

### SIMT vs SIMD

- SIMD: explicit vector instructions
- SIMT: scalar instructions, hardware groups threads

### Programmer View

```c
if (threadIdx.x < 16) {
    // some code
}
```

### Hardware Reality

- Warp executes both paths
- Threads not taking the path are masked off
- ➡ Leads to branch divergence

---

## 8. Branch Divergence

### What Is Branch Divergence?

- Threads in the same warp follow different control paths

### Consequences

- Serialized execution
- Reduced effective parallelism
- Performance loss

### Example

```c
if (threadIdx.x % 2 == 0) {
    A();
} else {
    B();
}
```

- Warp executes A() for half threads
- Then B() for the other half

---

## 9. Warp Scheduling

### Warp Scheduler

- Each SM has one or more warp schedulers
- Chooses a ready warp each cycle

### Ready Warp

- Not stalled on:
  - Memory access
  - Synchronization
  - Dependencies

➡ If enough warps exist, latency is hidden

---

## 10. Latency Hiding

### Memory Latency

- Global memory latency: hundreds of cycles

### Solution

- Have many active warps
- Switch to another warp when one stalls

### Key Requirement

- High occupancy

---

## 11. Occupancy

### Definition

\[
\text{Occupancy} = \frac{\text{Active warps per SM}}{\text{Maximum warps per SM}}
\]

### What Limits Occupancy?

- Registers per thread
- Shared memory per block
- Threads per block

---

## 12. Registers and Occupancy

### Register Allocation

- Registers allocated per thread
- Total registers per SM is fixed

### Effect

- More registers per thread →
  - Fewer threads resident
  - Lower occupancy

---

## 13. Shared Memory and Occupancy

### Shared Memory Allocation

- Allocated per block
- Reduces number of blocks per SM

### Trade-off

- More shared memory:
  - Better data reuse
  - Potentially lower occupancy

---

## 14. Occupancy ≠ Performance

### Important Warning

- Maximum occupancy ≠ maximum performance

### Reason

- Past a point:
  - Additional warps do not improve latency hiding
  - Other bottlenecks dominate

➡ Occupancy is a means, not a goal

---

## 15. Memory Hierarchy (Overview)

```
Registers (per thread)
Shared Memory (per block)
L1 Cache
L2 Cache
Global Memory
```

- Registers: fastest, private
- Shared memory: fast, block-wide
- Global memory: slow, large

(Detailed model covered in Lecture 5)

---

## 16. Instruction Throughput

### Execution Units

- Integer ALUs
- Floating-point units
- Load/Store units
- Special function units (SFU)

### Warp-Level Execution

- One instruction per warp per cycle (per scheduler)

---

## 17. Coalesced Memory Access (Preview)

### Idea

- Consecutive threads accessing consecutive addresses

### Benefit

- Fewer memory transactions
- Higher effective bandwidth

➡ Critical for performance  
➡ Detailed in Lecture 5

---

## 18. Summary of GPU Architecture Principles

- GPUs are designed for throughput, not latency
- Warps are the fundamental execution unit
- SIMT execution can cause divergence
- Performance depends on:
  - Warp scheduling
  - Occupancy
  - Memory behavior

---

## 19. Common Exam Pitfalls

- Confusing thread ↔ warp ↔ block
- Assuming warp size is configurable (it is always 32)
- Thinking divergence affects blocks (it affects warps)
- Assuming higher occupancy always improves performance

---

## 20. Looking Ahead

### Next Lectures

- Lecture 5: CUDA Memory Model
- Lecture 6: Data Locality & Tiled Matrix Multiply

### These lectures build directly on

- Warps
- Shared memory
- Latency hiding

---

## 21. Key One-Liners (시험용)

- Warp = scheduling unit
- SIMT hides SIMD
- Latency hiding via massive multithreading
- Occupancy enables, but does not guarantee, performance
