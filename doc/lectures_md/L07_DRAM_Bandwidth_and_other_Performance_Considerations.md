# ECE 408 / CS 483 / CSE 408
## Lecture 7: DRAM Bandwidth and Other Performance Considerations

**Instructor:** Volodymyr Kindratenko  
**학기:** Spring 2026

---

## 1. Course Reminders

### Lab 3

- **마감:** 이번 주 금요일
- 구현할 것: 필수 코드 전부 (host memory allocation/deallocation 포함)
- GitHub에 push 하는 것 잊지 말 것
- Lab 3 퀴즈는 코드 완료 후에만 제출

### Course Chatbot 활용

- 예시 질문:
  - How do I allocate device memory?
  - How do I copy memory from host to device?
  - How do I call a GPU kernel?
  - Show an example of declaring a shared memory array in CUDA kernel.
  - Explain vector addition kernel code provided in the lectures.
  - Explain following code line by line in detail: "code goes here"
  - I am preparing for the exam, please help me to review the course materials from lectures 2 and 3. Ask me questions about it. Ask one question at a time, wait for my answer, and then check it for correctness.

---

## 2. Today's Objectives

- 동적 RAM(DRAM) 기반 메모리 구조 이해
- Burst mode와 multiple banks(둘 다 병렬성 원천)를 통한 DRAM 성능(데이터 속도) 향상 이해
- GPU 커널 성능과 DRAM 구조를 연결하는 **memory access coalescing** 이해
- 다양한 커널 최적화 기법과 그 사이의 트레이드오프 학습

---

## 3. Performance Optimizations Overview

### Performance optimizations covered so far

- Tuning resource usage to maximize occupancy to hide latency in cores
- Threads per block, shared memory per block, registers per thread
- Occupancy calculation
- Reducing control divergence to increase SIMD efficiency
- Shared memory and register tiling to reduce memory traffic

### More optimizations to be covered today

- Memory coalescing
- Maximizing occupancy (again) to hide memory latency
- Thread coarsening
- Loop unrolling
- Double-buffering

### Optimizations to be covered later

- Privatization
- Warp-level primitives

---

## 4. Review: Parallel Computing Challenges

### Global Memory Bandwidth

- **Ideal vs. Reality**
- Conflicting data accesses cause serialization and delays
- Massively parallel execution cannot afford serialization
- Contentions in accessing critical data cause serialization

---

## 5. Most Large Memories Use DRAM

### Random Access Memory (RAM)

- Same time needed to read/write any address

### Dynamic RAM (DRAM)

- Bit stored on a capacitor
- Connected via transistor to bit line for read/write
- Bits disappear after a while (around 50 msec, due to tiny leakage currents through transistor), and must be rewritten (hence *dynamic*)

---

## 6. Many Cells (Bits) per Bit Line

- (Figure: BIT LINE / SELECT / BIT …)
- About 1,000 cells connect to each BIT LINE
- Connection/disconnection depends on SELECT line
- Some address bits decoded to connect exactly one cell to the BIT LINE
- A DRAM bank consists of a 2D array of DRAM cells activated one row at a time, and read at the column

---

## 7. DRAM is Slow But Dense

- (Figure: capacitance / sense amplifier…)
- **Capacitance:** tiny for the BIT, but huge for the BIT LINE
- Use an amplifier for higher speed — still slow
- But only need 1 transistor per bit

---

## 8. DRAM Bank Organization

- (Figure: BIT LINE / SELECT / Sense Amps …)
- A DRAM bank consists of a 2D array of DRAM cells activated one row at a time, and read at the column
- SELECT lines connect to about 1,000 bit lines
- Core DRAM array has about O(1M) bits
- Use more address bits to choose bit line(s)

---

## 9. A Very Small (8×2 bit) DRAM Bank Example

- (Figure: DRAM Array / Row Decoder / Sense Amps / Column Latches / Mux / Row Address / Column Address / Data / Burst)
- Use part of the address to select the row
- Read the entire row (**burst**)
- Use other part of the address to select columns within row

### Same burst vs different burst

- **Accessing data in the same burst:** No need to read the row again, just the multiplexer
- **Accessing data in different bursts:** Need to access the array again

➡ Accessing data in the same burst is faster than accessing data in different bursts

---

## 10. Memory Coalescing

- When threads in the **same warp** access **consecutive memory locations** in the same burst, the accesses can be combined and served by **one burst**
- One DRAM transaction is needed → **memory coalescing**
- If threads in the same warp access locations **not** in the same burst, accesses cannot be combined
- Multiple transactions are needed → takes longer; sometimes called **memory divergence**

---

## 11. Memory Coalescing Examples

### Vector addition

```c
int i = blockDim.x*blockIdx.x + threadIdx.x;
z[i] = x[i] + y[i];
```

- Accesses to x, y, and z are coalesced
- e.g., threads 0 to 31 access elements 0 to 31, resulting in one memory transaction to service warp 0

---

## 12. Review: Placing a 2D C Array Into Linear Memory Space

- (Figure: M0,0 M0,1 … linearized order in increasing address)

---

## 13. Two Access Patterns

- (Figure: patterns (a), (b))
- `N[k*Width+Col]`
- `M[Row*Width+k]`
- k is loop counter in the inner product loop of the kernel code

---

## 14. Fine-Grain Thread Granularity

- (Figure: multiple SM blocks over time)

### Advantage

- Provide hardware with as many threads as possible to fully utilize resources
- If more threads are provided than the GPU can support, the hardware can serialize the work with low overhead
- If future GPUs come out with more resources, more parallelism can be extracted without code being rewritten (transparent scalability)

### Disadvantage

- If there is an overhead for parallelizing work across more threads, that overhead is maximized
- Okay if threads actually run in parallel
- Suboptimal if threads are getting serialized by the hardware

---

## 15. Thread Coarsening

### Definition

- Thread coarsening: a thread is assigned **multiple units** of parallelizable work (thread is made more coarse-grain)

### Example

**Before (fine-grain):**

```c
unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
foo(i);
```

**After (thread coarsening):**

```c
unsigned int iStart = 4*(blockIdx.x*blockDim.x + threadIdx.x);
for(unsigned int c = 0; c < 4; ++c) {
  unsigned int i = iStart + c;
  foo(i);
}
```

### Advantages

- Reduces the overhead incurred for parallelization
- Could be redundant memory accesses, redundant computations, synchronization overhead, or control divergence
- Many examples throughout the course

### Disadvantages

- More resources per thread → may affect occupancy
- Underutilizes resources if coarsening factor is too high
- Need to retune coarsening factor for each device

---

## 16. Loop Unrolling

- Loop unrolling: transform a loop by **replicating the body** by some factor and **reducing the number of iterations** by the same factor
- In practice, loop unrolling and instruction scheduling are applied **automatically by the compiler**
- Can be controlled with directives

### Example

```c
#pragma unroll 4
for(unsigned int i = 0; i < 16; ++i) {
  A(i);
  B(i);
}
```

---

## 17. Loop Unrolling and Thread Coarsening

- A common source of loops that can be unrolled is **thread coarsening**
- Added benefit of unrolling: enabling **promoting local arrays to registers**

**Thread coarsening:**

```c
int x[4];
for(unsigned int c = 0; c < 4; ++c) {
  foo(x[c]);
}
```

- x placed in global memory instead of registers (variable index)

**Loop unrolling:**

```c
int x[4];
foo(x[0]);
foo(x[1]);
foo(x[2]);
foo(x[3]);
```

- x can be promoted to registers (constant index)
- Constant coarsening factor allows the coarsening loop to be fully unrolled

---

## 18. Double Buffering

### Idea

- Double buffering **eliminates false dependences** by using a **different memory buffer for writing** than the buffer containing the data being read

### Example (single buffer)

```c
for(unsigned int i = 0; i < N; ++i) {
  ... = buffer[anotherThreadID];
  __syncthreads();
  buffer[myThreadID] = ...;
  __syncthreads();
}
```

- This synchronization enforces a **false dependence** (we only need to finish reading before others write because we use the same buffer)
- This synchronization enforces a **true dependence** (we must finish writing before others can read)

### Example (double buffer)

```c
for(unsigned int i = 0; i < N; ++i) {
  ... = inBuffer[anotherThreadID];
  outBuffer[myThreadID] = ...;
  __syncthreads();
  swap(inBuffer, outBuffer);
}
```

- No extra synchronization needed: writes to outBuffer do not affect data in inBuffer

---

## 19. Checklist of Common Optimizations

### Compute utilization: Occupancy tuning

- **Benefit to Compute Cores:** More work to hide pipeline latency
- **Benefit to Memory:** More parallel memory accesses to hide DRAM latency
- **Strategy:** Tune SM resources (threads per block, shared memory per block, registers per thread)

### Loop unrolling

- **Benefit to Compute Cores:** Fewer branch instructions, more independent instruction sequences, fewer stalls
- **Benefit to Memory:** May enable promoting local arrays to registers → less global memory traffic
- **Strategy:** Done by compiler; use loops with constant bounds to help the compiler

### Reducing control divergence

- **Benefit:** High SIMD efficiency (fewer idle cores during SIMD execution)
- **Strategy:** Rearrange assignment of threads to work/data; rearrange data layout

### Memory utilization: Coalesceable global memory accesses

- **Benefit:** Fewer pipeline stalls waiting for global memory; less global memory traffic; better use of bursts/cache-lines
- **Strategy:** Rearrange mapping of threads to data; data layout; transfer global ↔ shared in coalesceable manner, do irregular accesses in shared memory (e.g., corner turning)

### Shared memory tiling

- **Benefit:** Fewer pipeline stalls; less global memory traffic
- **Strategy:** Place data reused within a block in shared memory so it is transferred from global to SM only once

### Register tiling

- **Benefit:** Fewer pipeline stalls for shared memory; less shared memory traffic
- **Strategy:** Place data reused within a warp/thread in registers so it is transferred from shared to registers only once

### Privatization

- **Benefit:** Fewer stalls waiting for atomic updates; less contention and serialization of atomics
- **Strategy:** Apply partial updates to a private copy, then update the public copy when done

### Synchronization latency: Warp-level primitives

- **Benefit:** Reduce block-wide barrier synchronizations; less shared memory traffic
- **Strategy:** Do barrier-requiring operations at warp-level, then consolidate at block-level

### Double buffering

- **Benefit:** Eliminates barriers that enforce false dependences
- **Strategy:** Use different buffers for writes and preceding reads

### General: Thread coarsening

- **Depends on:** Overhead of parallelization
- **Strategy:** Assign multiple units of parallelism per thread to reduce unnecessary parallelization overhead

---

## 20. Trade-off Between Optimizations

- **Maximizing occupancy:** Hides pipeline latency, but threads may compete for resources (registers, shared memory, cache)
- **Shared memory tiling:** More shared memory → more data reuse, but may limit occupancy
- **Thread coarsening:** Reduces parallelization overhead, but more resources per thread → may limit occupancy

➡ **How do we know which optimization to apply?**

---

## 21. Know Your Application's Bottleneck

- The constraint that limits performance on a device is the **bottleneck**
- Bottleneck depends on **application** and **device**
- Optimizations trade one resource for another to alleviate the bottleneck
- **Properly diagnose** your application's bottleneck before applying optimizations — otherwise you may optimize the wrong resource
- Know when to stop by comparing performance to **speed-of-light** performance

---

## 22. Things to Read / Things to Do

### Things to Read

- Textbook chapter 6
- CUDA BPG: Memory Optimizations

### Things to Do

- Submit Lab 3
- Sign up for GPT-2 project, if you wish

---

## 23. Problem Solving

### Q1

**Q:** Which of the following two lines of a kernel code has better performance?

```c
int tx = blockIdx.x * blockDim.x + threadIdx.x;
int ty = blockIdx.y * blockDim.y + threadIdx.y;
```

- A) `B[ty * Width + tx] = 2 * A[ty * Width + tx];`
- B) `B[tx * Width + ty] = 2 * A[tx * Width + ty];`

**A:** Cannot tell. We do not know how the kernel was launched.

---

### Q2

**Q:** Consider a DRAM system with a burst size of 512 bytes and a peak bandwidth of 240 GB/s. Assume a thread block size of 1024 and warp size of 32 and that A is a floating-point array in global memory. What is the maximal memory data access throughput we can hope to achieve in the following access to A?

```c
int i = blockIdx.x * blockDim.x + threadIdx.x;
float temp = A[4*i] + A[4*i+1];
```

**A:** From a burst of 512 bytes, we will only use every other 8 bytes (reading from [4i] and [4i+1]). So we can expect **120 GB/s**.
