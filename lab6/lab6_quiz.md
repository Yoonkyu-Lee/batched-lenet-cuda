# Lab 6 Quiz Notes

Detailed takeaways are summarized in [doc/lab6_notes.md](../doc/lab6_notes.md).

## Question 1: Shared Memory Usage in Reduction

### Problem statement

When the `BLOCK_SIZE` becomes large (`BLOCK_SIZE -> ∞`), which option is the limit of the average number of unique shared memory locations each thread accesses in the reduction step?

Choices:

- `1`
- `2`
- `3`
- `4`
- `log2(BLOCK_SIZE)`
- `log2(BLOCK_SIZE) + 1`
- `BLOCK_SIZE`
- `BLOCK_SIZE + 1`
- `2 * BLOCK_SIZE`
- `None of these are correct`

## Answer

**2**

## Short solution

In shared-memory reduction, each thread has its own slot:

```cpp
input_s[threadIdx.x]
```

So that contributes **1 unique shared-memory location per thread**.

During the tree reduction:

- at stride `BLOCK_SIZE / 2`, `BLOCK_SIZE / 2` threads access one extra partner location
- at stride `BLOCK_SIZE / 4`, `BLOCK_SIZE / 4` threads access one extra partner location
- ...
- at stride `1`, `1` thread accesses one extra partner location

So the total number of these extra partner-location accesses is:

```text
BLOCK_SIZE/2 + BLOCK_SIZE/4 + ... + 1 = BLOCK_SIZE - 1
```

Average extra unique locations per thread:

```text
(BLOCK_SIZE - 1) / BLOCK_SIZE
```

Therefore the average total unique shared-memory locations per thread is:

```text
1 + (BLOCK_SIZE - 1) / BLOCK_SIZE
= 2 - 1 / BLOCK_SIZE
```

As `BLOCK_SIZE -> ∞`:

```text
2 - 1 / BLOCK_SIZE -> 2
```

So the limit is:

**`2`**

## Question 2: T/F in Reduction

### Problem statement

Which of the following statements on an improved version of parallel reduction is **FALSE**?

- (a) The algorithm performs fewer than `BLOCK_SIZE` reduction operations (for example, floating-point additions).
- (b) The number of `__syncthreads` operations performed by each thread block depends on the size of the input array.
- (c) To perform reduction, the operator has to be associative and commutative.
- (d) On average, each thread accesses global memory approximately one time.
- (e) Data in the input are logically commuted to reduce branch divergence within CUDA warps.

## Answer

**False statements: (a), (b), (d)**

## Short solution

### (a) False

In the improved reduction, one block reduces `2 * BLOCK_SIZE` inputs to one output.

That needs:

```text
2 * BLOCK_SIZE - 1
```

additions in total, not fewer than `BLOCK_SIZE`.

### (b) False

For a fixed `BLOCK_SIZE`, the number of `__syncthreads()` calls per block is determined by the reduction tree depth, so it depends on:

```text
log2(BLOCK_SIZE)
```

not on the total input array length.

### (c) True

Parallel reduction relies on an operator that can be regrouped safely. In lecture terms, the operator should be **associative and commutative**.

### (d) False

In the improved shared-memory reduction, each thread usually reads **two** global values at the start:

```cpp
input[i] and input[i + BLOCK_SIZE]
```

Then the rest of the reduction happens in shared memory. So the average global-memory access count per thread is closer to **2 reads** than to 1.

### (e) True

The improved reduction reorders the reduction pattern so that active threads are grouped more cleanly. This reduces branch divergence compared with the naive `% stride == 0` mapping.

## Question 3: Reduction Memory, Computation, and Synchronization

### Problem statement

The improved version of the parallel reduction kernel discussed in class is used to reduce an input array of `63644` floats. Each thread block in the grid contains `256` threads.

Find:

- **A.** How many Bytes are read from global memory by the grid executing the kernel?
- **B.** How many Bytes are written to global memory by the grid executing the kernel?
- **C.** How many floating-point operations are performed by the grid executing the kernel?
- **D.** How many times does a single thread block synchronize to reduce its portion of the array to a single value?

## Answers

- **A.** `254576` Bytes
- **B.** `500` Bytes
- **C.** `63875` floating-point operations
- **D.** `9` calls to `__syncthreads()`

## Short solution

Each block has `256` threads, and the improved reduction handles:

```text
2 * BLOCK_SIZE = 512
```

input values per block.

So the number of blocks is:

```text
ceil(63644 / 512) = 125
```

### A. Global memory reads

Each input float is read once from global memory.

```text
63644 floats * 4 Bytes/float = 254576 Bytes
```

### B. Global memory writes

Each block writes one partial sum to global memory.

```text
125 blocks * 1 float/block * 4 Bytes = 500 Bytes
```

### C. Floating-point operations

For one block:

- initial pairwise sum during load: `256` additions
- reduction tree in shared memory: `128 + 64 + 32 + 16 + 8 + 4 + 2 + 1 = 255` additions

So per block:

```text
256 + 255 = 511 operations
```

For all blocks:

```text
125 * 511 = 63875
```

### D. Synchronizations per block

The improved shared-memory reduction does:

- `1` synchronization after loading to shared memory
- `log2(256) = 8` synchronizations inside the reduction loop

Total:

```text
1 + 8 = 9
```
