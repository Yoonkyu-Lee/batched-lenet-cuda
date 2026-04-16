# Lab 1 Quiz Notes

Detailed takeaways are summarized in [doc/lab1_notes.md](../doc/lab1_notes.md).

## Question 1: Moving Data with CUDA

### Problem statement

Which CUDA call correctly copies `50500` `int32_t` values from host array `h_A` to device array `d_A`, and stores the return value in `err`?

## Answer

```cpp
cudaError_t err = cudaMemcpy(d_A, h_A, 50500 * sizeof(int32_t), cudaMemcpyHostToDevice);
```

## Short solution

- The return type of `cudaMemcpy` is `cudaError_t`.
- The destination is `d_A`, the source is `h_A`.
- The size must be in **bytes**, so we use:

```text
50500 * sizeof(int32_t)
```

- The direction is host to device:

```text
cudaMemcpyHostToDevice
```

### Common trap

The most common mistakes are:

- swapping source and destination
- passing the element count instead of the byte count
- using `cudaMemcpyDeviceToHost` when the copy is actually host to device

## Question 2: Correct Grid Size

### Problem statement

Suppose the input vectors have `N` elements and each block has `32` threads. Which expressions correctly compute the grid dimension?

## Answer

**Correct choices: (d), (e), (f)**

```cpp
dim3 gridDim((N - 1) / 32 + 1);
dim3 gridDim((N + 31) / 32);
dim3 gridDim(ceil(float(N) / float(32)));
```

## Short solution

We need:

```text
ceil(N / 32)
```

blocks.

- `(d)` is the standard integer-ceiling form.
- `(e)` is another equivalent integer-ceiling form.
- `(f)` uses floating-point `ceil`, so it is also correct.
- `(a)` and `(b)` round down.
- `(c)` overestimates when `N` is already a multiple of `32`.

### Fast mental model

For CUDA 1D kernels, the grid size formula to memorize is:

```cpp
(N + blockSize - 1) / blockSize
```

It is the integer version of:

```text
ceil(N / blockSize)
```

## Question 3: Thread Indexing for a Vector Addition Kernel

### Problem statement

Each thread computes `8` adjacent elements. Which expression correctly maps thread and block indices to the first array index handled by a thread?

## Answer

```cpp
idx = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
```

## Short solution

The linear thread index is:

```cpp
blockIdx.x * blockDim.x + threadIdx.x
```

If each thread handles `8` consecutive elements, the first element handled by that thread is:

```cpp
linear_thread_id * 8
```

So the correct formula is:

```cpp
idx = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
```

### Why the `* 8` is outside

First compute **which thread this is globally**, then multiply by the number of elements handled by each thread.

That is:

```text
global_thread_id -> starting element of that thread's chunk
```

If the multiplication is placed in the wrong location, the mapping no longer stays contiguous across blocks.

## Question 4: Which are wrong?

### Problem statement

Which of the following statements are wrong?

## Answer

**Wrong choice: (b) only**

## Short solution

- `(a)` is correct: `threadIdx` and `blockIdx` are built-in read-only variables.
- `(b)` is wrong: CPU execution and GPU execution **can** overlap in general.
- `(c)` was accepted as correct in the quiz context.
- `(d)` is correct: a normal C program can be viewed as a CUDA program containing only host code.

The key idea is that CUDA does **not** mean CPU and GPU must always run strictly one after the other.

### Exam takeaway

If a statement says CPU and GPU execution can never overlap, it is too strong and is usually false.

## Question 5: Global Memory Read by Vector Addition Kernel

### Problem statement

In terms of vector length `N`, how many Bytes are read from global memory by the vector addition kernel?

## Answer

**`8N` Bytes**

## Short solution

Vector addition reads:

- `N` elements from input vector `A`
- `N` elements from input vector `B`

That is:

```text
2N floats
```

If each element is `4` bytes:

```text
2N * 4 = 8N Bytes
```

### Memory traffic pattern

For a float vector addition kernel:

- read `A[i]`
- read `B[i]`
- write `C[i]`

So the read traffic is `8N` Bytes, and the write traffic would be:

```text
4N Bytes
```

## Question 6: Thread Indexing for a Vector Addition Kernel II

### Problem statement

Each block processes `5 * blockDim.x` consecutive elements split into `5` contiguous sections. All threads process the first section together, then the second section together, and so on. Which expression gives the array index of the **second** element processed by a thread?

## Answer

```cpp
idx = blockIdx.x * blockDim.x * 5 + blockDim.x + threadIdx.x;
```

## Short solution

For one block:

- the block starts at:

```cpp
blockIdx.x * blockDim.x * 5
```

- the second section begins after one full section of size `blockDim.x`, so add:

```cpp
blockDim.x
```

- inside that section, each thread takes its own offset:

```cpp
threadIdx.x
```

So:

```cpp
idx = blockIdx.x * blockDim.x * 5 + blockDim.x + threadIdx.x;
```

### Pattern to remember

When a block processes several contiguous sections:

```text
block base + section offset + thread offset
```

In this problem:

- block base = `blockIdx.x * blockDim.x * 5`
- second-section offset = `blockDim.x`
- thread offset = `threadIdx.x`

## One-Page Summary

- `cudaMemcpy(dst, src, bytes, direction)` must match both the copy direction and byte count.
- 1D CUDA indexing starts from:

```cpp
blockIdx.x * blockDim.x + threadIdx.x
```

- Grid size usually uses integer ceiling:

```cpp
(N + blockSize - 1) / blockSize
```

- If each thread processes multiple adjacent elements, compute the global thread id first, then scale by the chunk length.
- For float vector addition, each output element causes:
  - `8` bytes of reads
  - `4` bytes of writes
- Lab 1 is mostly about getting the host/device flow and indexing logic correct before worrying about deeper optimization.
