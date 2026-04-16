# Lab 5 Quiz Notes

Detailed takeaways are summarized in [doc/lab5_notes.md](../doc/lab5_notes.md).

## Question 1: Debugging Kernel II

### Problem statement

The code below is meant to insert up to `CAPACITY` integers into `array`. Remaining integers are handled with global memory operations (not shown in the code).

```cpp
__shared__ unsigned int array[CAPACITY];
__shared__ unsigned int count;
unsigned int insertIdx;

// count is initialized to 0 before threads in a block
// execute the code below (many times)

if (CAPACITY > count) {
    // value is the number to be written into array
    insertIdx = atomicAdd(&count, 1);
    array[insertIdx] = value;
} else {
    // Array is full - do something else (omitted).
}
```

Unfortunately, the code does not work as intended. Which of the following **correctness** problems occurs in the code?

1. The length of `array` is incorrect: an element must be reserved at the end to avoid accidental out-of-bounds memory writes if overflow occurs.
2. The `insertIdx` variable should be initialized to `0`.
3. The `count` may exceed `CAPACITY`, causing future executions of the `if` statement's test to execute incorrectly.
4. Although reading the `count` may provide a performance boost, the code shown may write beyond the end of `array`.
5. When `array` overflows, a thread should write some of the elements from `array` back to global memory so that future insertions to `array` can succeed.

## Inferred answer

**Answer: 4**

> Although reading the `count` may provide a performance boost, the code shown may write beyond the end of `array`.

## Explanation

The bug is a classic **check-then-act race condition**.

The code first checks:

```cpp
if (CAPACITY > count)
```

but that check is **not atomic with** the later increment:

```cpp
insertIdx = atomicAdd(&count, 1);
```

So multiple threads can observe the same old value of `count` before any of them perform the atomic increment.

### Example

Assume:

- `CAPACITY = 10`
- `count = 9`

Now two threads run:

1. Thread A reads `count == 9`, so `CAPACITY > count` is true.
2. Thread B also reads `count == 9`, so the condition is also true.
3. Thread A executes `atomicAdd(&count, 1)` and gets `insertIdx = 9`.
4. Thread B executes `atomicAdd(&count, 1)` and gets `insertIdx = 10`.
5. Thread B writes `array[10] = value`, which is **out of bounds** because valid indices are `0` through `9`.

So the real correctness issue is not just that `count` changes, but that the code may actually perform an **invalid write past the end of the shared array**.

## Why the other choices are wrong

### 1. The length of `array` is incorrect

This is not the real fix. The array size `CAPACITY` is conceptually correct.  
The problem is the race between the non-atomic test and the atomic increment. Adding an extra slot would only hide one overflow case and would not solve the underlying correctness bug.

### 2. `insertIdx` should be initialized to 0

No. `insertIdx` is assigned by:

```cpp
insertIdx = atomicAdd(&count, 1);
```

so it does not need prior initialization.

### 3. `count` may exceed `CAPACITY`

This can happen, but it is not the most important correctness bug being asked about.  
The more direct failure is that an out-of-range index can be produced and used to write beyond `array`.

### 5. Spill data back to global memory

That may be part of a larger design, but it is not the correctness issue in the shown code.  
Even if overflow handling exists elsewhere, the code here is already unsafe because it can write out of bounds before overflow handling takes effect.

## Key takeaway

If the decision to insert depends on the current value of `count`, then the reservation of a slot must be done in a way that is consistent with that decision. In other words:

- checking capacity and
- claiming an insertion index

must be coordinated atomically, or the kernel can overflow the shared array.

## Question 2: Cached Atomic Operation Throughput

### Problem statement

For a processor that supports atomic operations in L2 cache, assume that each atomic operation takes `4 ns` to complete in L2 cache and `100 ns` to complete in DRAM. Assume that `90%` of the atomic operations hit in L2 cache. What is the approximate throughput for atomic operations on the same global memory variable?

Choices:

- (a) `0.0735G` atomic operations per second
- (b) `0.25G` atomic operations per second
- (c) `2.5G` atomic operations per second
- (d) `100G` atomic operations per second
- (e) None of these are correct

## Inferred answer

**Answer: (c) `2.5G` atomic operations per second**

## Explanation

Because the question asks about atomics on the **same global memory variable**, we assume the operations are effectively **serialized**. That means the throughput is approximately:

```text
throughput = 1 / (average time per atomic)
```

### Step 1: Compute the average latency

`90%` hit in L2:

```text
0.9 * 4 ns = 3.6 ns
```

`10%` go to DRAM:

```text
0.1 * 100 ns = 10 ns
```

So the average time per atomic is:

```text
3.6 ns + 10 ns = 13.6 ns
```

### Step 2: Convert latency to throughput

```text
throughput = 1 / 13.6 ns
           = 1 / (13.6 x 10^-9 s)
           ≈ 73.5 x 10^6 operations/s
           = 0.0735G operations/s
```

So the computed result is:

**`0.0735G` atomic operations per second**

That corresponds to:

**Answer: (a) `0.0735G` atomic operations per second**

## Important note

There is a mismatch between the arithmetic result and the choice label shown above in the draft answer section.  
The correct calculation gives:

- average latency = `13.6 ns`
- throughput = about **`0.0735G ops/s`**

So the mathematically correct answer is actually:

**Answer: (a), not (c)**

## Why the other choices are wrong

### (b) `0.25G`

This would correspond to about `4 ns` per atomic on average, which would only make sense if essentially all atomics hit in L2 and there were no DRAM misses.

### (c) `2.5G`

This corresponds to about `0.4 ns` per atomic, which is much too fast given the stated latencies.

### (d) `100G`

This corresponds to `0.01 ns` per atomic, which is clearly impossible under the given assumptions.

### (e) None of these are correct

This would only be true if `0.0735G` were not listed. But it is listed as choice (a).

## Key takeaway

For atomics targeting the same memory location:

- contention causes serialization
- throughput is limited by the average latency per atomic
- even a small fraction of slow DRAM atomics can significantly reduce throughput

## Question 3: Atomic Operation Count in Histogram

### Problem statement

For the histogram kernel with privatization, how many atomic operations are being performed by your kernel with:

- `N = 1024` (input size)
- `NUM_BINS = 256` (histogram length)
- `NUM_BLOCKS = 8`
- `BLOCK_SIZE = 128`

Consider atomic operations to both **shared memory** and **global memory**.

## Inferred answer

**Answer: `3072`**

## Explanation

In a privatized histogram kernel, atomic operations typically happen in two places:

1. **Shared-memory atomics**
   - Each input element contributes to exactly one histogram bin.
   - So we perform one atomic update to a block-private shared histogram per input element.
   - That gives:

```text
N = 1024 shared-memory atomic operations
```

2. **Global-memory atomics**
   - After each block finishes building its private histogram, it merges all `NUM_BINS` bins into the global histogram.
   - That means each block performs one global atomic add per bin.
   - So:

```text
NUM_BLOCKS * NUM_BINS = 8 * 256 = 2048 global-memory atomic operations
```

### Total atomic operations

```text
total = shared atomics + global atomics
      = 1024 + 2048
      = 3072
```

## Why this is the right counting method

The important idea is:

- privatization does **not** remove atomic operations completely
- it moves most of the updates from **global memory** to **shared memory**
- then it performs a smaller number of global atomic updates during the merge step

So instead of doing `1024` global atomics directly on the final histogram, the kernel does:

- `1024` shared atomics for the input updates
- `2048` global atomics for the merge from 8 private histograms

## Key takeaway

Privatization reduces **contention on global memory**, but it may increase the **total number of atomic operations**.  
The performance win comes from replacing many expensive global atomics with cheaper shared-memory atomics.

## Question 4: Debugging Kernel I

### Problem statement

Consider the following private histogram kernel. The function parameter `size` is the number of elements in `buffer`, and `histo` is the histogram of size `HIST_SIZE` properly allocated on the device. All elements in `buffer` have a value less than `HIST_SIZE`.

```cpp
#define HIST_SIZE 1200
__global__ void histo_kernel(uint16_t *buffer, int32_t size, uint32_t *histo) {
    __shared__ uint32_t histo_private[HIST_SIZE];
    for (int start = 0; start < HIST_SIZE; start += blockDim.x) {
        histo_private[start + threadIdx.x] = 0;
        __syncthreads();
    }
    int32_t i = threadIdx.x + blockIdx.x * blockDim.x;
    int32_t stride = blockDim.x * gridDim.x;
    while (i < size) {
        atomicAdd(histo_private + buffer[i], 1);
        i += stride;
    }
    __syncthreads();
    for (int start = 0; start < HIST_SIZE; start += blockDim.x) {
        atomicAdd(histo + threadIdx.x + start, histo_private[threadIdx.x + start]);
    }
}
```

Which of the following statements are TRUE? Select all that apply.

- (a) We could omit all `__syncthreads` calls for certain block size.
- (b) The access to `buffer` is not coalesced.
- (c) The first `__syncthreads()` should be moved outside of the `for` loop to increase overall parallelism.
- (d) A sorted `buffer` array is likely to decrease the kernel execution time.

If we omit all `__syncthreads()` from the code above, how many different block configurations could guarantee correct outputs?

Consider the block configurations from the previous problem. What is the sum of their number of threads?

## Inferred answers

- **True statements:** **(a)** and **(c)**
- **Number of valid block configurations without `__syncthreads()`:** **15**
- **Sum of their thread counts:** **181**

## Explanation

### Part 1: Which statements are true?

#### (a) True

For some block sizes, the kernel can still be correct even without `__syncthreads()`.

The usual reasoning is:

- if the entire block fits in a **single warp**
- then threads execute in lockstep in a warp-synchronous way
- so the synchronization barriers are not needed for correctness between threads of that one warp

However, that is only safe for certain block sizes, which we count below.

#### (b) False

The access to `buffer` is actually **coalesced**.

Each thread starts at:

```cpp
i = threadIdx.x + blockIdx.x * blockDim.x;
```

and then advances by:

```cpp
stride = blockDim.x * gridDim.x;
```

So within one iteration of the `while` loop, neighboring threads read neighboring `buffer[i]` elements. That is the standard grid-stride access pattern and is coalesced.

#### (c) True

This one is an important bug/performance observation.

Inside the initialization loop:

```cpp
for (int start = 0; start < HIST_SIZE; start += blockDim.x) {
    histo_private[start + threadIdx.x] = 0;
    __syncthreads();
}
```

the barrier is stronger than necessary. We do **not** need a barrier after every chunk of initialization.  
What we need is:

1. all threads finish zeroing their assigned entries
2. then all threads move on to histogram updates

So one barrier **after the loop** is sufficient. Moving it outside the loop improves parallelism and avoids repeated unnecessary synchronization.

#### (d) False

A sorted `buffer` is likely to make performance **worse**, not better.

If neighboring threads see the same or similar values, then many of them will update the same histogram bin at about the same time:

```cpp
atomicAdd(histo_private + buffer[i], 1);
```

That increases atomic contention and serialization. So sorting the input usually does **not** decrease execution time for this histogram kernel.

## Part 2: How many block configurations work if all `__syncthreads()` are removed?

Two conditions are needed.

### Condition 1: The block must fit in one warp

Without `__syncthreads()`, correctness can only be guaranteed when the whole block is executed as one warp:

```text
blockDim.x <= 32
```

### Condition 2: `blockDim.x` must divide `HIST_SIZE`

The initialization and final merge loops do:

```cpp
histo_private[start + threadIdx.x]
histo[threadIdx.x + start]
```

with no bounds check.

So for correctness, the last iteration must not step past `HIST_SIZE - 1`. That means:

```text
blockDim.x must divide HIST_SIZE = 1200
```

### Divisors of 1200 that are <= 32

The valid block sizes are:

```text
1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 16, 20, 24, 25, 30
```

That gives:

```text
15 valid block configurations
```

## Part 3: Sum of thread counts

Now sum those valid block sizes:

```text
1 + 2 + 3 + 4 + 5 + 6 + 8 + 10 + 12 + 15 + 16 + 20 + 24 + 25 + 30 = 181
```

So the final answer is:

```text
181 threads
```

## Key takeaways

- Grid-stride reads from `buffer` are coalesced.
- Repeated `__syncthreads()` inside the initialization loop are unnecessary; one barrier after the loop is enough.
- Removing all synchronization is only safe for special block sizes.
- In histogram kernels, sorted inputs can increase atomic contention.

## Question 5: T/F in Histogram

### Problem statement

Which of the following statements are **TRUE**?

- (a) A certain GPU without L2 cache has an atomic add operation latency of `100 ns` on global memory locations. Then, the maximal throughput we can get for atomic add operations on the same global memory variable is `0.01G` operations per second.
- (b) Privatization improves performance by transforming global atomic operations into shared memory atomic operations.

## Inferred answer

**Answer: both (a) and (b) are true**

## Explanation

### (a) True

If atomic operations target the **same global memory variable**, then they are effectively **serialized**.  
So the best possible throughput is approximately:

```text
throughput = 1 / latency
           = 1 / 100 ns
           = 1 / (100 x 10^-9 s)
           = 10^7 ops/s
           = 0.01G ops/s
```

So statement (a) is correct.

### (b) True

This is the main idea behind histogram privatization.

Instead of having all threads directly update one global histogram:

```cpp
atomicAdd(&global_hist[bin], 1);
```

each block first updates its own **private shared-memory histogram**, and only later merges into the global histogram.

That means many expensive global atomics are replaced by cheaper shared-memory atomics, and the number of contended global atomic operations is reduced.

So statement (b) is also correct.

## Key takeaway

- For atomics on the same location, throughput is limited by the atomic latency.
- Privatization helps because **shared-memory atomics are cheaper** and **global contention is reduced**.
