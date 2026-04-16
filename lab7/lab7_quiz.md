# Lab 7 Quiz Notes

## Question 1: Brent-Kung Scan Analysis

The questions below refer to the Brent-Kung scan discussed in lecture and implemented in the lab. For all questions, a large input array of `P` floats is scanned using a grid with `B` threads in each thread block (`BLOCK_SIZE = B`, and `B << P`). The operation used is floating-point addition. Note that `log` means logarithm base 2.

### A

How many floating-point operations are executed by all threads in a single grid launch?

Choices:

- `O(P + B log B)`
- `O(P log P + B log B)`
- `O(P)`
- `O(B log P)`
- `O(B log B)`

Answer:

- `O(P)`

Explanation:

- In the Brent-Kung block scan, one thread block with `B` threads handles `2B` elements.
- A work-efficient scan on `2B` elements performs `O(B)` total additions per block.
- The whole input of length `P` needs about `P / (2B)` blocks.
- Total work over the whole grid launch is therefore:

```text
O(B) * O(P / (2B)) = O(P)
```

- So the correct answer is `O(P)`.

### B

In terms of `B` and `P`, how many times does thread block 0 perform barrier synchronization in one grid launch?

Choices:

- `2 log B`
- `log(4B)`
- `log(P / B)`
- `2 log(2B)`
- `2 B`

Answer:

- `2 log(2B)`

Explanation:

- Brent-Kung scan on a block segment of `2B` elements has two phases:
  - reduction tree
  - post-scan / distribution tree
- Each phase takes `log(2B)` parallel steps.
- There is one block-wide barrier synchronization per step.
- So the total number of barrier synchronizations for a thread block in one launch is:

```text
log(2B) + log(2B) = 2 log(2B)
```

- Therefore the correct answer is `2 log(2B)`.

## Quick Takeaway

- Brent-Kung scan is work-efficient, so total arithmetic work over the whole grid is `O(P)`.
- A block with `B` threads typically scans `2B` elements.
- Because Brent-Kung has an upsweep and a downsweep, synchronization count is based on both phases, giving `2 log(2B)`.

## Question 2: Scan Statement T/F

Which of the following statements are `TRUE`?

Choices:

- (a) A Kogge-Stone kernel with double-buffer strategy will have fewer `__syncthreads()` but double the shared memory usage.
- (b) Considering an exclusive scan operation on an input array of type `int`, the output vector may differ between a correct CPU implementation and a correct GPU implementation.
- (c) The overhead between different kernel launches in a complete hierarchical scan is significant.
- (d) Because the Brent-Kung algorithm performs fewer operations compared to the Kogge-Stone algorithm with the same input vector, the Brent-Kung algorithm is a more popular choice in energy-constrained tasks.

Answer:

- `(a)`, `(c)`, and `(d)` are true.

Explanation:

### (a) True

- In Kogge-Stone, if we use double buffering, we can separate the read buffer and write buffer.
- That removes the false dependence caused by reading and writing the same shared-memory array in one iteration.
- As a result, one synchronization can be removed from each iteration.
- The tradeoff is that shared memory usage becomes roughly 2x because we keep two buffers.

### (b) False

- The operator here is integer addition on `int`.
- Integer addition is associative and deterministic for the same exact values when overflow is not part of the intended behavior.
- So a correct CPU exclusive scan and a correct GPU exclusive scan should produce the same output vector.
- This “may differ” statement is usually relevant for floating-point arithmetic, not integer scan.

### (c) True

- A hierarchical scan uses multiple kernel launches:
  - per-block scan
  - scan of block sums
  - add scanned block sums
- The lecture explicitly points out kernel-launch structure and memory traffic as meaningful overhead in complete scan pipelines.
- So the overhead between launches is important enough to be discussed as a performance issue.

### (d) True

- Brent-Kung is work-efficient, while Kogge-Stone does more total operations.
- Fewer arithmetic operations usually imply less wasted work and better energy efficiency.
- The tradeoff is that Brent-Kung takes more steps, but if energy is the concern, lower total work makes it an attractive choice.

## Quick Takeaway for Q2

- Double buffering in Kogge-Stone trades more shared memory for fewer barriers.
- Integer scan should match exactly across correct CPU and GPU implementations.
- Hierarchical scan pays nontrivial multi-kernel overhead.
- Brent-Kung is attractive when total work matters more than minimum latency.

## Question 3: Kogge-Stone Scan Analysis

The questions below refer to the Kogge-Stone scan discussed in lecture, which is not the approach that you implemented in the lab. For all questions, an input array of `P` floats is scanned using a grid with `B` threads in each thread block (`BLOCK_SIZE = B`, and `B << P`). The operation used is floating-point addition. Note that `log` means logarithm base 2.

### A

How many floating-point operations are executed by all threads in a single grid launch?

Choices:

- `O(B log B)`
- `O(log P + B log B)`
- `O(P log B)`
- `O(B log P)`
- `O(B log P + P)`

Answer:

- `O(P log B)`

Explanation:

- In Kogge-Stone, one block scans its own local segment of `B` elements.
- That local scan costs `O(B log B)` work per block.
- The full input of size `P` needs about `P / B` blocks.
- So one grid launch performs:

```text
O(B log B) * O(P / B) = O(P log B)
```

- Therefore the correct answer is `O(P log B)`.

### B

If a GPU were able to launch a grid with `P` threads in a thread block, one kernel launch would suffice to completely scan the array of `P` values, using `O(P log P)` floating-point operations. Choose the statement that best explains the relationship between this fact and the correct answer to Part A above.

Choices:

- The need to make use of additional barrier synchronizations in subsequent grid launches explains the difference in computation between the two answers.
- The answer to Part A, which is larger than necessary, reflects the work inefficiency of the Kogge-Stone scan.
- Launching the kernel more than once puts additional pressure on the memory system, which leads to the differences in the answers.
- With `B << P`, one grid does not finish the work, thus fewer than `P log P` operations are needed in Part A.
- Splitting the work amongst thread blocks always leads to inefficiencies in computation, and the answer to Part A reflects that fact.

Answer:

- With `B << P`, one grid does not finish the work, thus fewer than `P log P` operations are needed in Part A.

Explanation:

- Part A counts only one grid launch.
- In that one launch, each block only scans a local segment of length `B`.
- So the cost is `O(P log B)`, not `O(P log P)`.
- The difference is not because of synchronization or memory pressure. It is because a single segmented-scan launch does not yet complete the full global scan.

### C

Now assume that `B = 512` and that each warp contains 32 threads. How many warps in thread block have branch/control divergence during the iterations in which stride is `8`?

Answer:

- `1`

Explanation:

- In Kogge-Stone, the branch condition is effectively:

```text
threadIdx.x >= stride
```

- With `stride = 8`, only warp 0 contains both:
  - threads `0..7` that do not enter
  - threads `8..31` that do enter
- Every later warp has all thread indices greater than or equal to `8`, so they all follow the same path.
- Therefore only `1` warp diverges.

### D

Now assume that `B = 256` and that each warp contains 32 threads. How many warps in thread block have branch/control divergence during the iterations in which stride is `64`?

Answer:

- `0`

Explanation:

- With `stride = 64`, the condition is again `threadIdx.x >= 64`.
- Warp 0 (`0..31`) is entirely false.
- Warp 1 (`32..63`) is entirely false.
- Warp 2 (`64..95`) and all later warps are entirely true.
- No warp contains a mix of true and false threads, so there is no branch divergence.

## Quick Takeaway for Q3

- A segmented Kogge-Stone launch over the whole input costs `O(P log B)`.
- That is smaller than `O(P log P)` because one launch only scans block-local segments.
- Divergence depends on whether a single warp straddles the branch threshold `threadIdx.x = stride`.

## Question 4: Large Scan Hierarchy

Suppose we need to perform an inclusive scan on an input of `2^42` elements. A customized GPU supports at most `2^9` blocks per grid and at most `2^12` threads per block. Using Brent-Kung in a hierarchical fashion, with none of the scan work done by the host, how many times do we need to launch the Brent-Kung kernel? Please answer the exact number.

Answer:

- `1048706`

Explanation:

- The key detail is that the question asks how many times we launch the **Brent-Kung kernel**, and the GPU can launch at most `2^9` blocks per grid.
- So we must count not only the recursion levels, but also how many separate grid launches are needed at each level.

- One Brent-Kung block with `B = 2^12` threads scans:

```text
2B = 2^13 elements
```

- Since a grid can have at most `2^9` blocks, one Brent-Kung kernel launch can cover at most:

```text
2^9 * 2^13 = 2^22 elements
```

- Now count launches level by level.

### Level 0: scan the original input

- Input size: `2^42`
- One launch covers: `2^22`
- Required launches:

```text
2^42 / 2^22 = 2^20 launches
```

- After all these launches, the total block-sum array size is:

```text
2^42 / 2^13 = 2^29 block sums
```

### Level 1: scan the block sums of size `2^29`

- One launch still covers `2^22` elements
- Required launches:

```text
2^29 / 2^22 = 2^7 launches
```

- This produces a new block-sum array of size:

```text
2^29 / 2^13 = 2^16
```

### Level 2: scan the block sums of size `2^16`

- `2^16 < 2^22`, so this fits in one grid launch.
- Required launches: `1`

- This produces a new block-sum array of size:

```text
2^16 / 2^13 = 2^3
```

### Level 3: scan the block sums of size `2^3`

- `2^3` elements fit in one Brent-Kung block.
- Required launches: `1`

### Total Brent-Kung kernel launches

```text
2^20 + 2^7 + 1 + 1
= 1,048,576 + 128 + 2
= 1,048,706
```

- Therefore the exact answer is:

```text
1048706
```

## Quick Takeaway for Q4

- First compute how many elements one launch can cover: `max_blocks * 2B`.
- Then count **launches per level**, not just the number of recursion levels.
- Here the dominant term is the top level: `2^20` launches just to scan the original `2^42` elements.
