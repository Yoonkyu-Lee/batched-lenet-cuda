# ECE408 Glossary

## Atomic

- **Meaning:** An atomic operation is executed as one indivisible update on a memory location.
- **Why it matters:** Needed when multiple threads may update the same value, such as a histogram bin.
- **Example:** `atomicAdd(&hist[42], 1);`
- **Common pitfall:** Only the atomic operation is protected; the surrounding logic is not automatically atomic.

## Serialized

- **Meaning:** Operations are forced to happen one at a time instead of in parallel.
- **Why it matters:** Atomic updates to the same address often serialize.
- **Example:** 100 threads updating the same histogram bin behave like 100 people waiting in one line.
- **Common pitfall:** Correctness may be preserved while performance collapses.

## Contention

- **Meaning:** Many threads compete for the same hardware resource or memory location.
- **Why it matters:** Contention is the root cause of many atomic slowdowns.
- **Example:** Many grayscale pixels mapping to `hist[128]`.
- **Common pitfall:** A kernel may look memory-bound when the real bottleneck is contention.

## Coalesced Access

- **Meaning:** Neighboring threads access neighboring global-memory addresses so the GPU can combine requests efficiently.
- **Why it matters:** Coalesced loads and stores use bandwidth much better.
- **Example:** thread 0 reads `buffer[0]`, thread 1 reads `buffer[1]`, thread 2 reads `buffer[2]`.
- **Common pitfall:** Accesses that are contiguous inside one thread may still be uncoalesced across a warp.

## Grid-Stride Loop

- **Meaning:** Each thread processes index `i`, then `i + stride`, then `i + 2 * stride`, with `stride = blockDim.x * gridDim.x`.
- **Why it matters:** Lets one kernel handle arbitrary input size cleanly.
- **Example:** `for (int i = idx; i < N; i += stride) { ... }`
- **Common pitfall:** Coalescing should be judged within one loop iteration, not across all iterations together.

## Shared Memory

- **Meaning:** Fast on-chip memory shared by all threads in a block.
- **Why it matters:** Useful for tiling and reduction because it is much faster than global memory.
- **Example:** `__shared__ float input_s[BLOCK_SIZE];`
- **Common pitfall:** Shared memory is block-local and does not solve cross-block communication.

## Reduction

- **Meaning:** Combine many input values into one value, such as sum, min, max, or product.
- **Why it matters:** Reduction is a core parallel pattern and the basis of Lab 6.
- **Example:** Summing all elements of an array.
- **Common pitfall:** Parallel reduction assumes the operator can be safely regrouped.

## Scan / Prefix Sum

- **Meaning:** Compute running totals across an array.
- **Why it matters:** Used for CDF computation in Lab 5 and a fundamental parallel primitive.
- **Example:** `[2, 3, 5, 1] -> [2, 5, 10, 11]`
- **Common pitfall:** Mixing up inclusive and exclusive scan.

## Inclusive Scan

- **Meaning:** A scan where output element `i` includes input element `i`.
- **Why it matters:** Lab 7 expects the running sum form, not the shifted exclusive form.
- **Example:** `[3, 6, 7, 4] -> [3, 9, 16, 20]`
- **Common pitfall:** Writing exclusive shared-memory code and forgetting to convert it to inclusive output.

## Exclusive Scan

- **Meaning:** A scan where output element `i` contains the prefix sum before input element `i`.
- **Why it matters:** Many work-efficient scan trees naturally produce exclusive form internally.
- **Example:** `[3, 6, 7, 4] -> [0, 3, 9, 16]`
- **Common pitfall:** Comparing an exclusive intermediate result against an inclusive expected output.

## Coarsening

- **Meaning:** Assign more work to each thread so each thread handles multiple input elements.
- **Why it matters:** Can reduce overhead and the number of blocks.
- **Example:** One thread reducing four values instead of one.
- **Common pitfall:** Too much coarsening can raise register pressure and hurt occupancy.

## Divergence

- **Meaning:** Threads in the same warp follow different control-flow paths.
- **Why it matters:** Divergence reduces SIMD efficiency.
- **Example:** `if (threadIdx.x % stride == 0)` in naive reduction.
- **Common pitfall:** Boundary checks and modulo-based conditions often create hidden divergence.

## Kogge-Stone Scan

- **Meaning:** A low-latency scan algorithm that propagates prefix information every `stride = 1, 2, 4, ...`.
- **Why it matters:** It is a common scan baseline and appears heavily in Lab 7 quiz analysis.
- **Example:** Each thread reads a value `stride` positions to the left and accumulates it.
- **Common pitfall:** It is not work-efficient; total work is `O(n log n)`.

## Brent-Kung Scan

- **Meaning:** A work-efficient scan algorithm built from an upsweep tree followed by a downsweep tree.
- **Why it matters:** This is the scan style most directly connected to the Lab 7 implementation.
- **Example:** One block scans `2 * BLOCK_SIZE` elements using a reduction phase and a distribution phase.
- **Common pitfall:** Forgetting to save the block sum before resetting the root breaks the second phase.

## Upsweep

- **Meaning:** The reduction phase of a work-efficient scan tree that builds partial sums toward the root.
- **Why it matters:** The block total is produced at the end of upsweep.
- **Example:** `stride = 1, 2, 4, ...` with `T[index] += T[index - stride]`
- **Common pitfall:** Treating the upsweep result as the final inclusive scan output.

## Downsweep

- **Meaning:** The distribution phase of a work-efficient scan tree that pushes prefix values back toward the leaves.
- **Why it matters:** It converts the partial sums from upsweep into per-element prefix information.
- **Example:** Swap-and-add style updates while stride shrinks from large to small.
- **Common pitfall:** Forgetting to reset the root to the identity value before downsweep.

## Block Sums

- **Meaning:** One auxiliary value per block storing the total of that block's segment.
- **Why it matters:** Hierarchical scan uses block sums to stitch local scans into a full global scan.
- **Example:** `blockSums[blockIdx.x] = segment_total`
- **Common pitfall:** Scanning blocks locally without a second phase leaves the global scan incomplete.

## Hierarchical Scan

- **Meaning:** A large-array scan built from local block scans, a scan of block sums, and an add-back phase.
- **Why it matters:** CUDA blocks cannot globally synchronize inside one kernel, so large scans need hierarchy.
- **Example:** `scan -> scan block sums -> add scanned sums`
- **Common pitfall:** Counting only recursion depth and forgetting that each level may require many grid launches.

## Synchronization

- **Meaning:** A point where all threads in a block must wait for each other.
- **Why it matters:** Necessary when threads read shared memory values written by other threads.
- **Example:** `__syncthreads();`
- **Common pitfall:** Calling `__syncthreads()` from only some threads in a block is invalid.

## Privatization

- **Meaning:** Use multiple private copies of an output instead of one shared output.
- **Why it matters:** Reduces contention in histogram-style kernels.
- **Example:** One shared-memory histogram per block, then merge into one global histogram.
- **Common pitfall:** Privatization reduces contention, but the final merge still has a cost.

## Segmented Reduction

- **Meaning:** Split the input into segments and let each block reduce one segment.
- **Why it matters:** CUDA blocks cannot globally synchronize inside a kernel.
- **Example:** Each block reduces `2 * BLOCK_SIZE` inputs and writes one partial sum.
- **Common pitfall:** A second phase is still needed to combine block outputs.

## Identity Value

- **Meaning:** The neutral element of an operator.
- **Why it matters:** Out-of-range values in a reduction can be padded with the identity.
- **Example:** For sum reduction, the identity is `0`.
- **Common pitfall:** Using the wrong identity silently corrupts boundary cases.

## Reduction Tree

- **Meaning:** A staged pattern that repeatedly halves the number of active values.
- **Why it matters:** It is the core structure of an efficient block-level reduction.
- **Example:** `stride = blockDim.x / 2; stride >>= 1`
- **Common pitfall:** Naive trees often cause unnecessary divergence and uncoalesced access.

## Partial Sum

- **Meaning:** An intermediate result produced by one thread or one block.
- **Why it matters:** Partial sums bridge per-thread work and the final reduced value.
- **Example:** `output[blockIdx.x]` in Lab 6.
- **Common pitfall:** Partial sums still need to be reduced again.

## Warp Shuffle

- **Meaning:** Warp-level register exchange using instructions like `__shfl_down_sync`.
- **Why it matters:** Can finish the last warp of a reduction without shared memory or block-wide barriers.
- **Example:** `partial += __shfl_down_sync(0xffffffff, partial, stride);`
- **Common pitfall:** Warp shuffle works only within a warp, not across the whole block.
