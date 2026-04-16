# Lab 7 Notes: Parallel Scan

## Title / Objective

Implement a hierarchical parallel scan for a 1D list using the work-efficient scan pattern from lecture.

## What I Built

- A block-level shared-memory scan kernel based on the Brent-Kung upsweep/downsweep structure.
- A kernel that adds scanned block sums back into each scanned block segment.
- Host code that launches:
  - the per-block scan
  - the scan of the auxiliary block-sum array
  - the final add-back kernel
- Boundary-safe handling for input lengths that are not multiples of the block segment size.

## Host Flow

1. Read the input list from disk.
2. Allocate device input and output buffers.
3. Copy the input list to the GPU.
4. Set `blockDim = BLOCK_SIZE` and `gridDim = ceil(N / (2 * BLOCK_SIZE))`.
5. If there is more than one block:
   - allocate `deviceBlockSums`
   - allocate `deviceScannedBlockSums`
6. Launch the first scan kernel on the full input.
7. If needed, launch the same scan kernel again on the block-sum array using one block.
8. Launch the add-back kernel to add scanned block offsets to the already scanned segments.
9. Synchronize and copy the final output back to the host.
10. Free all device memory.

## Kernel / Algorithm Flow

### Block-level scan kernel

- Each block processes `2 * blockDim.x` elements.
- Each thread loads up to two input values into shared memory.
- Out-of-range loads are replaced with the identity value `0`.
- Run the reduction tree (upsweep):
  - `stride = 1, 2, 4, ...`
- Save the total block sum into the auxiliary array.
- Reset the root to `0`.
- Run the distribution tree (downsweep):
  - `stride = blockDim.x, blockDim.x / 2, ...`
- Convert the exclusive shared-memory result back into inclusive output by adding the original input values.

### Add-back kernel

- Read the scanned block-sum offset for each block except block 0.
- Add that offset to both elements owned by each thread.

## Important Lecture Connections

- [Lecture 18](lectures_md/L18_Parallel_Computation_Patterns_Parallel_Scan.md)
- [Lecture 19](lectures_md/L19_GPU_Systems_Architecture.md)
- [Lecture 15](lectures_md/L15_Parallel_Computation_Patterns_Reduction_Trees.md)

### Most relevant ranges

- Lecture 18:
  - scan definitions
  - Kogge-Stone vs Brent-Kung
  - three-kernel hierarchical scan
  - arbitrary-length scan with block sums
- Lecture 19:
  - system-level bandwidth thinking
  - why scan is memory-bound in practice

## Core Patterns / Formulas

- Block segment size:
  - `2 * BLOCK_SIZE`
- Grid size:
  - `ceil(numElements / (2 * BLOCK_SIZE))`
- Upsweep index:
  - `index = (threadIdx.x + 1) * stride * 2 - 1`
- Downsweep index:
  - same index formula, with stride halving each iteration
- Auxiliary array size:
  - `numBlocks`

## Common Bugs / Edge Cases

- Forgetting that each block scans `2 * BLOCK_SIZE` elements, not `BLOCK_SIZE`.
- Using uninitialized shared memory for the last partial block.
- Forgetting to store the block total before resetting the root for downsweep.
- Mixing inclusive and exclusive logic by writing the downsweep result directly without adding back the original input.
- Forgetting that block 0 should not receive any add-back offset.
- Forgetting that the auxiliary block-sum scan must run with one block only in the simple hierarchical approach.

## Quiz / Exam Takeaways

- Brent-Kung scan is work-efficient, so one full segmented launch performs `O(P)` arithmetic work over the whole input.
- A block with `B` threads usually handles `2B` elements, so the synchronization count in Brent-Kung is based on `2B`.
- Kogge-Stone and Brent-Kung differ mainly in latency vs total work.
- For Kogge-Stone, branch divergence at stride `s` depends on whether a warp crosses the threshold `threadIdx.x = s`.
- In hierarchical scan problems, count not only recursion depth but also how many grid launches each level needs.

## Terminology Used in This Lab

- Scan
- Prefix sum
- Inclusive scan
- Exclusive scan
- Segmented scan
- Brent-Kung scan
- Kogge-Stone scan
- Upsweep
- Downsweep
- Identity value
- Block sums
- Hierarchical scan
