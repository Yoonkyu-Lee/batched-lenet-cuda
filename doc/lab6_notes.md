# Lab 6 Notes: List Reduction

## Title / Objective

Implement a 1D sum reduction kernel using the improved reduction pattern from lecture.

## What I Built

- A shared-memory reduction kernel that reduces `2 * BLOCK_SIZE` inputs per block.
- Host code that launches the kernel and then finishes the final sum on the CPU.
- Boundary-safe handling for arbitrary input length by treating out-of-range values as `0`.

## Host Flow

1. Read the input list from disk.
2. Compute `numOutputElements = ceil(numInputElements / (2 * BLOCK_SIZE))`.
3. Allocate device input and output buffers.
4. Copy the input list to the GPU.
5. Configure the grid and block.
6. Launch the reduction kernel.
7. Copy block partial sums back to the CPU.
8. Sum those partial sums on the host.
9. Free device memory.

## Kernel / Algorithm Flow

- Compute the segment start:
  - `segment = 2 * blockDim.x * blockIdx.x`
- Each thread loads up to two global-memory values:
  - `input[i]`
  - `input[i + blockDim.x]`
- Add them immediately into one local partial sum.
- Store the partial sum into shared memory.
- Synchronize.
- Run the reduction tree:
  - `for (stride = blockDim.x / 2; stride > 0; stride >>= 1)`
- Thread 0 writes the block result to `output[blockIdx.x]`.

## Important Lecture Connections

- [Lecture 15](lectures_md/L15_Parallel_Computation_Patterns_Reduction_Trees.md)
- [Lecture 16](lectures_md/L16_Advanced_Optimizations_for_Projects.md)

## Core Patterns / Formulas

- Segment start:
  - `segment = 2 * blockDim.x * blockIdx.x`
- Grid size:
  - `ceil(numInputElements / (2 * BLOCK_SIZE))`
- Reduction loop:
  - `for (stride = blockDim.x / 2; stride > 0; stride >>= 1)`
- Per-block additions:
  - `2 * BLOCK_SIZE - 1`

## Common Bugs / Edge Cases

- Forgetting that the last block may be partial.
- Reading `input[i + blockDim.x]` without a bounds check.
- Using a naive `% stride == 0` tree and creating unnecessary divergence.
- Forgetting the synchronization after writing shared memory.
- Assuming blocks can finish the whole reduction without a second phase.

## Quiz / Exam Takeaways

- The average number of unique shared-memory locations accessed per thread in the reduction step tends to `2` as `BLOCK_SIZE` grows.
- In the improved reduction, common false statements involve undercounting additions, undercounting global reads, or claiming synchronization depends on full input size.
- For one kernel launch:
  - global reads scale with input size
  - global writes scale with number of blocks
  - synchronizations per block scale with `1 + log2(BLOCK_SIZE)`
- The improved mapping matters because it preserves coalesced loads and reduces divergence.

## Terminology Used in This Lab

- Reduction
- Segmented reduction
- Identity value
- Shared memory reduction
- Reduction tree
- Divergence
- Synchronization
- Partial sum
- Warp shuffle
- Two-phase reduction
