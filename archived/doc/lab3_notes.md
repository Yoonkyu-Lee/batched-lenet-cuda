# Lab 3 Notes: Tiled Matrix Multiplication

## Title / Objective

Implement a tiled dense matrix multiplication kernel using shared memory.

## What I Built

- A shared-memory tiled matrix multiplication kernel.
- Tile loading with zero-padding for boundary safety.

## Host Flow

1. Read matrices `A` and `B`.
2. Set output dimensions.
3. Allocate host output and device buffers.
4. Copy inputs to the GPU.
5. Configure a 2D grid and block using `TILE_WIDTH`.
6. Launch the tiled kernel.
7. Copy output back to the host.
8. Free host and device memory.

## Kernel / Algorithm Flow

- Each block computes one output tile.
- Load one tile of `A` and one tile of `B` into shared memory.
- Synchronize after load.
- Accumulate partial dot products for the current tile.
- Synchronize before the next tile overwrite.
- Repeat over all tiles in the `k` dimension.

## Important Lecture Connections

- [Lecture 5](lectures_md/L05_CUDA_Memory_Model.md)
- [Lecture 6](lectures_md/L06_Data_Locality_and_Tiled_Matrix_Multiply.md)

## Core Patterns / Formulas

- Tile count:
  - `(numAColumns - 1) / TILE_WIDTH + 1`
- Shared tiles:
  - `__shared__ float A_s[TILE_WIDTH][TILE_WIDTH];`
  - `__shared__ float B_s[TILE_WIDTH][TILE_WIDTH];`

## Common Bugs / Edge Cases

- Forgetting one of the two `__syncthreads()` calls per tile.
- Loading garbage instead of zero for boundary elements.
- Using the wrong dimension in the tile loop count.

## Quiz / Exam Takeaways

- Be able to count global-memory reads and writes under tiling.
- Understand why all threads in a block must reach `__syncthreads()`.
- Explain how shared memory improves data reuse.

## Terminology Used in This Lab

- Shared memory
- Tiling
- Data locality
- Barrier
- Tile reuse
- Memory-bound kernel
