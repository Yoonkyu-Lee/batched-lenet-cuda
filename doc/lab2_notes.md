# Lab 2 Notes: Basic Matrix Multiplication

## Title / Objective

Implement a basic dense matrix multiplication kernel on the GPU.

## What I Built

- A naive matrix multiplication kernel.
- A 2D launch configuration mapping one thread to one output element.

## Host Flow

1. Read matrices `A` and `B`.
2. Set output dimensions.
3. Allocate host output and device buffers.
4. Copy `A` and `B` to device.
5. Configure a 2D grid and block.
6. Launch the kernel.
7. Copy `C` back to the host.
8. Free device memory and host output.

## Kernel / Algorithm Flow

- Map each thread to one output `(row, col)`.
- For valid `(row, col)`, loop over the reduction dimension `k`.
- Accumulate:
  - `C[row][col] = sum(A[row][k] * B[k][col])`

## Important Lecture Connections

- [Lecture 3](lectures_md/L03_Multidimensional_Grids_and_Basic_Matrix_Multiplication.md)

## Core Patterns / Formulas

- Row index:
  - `row = blockIdx.y * blockDim.y + threadIdx.y`
- Column index:
  - `col = blockIdx.x * blockDim.x + threadIdx.x`
- Row-major indexing:
  - `A[row * numAColumns + k]`
  - `B[k * numBColumns + col]`
  - `C[row * numCColumns + col]`

## Common Bugs / Edge Cases

- Swapping row and column dimensions.
- Forgetting the output bounds check.
- Handling non-square matrices incorrectly.
- Misunderstanding the reduction dimension.

## Quiz / Exam Takeaways

- Boundary conditions and branch structure can create control divergence.
- Be able to count warps in a 2D launch.
- Understand why naive matrix multiply is correct but memory-bound.

## Terminology Used in This Lab

- 2D grid
- Row-major indexing
- Reduction dimension
- Control divergence
- Arithmetic intensity
