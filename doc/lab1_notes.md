# Lab 1 Notes: Vector Addition

## Title / Objective

Implement a simple CUDA vector addition kernel and learn the basic host/device workflow.

## What I Built

- A kernel that computes `C[i] = A[i] + B[i]`.
- The basic CUDA host flow for allocation, copies, launch, and cleanup.

## Host Flow

1. Allocate device memory for input and output vectors.
2. Copy host vectors to device.
3. Configure `gridSize` and `blockSize`.
4. Launch the vector addition kernel.
5. Copy the output back to the host.
6. Free device memory.

## Kernel / Algorithm Flow

- Each thread computes one output element.
- Compute the global index with `i = blockIdx.x * blockDim.x + threadIdx.x`.
- Perform the addition only if `i < n`.

## Important Lecture Connections

- [Lecture 2](lectures_md/L02_Introduction_to_CUDA_C_and_Data_Parallel_Programming.md)
- [Lecture 3](lectures_md/L03_Multidimensional_Grids_and_Basic_Matrix_Multiplication.md)

## Core Patterns / Formulas

- Global index:
  - `i = blockIdx.x * blockDim.x + threadIdx.x`
- Grid size:
  - `(inputLength + blockSize - 1) / blockSize`

## Common Bugs / Edge Cases

- Forgetting the bounds check.
- Using the wrong `cudaMemcpy` direction.
- Launching too few blocks.

## Quiz / Exam Takeaways

- Know the standard CUDA host flow in order.
- Be able to explain why the grid often overshoots the input.
- Be able to compute the number of blocks from an input length and block size.
- Be comfortable with `cudaMemcpy(dst, src, bytes, direction)` and common direction mistakes.
- Remember that multi-element-per-thread indexing usually means:
  - global thread id first
  - chunk offset second
- For float vector addition:
  - global reads = `8N` Bytes
  - global writes = `4N` Bytes

## Terminology Used in This Lab

- Host
- Device
- Kernel
- Thread
- Block
- Grid
- Bounds check
- Data parallelism
