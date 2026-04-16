# Lab 4 Notes: 3D Convolution

## Title / Objective

Implement 3D convolution using constant memory for the mask and shared-memory tiling for the input volume.

## What I Built

- A 3D convolution kernel with a `3 x 3 x 3` mask in constant memory.
- A 3D shared-memory tile with halo handling.

## Host Flow

1. Read input volume and 3D kernel.
2. Extract `z`, `y`, and `x` sizes from the first three input values.
3. Allocate device memory for input and output.
4. Copy the input volume to device and the kernel to constant memory.
5. Configure a 3D grid and block.
6. Launch the convolution kernel.
7. Copy the result back to the host.
8. Free device memory.

## Kernel / Algorithm Flow

- Compute output coordinates.
- Shift to input coordinates to account for halo.
- Load one input tile with halo into shared memory.
- Synchronize after loading.
- Threads inside the valid output tile compute the `3 x 3 x 3` sum.
- Write valid output values only.

## Important Lecture Connections

- [Lecture 8](lectures_md/L08_Convolution_Concept_and_Constant_Cache.md)
- [Lecture 9](lectures_md/L09_2D_Convolution.md)

## Core Patterns / Formulas

- Input tile size per dimension:
  - `TILE + MASK_WIDTH - 1`
- Output index:
  - `z * y_size * x_size + y * x_size + x`

## Common Bugs / Edge Cases

- Forgetting that the first three input values are metadata.
- Loading halo values incorrectly.
- Writing output outside valid bounds.
- Mixing input-tile and output-tile coordinates.

## Quiz / Exam Takeaways

- Small masks can favor strategies with lower shared-memory footprint.
- Input tile size equals output tile size plus halo on both sides.
- Reuse analysis matters for both bandwidth and occupancy.

## Terminology Used in This Lab

- Constant memory
- 3D tiling
- Halo
- Stencil
- Reuse
- Occupancy
