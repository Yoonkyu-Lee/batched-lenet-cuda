# Lab 5 Notes: Histogram Equalization

## Title / Objective

Implement histogram equalization for an RGB image on the GPU, with the CDF scan computed on the CPU.

## What I Built

- A full image-processing pipeline:
  - float RGB -> uchar RGB -> grayscale -> histogram -> CDF -> equalization -> float RGB
- A block-private shared-memory histogram kernel merged into a global histogram.
- CPU-side CDF construction and `cdfMin` handling.

## Host Flow

1. Import the input image and create the output image.
2. Allocate device buffers for the float image, uchar image, grayscale image, histogram, and CDF.
3. Copy the float input image to the GPU.
4. Launch the float-to-uchar kernel.
5. Launch the RGB-to-grayscale kernel.
6. Launch the histogram kernel.
7. Copy the histogram back to the CPU and compute the CDF.
8. Copy the CDF to the GPU.
9. Launch the equalization kernel.
10. Launch the uchar-to-float kernel.
11. Copy the final float output image back to the host.
12. Free all device buffers.

## Kernel / Algorithm Flow

- `floatToUcharKernel`
  - scale `[0, 1]` float values to `[0, 255]`
- `rgbToGrayKernel`
  - compute `gray = 0.21 * r + 0.71 * g + 0.07 * b`
- `histogramKernel`
  - initialize one shared histogram per block
  - use `atomicAdd` into the private histogram
  - merge each block-private histogram into the global histogram
- CPU CDF phase
  - normalize histogram counts by pixel count
  - compute prefix sum
  - find `cdfMin`
- `equalizeImageKernel`
  - apply the equalization mapping to each channel value
- `ucharToFloatKernel`
  - rescale `[0, 255]` back to `[0, 1]`

## Important Lecture Connections

- [Lecture 14](lectures_md/L14_Atomic_Operations_and_Histogramming.md)
- [Lecture 15](lectures_md/L15_Parallel_Computation_Patterns_Reduction_Trees.md)
- [Lecture 8](lectures_md/L08_Convolution_Concept_and_Constant_Cache.md)
- [Lecture 9](lectures_md/L09_2D_Convolution.md)

## Core Patterns / Formulas

- Grayscale conversion:
  - `gray = 0.21 * r + 0.71 * g + 0.07 * b`
- Probability of a histogram bin:
  - `p(x) = x / (width * height)`
- CDF:
  - `cdf[i] = cdf[i - 1] + p(histogram[i])`
- Equalization:
  - `255 * (cdf[val] - cdfMin) / (1 - cdfMin)`
- 1D grid-stride loop:
  - `i = blockIdx.x * blockDim.x + threadIdx.x`
  - `stride = blockDim.x * gridDim.x`

## Common Bugs / Edge Cases

- Forgetting to zero the global histogram.
- Treating a separate check and `atomicAdd` as if both were atomic.
- Using an unsafe shared-histogram initialization when `blockDim.x < 256`.
- Mishandling `cdfMin` when initial bins are empty.
- Forgetting that the CDF stage in this lab is done on the CPU.

## Quiz / Exam Takeaways

- Histogram insertion with a separate `if (count < CAPACITY)` check is unsafe because the check and `atomicAdd` are not one atomic unit.
- Throughput for atomics on the same address is the inverse of average latency, so cache-hit rate matters.
- Privatized histogram cost includes one shared-memory atomic per input value and one global-memory atomic per bin per block in the merge stage.
- In histogram kernels, sorted inputs can increase contention because many threads hit the same bin together.
- Atomic latency and privatization trade-offs are likely quiz targets.

## Terminology Used in This Lab

- Atomic
- Serialized
- Contention
- Privatization
- Coalesced access
- Grid-stride loop
- Histogram
- CDF
- Prefix sum
- Coarsening
