# Profiling Lecture Notes

## Objective

The profiling lecture is not a graded lab. It is a worked example that teaches how to compare CUDA kernel implementations and validate performance claims with profiler tools instead of intuition alone.

The example uses one workload, matrix multiplication, and evaluates three kernels:

1. a simple baseline kernel
2. a shared-memory tiled kernel
3. a shared-memory tiled kernel with a more optimized shared-memory access pattern

For the full walkthrough with code-level discussion and actual output logs, see [Profiling-Lecture Detailed Guide](../Profiling-Lecture/lecture_notes.md).

## What It Teaches

- A faster kernel should be explained, not just observed.
- Shared memory is a major optimization step, but it is not the last step.
- NVTX ranges make profiler timelines much easier to read.
- `nsys` and `ncu` answer different questions and should be used together.

## Main Workflow

The example follows a very practical optimization workflow:

1. write multiple implementations of the same kernel
2. run them on the same input
3. compare runtime and throughput
4. inspect the timeline with `nsys`
5. inspect kernel-level bottlenecks with `ncu`

This is a useful mental model for project work because it turns performance tuning into a measurable process.

## Code Connections

- [`matmul_pageable.cu`](../Profiling-Lecture/matmul_pageable.cu) runs all three versions on the same matrices and wraps major phases in NVTX labels.
- [`simple_matmul.cuh`](../Profiling-Lecture/kernels/matmul/simple_matmul.cuh) is the baseline global-memory-heavy implementation.
- [`shared_matmul.cuh`](../Profiling-Lecture/kernels/matmul/shared_matmul.cuh) introduces classic shared-memory tiling.
- [`shared_improved_matmul.cuh`](../Profiling-Lecture/kernels/matmul/shared_improved_matmul.cuh) refines shared-memory access behavior further.

The progression is useful because each version isolates a specific optimization idea.

## Actual Results

From [`run.out`](../Profiling-Lecture/run.out):

| Kernel | Time (ms) | GFLOPS |
| --- | ---: | ---: |
| Simple | 9.91462 | 677.787 |
| Shared | 3.33962 | 2012.2 |
| Shared Improved | 2.29391 | 2929.49 |

Key speedups:

- shared vs simple: `2.97x`
- improved vs shared: `1.46x`
- improved vs simple: `4.32x`

These numbers show the expected pattern:

- the move to shared memory gives the largest gain
- the improved version still gets a meaningful second-stage gain
- memory hierarchy and access pattern quality both matter

## Profiler Takeaways

- [`nsys.out`](../Profiling-Lecture/nsys.out) confirms that timeline profiling succeeded and produced [`matmul_pageable.nsys-rep`](../Profiling-Lecture/profile_nsys/matmul_pageable.nsys-rep).
- [`ncu.out`](../Profiling-Lecture/ncu.out) confirms that all three kernels were profiled individually and produced [`matmul_pageable.ncu-rep`](../Profiling-Lecture/profile_ncu-rep/matmul_pageable.ncu-rep).
- `nsys` is the right tool for execution order, phase boundaries, and copy-vs-kernel timing.
- `ncu` is the right tool for occupancy, memory behavior, stall reasons, and other per-kernel metrics.

## Why It Matters for ECE408

This lecture example connects directly to the later part of the course:

- It reinforces the value of shared memory and tiling from earlier labs.
- It shows that "optimized" kernels still have deeper bottlenecks worth investigating.
- It introduces the workflow needed for project-level performance tuning.

The most important mindset shift is this:

- lab-style thinking: "My kernel is correct."
- profiling mindset: "My kernel is correct, I measured it, and I can explain its bottleneck."
