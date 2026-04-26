# ECE408 Quick Review

## Lab 1

- Vector addition introduced the full CUDA host flow: allocate, copy, launch, copy back, free.
- Basic indexing pattern: `i = blockIdx.x * blockDim.x + threadIdx.x`.
- Bounds checks are required because the grid often overshoots the data length.
- Main takeaway: correctness first, then think about performance.

## Lab 2

- Basic matrix multiplication mapped one thread to one output element.
- 2D grid/block mapping matters as much as the arithmetic.
- Row-major indexing and dimension checks are the most common correctness traps.
- Main takeaway: a correct kernel can still be heavily memory-bound.

## Lab 3

- Shared-memory tiling improved matrix multiply by reusing input tiles.
- The core loop pattern is load tile, synchronize, compute, synchronize.
- Zero-padding solves out-of-bounds tile loads cleanly.
- Main takeaway: shared memory trades coordination for lower global-memory traffic.

## Lab 4

- 3D convolution extended tiling to 3D and moved the mask into constant memory.
- Output tile plus halo input tile is the key geometric idea.
- Data layout is as important as the convolution arithmetic.
- Main takeaway: local reuse and correct boundary handling dominate stencil kernels.

## Lab 5

- Histogram equalization is a pipeline, not a single kernel.
- The critical issue is many threads updating the same histogram bins.
- Privatization turns one hot global histogram into many block-private histograms.
- CDF is just a prefix sum over histogram probabilities.
- Main takeaway: atomics solve correctness, but kernel structure determines performance.

## Lab 6

- Reduction is a staged combination pattern that fits shared memory very well.
- One block reduces one segment, then the host combines block partial sums.
- Improved reduction reorders the tree to reduce divergence and support coalesced loads.
- Shared memory handles the tree after the initial global loads.
- Main takeaway: reduction performance depends on mapping, synchronization, and memory hierarchy together.

## Lab 7

- Parallel scan computes all prefix results, not just one final reduction value.
- Inclusive and exclusive scan are easy to mix up, especially when using a work-efficient tree.
- Hierarchical scan means: local block scan, scan the block sums, then add scanned block sums back.
- Brent-Kung is more work-efficient than Kogge-Stone, while Kogge-Stone often has lower latency.
- Scan is memory-bound, so clean block mapping and low extra traffic matter a lot.
- Main takeaway: large scans need both a correct tree inside each block and a correct hierarchy across blocks.

## Lectures 18-19

- Lecture 18 is the direct algorithm lecture for Lab 7: scan definitions, Kogge-Stone, Brent-Kung, and hierarchical scan.
- Lecture 19 is the systems context lecture: bandwidth, PCIe vs GPU memory, and why app-level performance is not only about one kernel.
- Scan is a good example of a memory-bound pattern where algorithm design and system bandwidth intuition meet.
- Main takeaway: understand both the local tree algorithm and the larger system bottlenecks around it.

## Lab 8

- JDS (Jagged Diagonal Storage) solves the coalescing and divergence problems of CSR for SpMV.
- The key insight: sort rows by length, transpose to column-major, so adjacent threads access adjacent memory.
- One thread per sorted row; `matRowPerm` maps the result back to the original row order.
- No padding (unlike ELL), no atomics (unlike COO), coalesced reads, and low divergence.
- Main takeaway: for sparse problems, choosing the right storage format matters more than kernel-level tricks.

## Profiling Lecture

- The profiling lecture is a worked example, not a graded lab, built around three matrix multiplication kernels.
- The key progression is baseline global-memory kernel, shared-memory tiled kernel, then improved shared-memory access behavior.
- `nsys` explains when things happen on the timeline, especially with NVTX ranges.
- `ncu` explains why a specific kernel is fast or slow through detailed kernel metrics.
- Main takeaway: performance work should move from guesswork to profiler-guided explanation.
