# Profiling-Lecture Detailed Guide

## What This Folder Is

`Profiling-Lecture` is a lecture example, not a graded lab. Its purpose is to show how we move from "I wrote a CUDA kernel" to "I can compare implementations, measure them, and explain why one is faster."

The example uses one workload, matrix multiplication, and runs it with three different GPU kernels:

1. a simple global-memory-heavy kernel
2. a shared-memory tiled kernel
3. a shared-memory tiled kernel with a more optimized shared-memory access pattern

The real lesson is not only that the third kernel is faster. The real lesson is that performance work should be guided by tools. This folder is designed to be opened together with `nsys`, `ncu`, and the source code.

## Folder Structure

### Main driver

- [`matmul_pageable.cu`](./matmul_pageable.cu): creates input matrices, copies them to the GPU, runs all three kernels, and places NVTX ranges around major phases so the profiler timeline is readable.

### Kernel implementations

- [`kernels/matmul/simple_matmul.cuh`](./kernels/matmul/simple_matmul.cuh): one thread computes one output element and reads directly from global memory across the full `K` dimension.
- [`kernels/matmul/shared_matmul.cuh`](./kernels/matmul/shared_matmul.cuh): standard `16 x 16` tiled shared-memory matrix multiplication.
- [`kernels/matmul/shared_improved_matmul.cuh`](./kernels/matmul/shared_improved_matmul.cuh): builds on the shared-memory version and improves how one of the shared-memory tiles is consumed.
- [`kernels/matmul/matmul.hpp`](./kernels/matmul/matmul.hpp): holds the GEMM shape and pointer parameter structs.

### Build and job scripts

- [`build.sh`](./build.sh): configures and builds the executable.
- [`run.slurm`](./run.slurm): submits a regular performance run.
- [`nsys.slurm`](./nsys.slurm): submits an Nsight Systems profiling run.
- [`ncu.slurm`](./ncu.slurm): submits an Nsight Compute profiling run.
- [`generate_nsys.sh`](./generate_nsys.sh): creates a `.nsys-rep` report.
- [`generate_ncu-rep.sh`](./generate_ncu-rep.sh): creates a `.ncu-rep` report.

### Runtime outputs and reference reports

- Generated logs:
  - [`run.out`](./run.out)
  - [`run.err`](./run.err)
  - [`nsys.out`](./nsys.out)
  - [`nsys.err`](./nsys.err)
  - [`ncu.out`](./ncu.out)
  - [`ncu.err`](./ncu.err)
- Generated reports:
  - [`profile_nsys/matmul_pageable.nsys-rep`](./profile_nsys/matmul_pageable.nsys-rep)
  - [`profile_ncu-rep/matmul_pageable.ncu-rep`](./profile_ncu-rep/matmul_pageable.ncu-rep)
- Reference reports from other GPUs:
  - [`reference_nsys-rep`](./reference_nsys-rep)
  - [`reference_ncu-rep`](./reference_ncu-rep)

## Main Lecture Takeaway

The main point of this example is:

- optimization is not guesswork
- optimization is not only "use shared memory"
- optimization is a process of building variants, measuring them, and checking the bottleneck with the right profiler

This folder teaches two levels of profiling:

- `nsys` gives a timeline view
  - When does host-side work happen?
  - When do copies happen?
  - In what order do kernels execute?
  - Do the NVTX regions line up with what we expect?
- `ncu` gives a kernel view
  - What is each kernel spending hardware resources on?
  - Is the kernel limited by memory traffic, shared-memory behavior, instruction mix, or occupancy?
  - Did a kernel optimization actually reduce the expected bottleneck?

In other words, `nsys` helps us understand the execution story, and `ncu` helps us understand the kernel bottleneck story.

## How the Lecture Connects to the Code

The design of the code mirrors a lecture progression from simple implementation to optimized implementation.

### 1. One program runs all versions on the same input

[`matmul_pageable.cu`](./matmul_pageable.cu) sets up one consistent experiment:

- default matrix sizes are `M = 1600`, `N = 1400`, `K = 1500`
- host matrices are created and filled with random values
- device buffers are allocated once
- the same input matrices are used for all three kernels

This is important because it makes the comparison fair. We are not changing the workload, only the implementation.

### 2. NVTX ranges make the profiler readable

The driver places NVTX markers around:

- `generate data`
- `host-to-device`
- `matmul simple kernel`
- `matmul shared kernel`
- `matmul shared improved kernel`

That means the `nsys` timeline is not just a wall of API calls. It becomes a labeled story that we can interpret much more quickly.

### 3. The three kernels represent three optimization stages

- `SimpleGemm` shows the baseline idea: one output element per thread, direct global-memory reads for the dot product.
- `SharedGemm` introduces tiling and shared memory so data loaded from global memory can be reused by many threads in the block.
- `SharedImprovedGemm` keeps the tiling idea but improves the shared-memory consumption pattern, especially for tile `A`, to reduce the number of shared-memory requests.

This progression is pedagogically useful because it isolates optimization concepts:

- first: a correct but less efficient kernel
- second: a classic memory reuse optimization
- third: a more hardware-conscious micro-optimization

## Three Kernel Comparison

### Simple kernel

Source: [`simple_matmul.cuh`](./kernels/matmul/simple_matmul.cuh)

Execution pattern:

- one thread computes one output element of `C`
- thread index is mapped to a flattened output index
- each thread loops over the full `K` dimension

Memory behavior:

- each multiply-add reads from global memory directly
- there is little data reuse at the block level
- many nearby threads need overlapping input data but do not explicitly share it

Expected bottleneck:

- global memory traffic and low reuse

Why it matters:

- this is the easiest version to understand
- it is often the right first step for correctness
- it provides the baseline that later optimizations must beat

### Shared-memory tiled kernel

Source: [`shared_matmul.cuh`](./kernels/matmul/shared_matmul.cuh)

Execution pattern:

- one block computes one output tile
- each block is `16 x 16`
- tiles of `A` and `B` are loaded into shared memory
- threads synchronize, compute partial products, then move to the next tile along `K`

Memory behavior:

- global memory loads are amortized through reuse in shared memory
- each loaded tile is used by many threads inside the block
- extra synchronization is required, but reuse usually outweighs the sync cost

Expected bottleneck:

- much less pressure on global memory than the simple version
- remaining cost shifts toward shared-memory use, synchronization, and arithmetic throughput

Why it improves performance:

- shared memory allows data reuse that the simple kernel did not exploit
- this is the classic "tiling makes matrix multiplication fast" step

### Shared-memory improved kernel

Source: [`shared_improved_matmul.cuh`](./kernels/matmul/shared_improved_matmul.cuh)

Execution pattern:

- keeps the same broad tiling strategy as `SharedGemm`
- uses an aligned shared-memory buffer abstraction
- accesses tile `A` in vectorized chunks

Memory behavior:

- tile `A` is consumed with wider shared-memory accesses
- the code comment explicitly states that this is meant to reduce the number of shared-memory requests and decrease MIO throttle
- tile `B` is still accessed in the natural pattern needed for column-wise use

Expected bottleneck:

- shared-memory behavior is more optimized than the standard tiled version
- arithmetic work is unchanged in spirit, but the path to supplying operands becomes more efficient

Why it improves performance:

- it shows that "use shared memory" is not the end of optimization
- even after tiling, the exact access pattern inside shared memory still matters

### When this matters for labs and projects

This progression is directly relevant to ECE408 work:

- in earlier labs, the first goal is often correctness
- then we learn to use shared memory and tiling for reuse
- later, especially in projects, we need to understand second-order effects such as access pattern quality and profiler-visible bottlenecks

This example is a compact version of that journey.

## Actual Run Results

The regular run output in [`run.out`](./run.out) shows the following times and throughput values:

| Kernel | Time (ms) | GFLOPS |
| --- | ---: | ---: |
| Simple | 9.91462 | 677.787 |
| Shared | 3.33962 | 2012.2 |
| Shared Improved | 2.29391 | 2929.49 |

### Speedup summary

Using runtime as the basis:

- shared vs simple: `9.91462 / 3.33962 = 2.97x`
- improved vs shared: `3.33962 / 2.29391 = 1.46x`
- improved vs simple: `9.91462 / 2.29391 = 4.32x`

What this means in plain language:

- moving from the naive implementation to the tiled shared-memory version gives the biggest jump
- moving from the tiled version to the improved tiled version still matters and gives a meaningful second-stage improvement
- the best kernel is not just slightly better than the baseline; it is more than four times faster on this workload

This is exactly the kind of pattern we want students to recognize:

- big gains often come from data reuse and memory hierarchy awareness
- smaller but still valuable gains come from refining access patterns after the major optimization is already in place

### Why the GFLOPS values make sense

The GFLOPS values follow the same ranking:

- simple: `677.787`
- shared: `2012.2`
- shared improved: `2929.49`

This consistency is reassuring. Lower runtime and higher throughput tell the same story, which suggests the measurements are behaving as expected.

## Profiler Results and Interpretation

## Nsight Systems (`nsys`)

The profiling run in [`nsys.out`](./nsys.out) completed successfully and generated:

- [`profile_nsys/matmul_pageable.nsys-rep`](./profile_nsys/matmul_pageable.nsys-rep)

What this means:

- the timeline capture worked
- the executable was run under `nsys`
- the report can now be opened in the Nsight Systems GUI for a visual inspection

What you should expect to see in the timeline:

1. host-side setup and data generation
2. host-to-device copies
3. the three kernel regions in sequence:
   - simple
   - shared
   - shared improved

Because the code uses NVTX ranges in [`matmul_pageable.cu`](./matmul_pageable.cu), those regions should appear with human-readable labels rather than only raw kernel launches. That is one of the most useful teaching features of this example.

How to interpret the timeline:

- use it to confirm ordering and phase boundaries
- use it to confirm that the kernel regions correspond to the expected high-level phases
- use it to check whether copy time is significant compared with kernel time
- use it to build intuition before diving into low-level metrics

`nsys` answers: "What happened, and when did it happen?"

## Nsight Compute (`ncu`)

The profiling run in [`ncu.out`](./ncu.out) completed successfully and generated:

- [`profile_ncu-rep/matmul_pageable.ncu-rep`](./profile_ncu-rep/matmul_pageable.ncu-rep)

The console log shows that `ncu` profiled three kernels individually:

- `"cudaKernel" - 0`
- `"cudaKernel" - 1`
- `"cudaKernel" - 2`

Each one required many passes, which is normal for Nsight Compute when collecting a rich set of metrics with `--set full`.

What this means:

- `ncu` is not mainly about end-to-end timeline reading
- it is about detailed per-kernel analysis
- this report is where you would inspect occupancy, memory throughput, instruction behavior, and stall reasons

Even without embedding screenshots, we can already interpret the outcome conservatively:

- all three kernels were profiled successfully
- the generated report is ready for deeper inspection
- the fastest kernel in the plain run is also the most optimized kernel structurally
- the ranking from the text output matches the intended optimization story

`ncu` answers: "Why is this kernel fast or slow?"

## What the two profilers together tell us

Taken together, the outputs support the lecture goal:

- `run.out` shows the performance ranking
- `nsys.out` confirms the timeline capture and region structure
- `ncu.out` confirms per-kernel profiling took place

So this folder is not just code. It is a complete mini case study:

- write several kernel variants
- label phases with NVTX
- run the code normally
- inspect the timeline with `nsys`
- inspect per-kernel details with `ncu`

## Important Caveats

The stderr logs in [`run.err`](./run.err), [`nsys.err`](./nsys.err), and [`ncu.err`](./ncu.err) include a module warning:

- `cuda/12.4` could not be found by the module system on that run

However, despite that warning:

- the executable still ran
- `run.out` contains valid timing output
- `nsys` still produced a `.nsys-rep` file
- `ncu` still produced a `.ncu-rep` file

So the correct interpretation is not "the experiment failed." The correct interpretation is:

- there was an environment/module warning
- but usable outputs were still produced

One additional `nsys` warning mentions device-side CUDA event completion tracing. That warning is about possible overhead and trace dependencies, not about complete failure of profiling.

Also note:

- the deepest profiler interpretation still lives in the report files, not in the terminal text logs
- to inspect stall reasons or detailed metric tables, you should open the generated reports with the Nsight GUI tools

## How to Reproduce

From [`Profiling-Lecture`](./), the workflow is:

```bash
bash build.sh
sbatch run.slurm
sbatch nsys.slurm
sbatch ncu.slurm
```

Expected outputs:

- regular run logs:
  - [`run.out`](./run.out)
  - [`run.err`](./run.err)
- Nsight Systems logs and report:
  - [`nsys.out`](./nsys.out)
  - [`nsys.err`](./nsys.err)
  - [`profile_nsys/matmul_pageable.nsys-rep`](./profile_nsys/matmul_pageable.nsys-rep)
- Nsight Compute logs and report:
  - [`ncu.out`](./ncu.out)
  - [`ncu.err`](./ncu.err)
  - [`profile_ncu-rep/matmul_pageable.ncu-rep`](./profile_ncu-rep/matmul_pageable.ncu-rep)

## Why This Matters for ECE408

This example connects naturally to ECE408 themes:

- **Shared memory**
  - the jump from `SimpleGemm` to `SharedGemm` is a concrete demonstration of why shared memory matters
- **Tiling**
  - the standard tiled kernel is exactly the kind of idea that appears again and again in GPU programming
- **Memory access patterns**
  - the improved kernel shows that performance depends not only on what memory is used, but how it is used
- **Profiler-guided optimization**
  - `nsys` and `ncu` turn optimization from guessing into measurement
- **From correctness to performance diagnosis**
  - many labs teach how to make kernels correct first
  - this example teaches the next step: how to explain performance with evidence

If you are preparing for later labs or the CNN project, this folder is worth studying even though it is not itself a graded assignment. It models the mindset you eventually want:

- write a baseline
- improve it with a clear optimization idea
- measure the result
- verify the story with profiling tools
