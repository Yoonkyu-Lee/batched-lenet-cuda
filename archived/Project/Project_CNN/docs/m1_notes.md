# Milestone 1 Notes

## Objective

Milestone 1 was about implementing the forward convolution layer on both CPU and GPU for the CNN project.

Files edited for the milestone:

- `project/src/layer/custom/cpu-new-forward.cc`
- `project/src/layer/custom/new-forward.cu`

## What I Implemented

### CPU path

- Implemented the full forward convolution loop nest in `cpu-new-forward.cc`
- Followed the reference formulation:
  - batch loop
  - output map loop
  - output height/width loop
  - channel and `K x K` accumulation loop

### GPU path

- Implemented a basic CUDA forward convolution kernel in `new-forward.cu`
- Added:
  - device memory allocation in `conv_forward_gpu_prolog`
  - host-to-device copies for input and mask
  - kernel launch in `conv_forward_gpu`
  - device-to-host copy and cleanup in `conv_forward_gpu_epilog`

## Expected Correctness

From the milestone handout, the expected accuracy values are approximately:

| Batch Size | Expected Accuracy |
| --- | ---: |
| 100 | 0.86 |
| 1000 | 0.886 |
| 10000 | 0.8714 |

CPU reached the expected values.

GPU initially passed small batches but failed large batches.

## The Bug I Found

### Symptom from the autograder

The autograder feedback showed:

- GPU correct for `100`
- GPU correct for `1000`
- GPU wrong for `8767`
- GPU wrong for `10000`

This strongly suggested that the issue was not the convolution formula itself. Instead, it pointed to a batch-size-dependent launch or indexing bug.

### Root cause

In the original GPU implementation, the launch configuration used:

```cpp
gridDim.z = Batch * Map_out;
```

This was unsafe for the second convolution layer:

- Conv2 uses `Map_out = 16`
- so large batches make `grid.z` exceed the CUDA limit

Examples:

- `4095 * 16 = 65520` -> still valid
- `4096 * 16 = 65536` -> overflow of the common `grid.z` limit
- `8767 * 16 = 140272` -> invalid
- `10000 * 16 = 160000` -> invalid

That explains why the bug only appeared for larger batches.

### Why it was hard to notice at first

The original code did not check CUDA errors after:

- `cudaMalloc`
- `cudaMemcpy`
- kernel launch
- `cudaDeviceSynchronize`

So the kernel could fail silently, and the program would continue to produce misleading outputs such as:

- extremely small `Op Time`
- collapsed accuracy

## The Fix

### Safer kernel mapping

Instead of storing `Batch * Map_out` in `grid.z`, I changed the mapping to:

- `grid.z = Batch`
- fold `Map_out` into `grid.x`

Conceptually:

- `z` dimension -> batch
- `y` dimension -> output height tiles
- `x` dimension -> `(output map, output width tile)`

Then inside the kernel, I decode:

- `b` from `blockIdx.z`
- `m` from `blockIdx.x / tiles_w`
- `tile_w` from `blockIdx.x % tiles_w`

This keeps the launch within valid grid bounds even for large batch sizes.

### Error checking

I also added a `CUDA_CHECK(...)` macro around:

- `cudaMalloc`
- `cudaMemcpy`
- `cudaGetLastError`
- `cudaDeviceSynchronize`
- `cudaFree`

This makes runtime failures visible immediately instead of silently corrupting the result.

## How I Verified the Fix

### Manual validation

I re-ran GPU inference on larger batches and confirmed that the final `10000` case returned:

- `Test Accuracy: 0.8714`

The Conv2 runtime also became reasonable again instead of appearing as an unrealistically tiny near-zero value.

### Self-check tool

I created:

- `m1_selfcheck.sh`
- `m1_selfcheck.slurm`

The self-check script runs a set of public and edge batches:

- `100`
- `1000`
- `4095`
- `4096`
- `8767`
- `10000`

It checks:

- execution success
- parsed accuracy
- public expected accuracy for key batches
- edge-batch sanity for large hidden-style cases
- total `Op Time` on batch `10000`

### Final self-check result

The self-check reported:

- `100` -> pass
- `1000` -> pass
- `4095` -> pass
- `4096` -> pass
- `8767` -> pass
- `10000` -> pass
- overall result -> `PASS`

This confirmed that the large-batch failure was fixed.

## Workflow Lessons

### 1. Use `sbatch` for GPU validation

On Delta, the reliable workflow is:

1. build on the login node
2. submit jobs with `sbatch`
3. inspect `.out`, `.err`, and summary files

Trying to run GPU binaries directly on the login node is misleading because the login node does not provide a CUDA-capable device.

### 2. Check edge cases, not only public cases

Public cases like `100` and `1000` were not enough to expose the real bug.

The key extra cases were:

- `4095`
- `4096`
- `8767`
- `10000`

This is a useful pattern for future milestones: always add boundary and near-boundary test cases.

### 3. Silent CUDA failures are dangerous

A kernel launch can fail without immediately crashing the whole program if errors are not checked.

That means:

- low accuracy
- strange timing
- incomplete output

may actually be launch/configuration bugs rather than math bugs.

### 4. Correctness first, then performance

M1 reinforced an important project habit:

- first make the implementation correct for all batch sizes
- then trust timing numbers

Performance numbers from an invalid launch are meaningless.

## What This Means for Milestone 2

Before starting M2:

- keep the safer CUDA error-checking habit
- keep using `sbatch`-based validation
- reuse the self-check mindset for larger or hidden-style batches
- verify correctness before collecting profiling data

M2 adds profiling, unrolling, and kernel fusion, so the risk of silent runtime mistakes is even higher. The debugging discipline learned in M1 is directly reusable.
