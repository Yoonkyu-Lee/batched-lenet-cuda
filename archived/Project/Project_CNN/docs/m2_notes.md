# Milestone 2 Notes

## Status

This is a living document for Milestone 2. It is meant to be updated during implementation, profiling, debugging, and final reflection.

Recommended use:

- write what was attempted
- record what actually worked
- save important command lines
- keep accuracy and timing evidence in one place
- note any bug patterns that may matter again in later milestones

## Objective

Milestone 2 builds on Milestone 1 by adding:

1. profiling of CPU convolution
2. profiling of the basic GPU convolution
3. input feature unrolling
4. profiling of the unrolled version
5. kernel fusion
6. profiling of the fused version

Primary files for this milestone:

- `project/src/layer/custom/unroll-new-forward.cu`
- `project/src/layer/custom/kernel-fusion-forward.cu`

Reference materials:

- `README_m2.md`
- `m1_notes.md`
- `doc/lectures_md/L11_Computation_in_CNNs_and_Transformers.md`
- `doc/lectures_md/L13_Profiling_on_Nvidia_GPU.md`
- `doc/lectures_md/L16_Advanced_Optimizations_for_Projects.md`

## Milestone Structure

### Part 1: CPU Profiling

Goal:

- profile the CPU convolution with `gprof`

What to record here:

- exact build command
- exact run command
- top functions in flat profile
- whether `conv_forward_cpu` dominates execution as expected

Notes:

- `-pg` must be added temporarily
- remove `-pg` before GPU performance measurements

### Part 2: Unrolled Convolution

Goal:

- implement input unrolling
- keep provided matrix multiplication and permutation kernels unchanged

What to implement:

- `matrix_unrolling_kernel`
- host-side memory management in `conv_forward_gpu_prolog`
- kernel launch logic in `conv_forward_gpu`
- copy-back and cleanup in `conv_forward_gpu_epilog`

Key shape:

```text
unrolled matrix:
(Channel * K * K) x (Batch * Height_out * Width_out)
```

Important caution:

- for large batches, indexing may exceed `INT_MAX`
- prefer `size_t` where flattening sizes or offsets can become large

### Part 3: Profiling the Basic and Unrolled GPU Versions

Goal:

- compare the basic M1 GPU kernel and the M2 unrolled kernel with:
  - `nsys`
  - `ncu`

What to record:

- which binary was profiled
- which command was used
- where the `.nsys-rep` and `.ncu-rep` outputs were saved
- kernel names that dominate runtime
- memory copy vs kernel time observations
- one or two bottlenecks worth remembering

### Part 4: Kernel Fusion

Goal:

- implement a fused kernel that avoids materializing the full unrolled matrix in global memory

What to implement:

- fused kernel body in `matmul_conv_fused`
- prolog / launch / epilog host functions

Conceptual target:

- read from original input directly
- perform tiled multiply logic
- write the final output in the correct permuted layout

### Part 5: Profiling the Fused Version

Goal:

- verify that fusion improves performance over the separate-kernel unrolled pipeline

What to record:

- accuracy
- total Op Time
- comparison to unroll version
- important `nsys` or `ncu` observations

## Implementation Log

Use this section as a chronological work log.

### Session Template

```text
Date:
Goal:
Files touched:
What changed:
How it was tested:
Result:
Next step:
```

### Session 1

- Date: 2026-03-31
- Goal: Implement the first correctness-oriented version of both M2 kernels and prepare repeatable validation.
- Files touched:
  - `project/src/layer/custom/unroll-new-forward.cu`
  - `project/src/layer/custom/kernel-fusion-forward.cu`
  - `m2_selfcheck.sh`
  - `m2_selfcheck.slurm`
- What changed:
  - completed the unrolling kernel and host-side unroll pipeline
  - added CUDA error checks to unroll and fused host paths
  - implemented a tiled fused convolution kernel that computes output directly from the original input
  - added an M2 self-check script to validate both `m2_unroll` and `m2_fused`
- How it was tested:
  - `bash -n m2_selfcheck.sh`
  - `./run.sh build`
- Result:
  - `m2_unroll` and `m2_fused` both compile successfully
  - implementation is in place
  - correctness and performance on Delta are still pending
- Next step:
  - run `./run.sh build`
  - submit `sbatch m2_selfcheck.slurm`
  - update the tables below with real accuracy/timing results

### Session 2

- Date: 2026-03-31
- Goal: Validate the first unroll and fused implementations on Delta with public and edge batches.
- Files touched:
  - `m2_notes.md`
- What changed:
  - recorded the first successful M2 self-check results
  - captured the unroll and fused timing baseline for later profiling comparison
- How it was tested:
  - `sbatch m2_selfcheck.slurm`
  - `cat m2_selfcheck.out`
  - `cat selfcheck_results_m2/20260331_041756/summary.txt`
- Result:
  - both `m2_unroll` and `m2_fused` passed all public and edge-batch checks
  - both versions met the README timing thresholds on batch `10000`
- Next step:
  - run `compute-sanitizer` on batch `100`
  - collect `nsys` and `ncu` results for M1 GPU, unroll, and fused
  - fill the profiling notes with actual dominant kernels and bottlenecks

### Session 3

- Date: 2026-03-31
- Goal: Prepare repeatable profiling and sanitizer scripts for the rest of M2.
- Files touched:
  - `build_pg.sh`
  - `m2_cpu_gprof.slurm`
  - `m2_unroll_sanitize.slurm`
  - `m2_fused_sanitize.slurm`
  - `m2_m1gpu_nsys.slurm`
  - `m2_unroll_nsys.slurm`
  - `m2_fused_nsys.slurm`
  - `m2_unroll_ncu.slurm`
  - `m2_fused_ncu.slurm`
  - `m2_notes.md`
- What changed:
  - added a dedicated `-pg` build script for CPU profiling
  - added Slurm scripts for `compute-sanitizer`, `nsys`, and `ncu`
  - replaced placeholder profiling commands in the notes with concrete commands
- How it was tested:
  - `bash -n` on every new script
- Result:
  - all helper scripts are syntax-valid and ready for Delta submission
- Next step:
  - run sanitizer first
  - then collect `nsys`
  - then collect `ncu`

### Session 4

- Date: 2026-03-31
- Goal: Collect the first sanitizer and profiling outputs and extract the main comparisons.
- Files touched:
  - `m2_notes.md`
- What changed:
  - recorded `compute-sanitizer` results for unroll and fused
  - recorded first `nsys` comparisons for M1 GPU, unroll, and fused
  - documented the current `ncu` permission blocker
- How it was tested:
  - `cat m2_unroll_sanitize_run.out`
  - `cat m2_fused_sanitize_run.out`
  - `cat m1_gpu_nsys_profile.out`
  - `cat m2_unroll_nsys_profile.out`
  - `cat m2_fused_nsys_profile.out`
  - `cat m2_unroll_ncu_profile.out`
  - `cat m2_fused_ncu_profile.out`
- Result:
  - both sanitizer runs finished with `ERROR SUMMARY: 0 errors`
  - `nsys` reports were generated successfully for M1 GPU, unroll, and fused
  - `ncu` did not complete because of `ERR_NVGPUCTRPERM`
- Next step:
  - use the generated `nsys` reports for the milestone report
  - resolve or work around the `ncu` performance-counter permission issue before finalizing the kernel-level profiling part
  - run CPU `gprof` if still needed for the report

## Correctness Tracking

Use this table for quick validation results.

| Version | Batch | Accuracy | Expected | Pass/Fail | Notes |
| --- | ---: | ---: | ---: | --- | --- |
| Basic GPU | 100 | 0.86 | 0.86 | Pass | M1 baseline |
| Basic GPU | 1000 | 0.886 | 0.886 | Pass | M1 baseline |
| Basic GPU | 10000 | 0.8714 | 0.8714 | Pass | M1 baseline |
| Unroll | 100 | 0.86 | 0.86 | Pass | selfcheck |
| Unroll | 1000 | 0.886 | 0.886 | Pass | selfcheck |
| Unroll | 10000 | 0.8714 | 0.8714 | Pass | selfcheck |
| Fused | 100 | 0.86 | 0.86 | Pass | selfcheck |
| Fused | 1000 | 0.886 | 0.886 | Pass | selfcheck |
| Fused | 10000 | 0.8714 | 0.8714 | Pass | selfcheck |

## Performance Tracking

Record timing summaries here.

### Public Targets from README

- unroll: total Op Time on batch `10000` should be around `200 ms`
- unroll full-credit threshold: `< 1200 ms`
- fused: total Op Time on batch `10000` should be around `60 ms`
- fused full-credit threshold: `< 200 ms`

### Timing Table

| Version | Batch | Conv1 Op Time (ms) | Conv2 Op Time (ms) | Total Op Time (ms) | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| Basic GPU | 10000 | 11.7012 | 45.4774 | 57.1786 | M1 corrected baseline |
| Unroll | 10000 |  |  | 163.8049 | selfcheck total |
| Fused | 10000 |  |  | 81.5453 | selfcheck total |

## Commands I Actually Used

Put real command lines here once they are working.

### Build

```bash
./run.sh build
```

### M2 self-check

```bash
sbatch m2_selfcheck.slurm
latest=$(ls -td selfcheck_results_m2/* | head -n 1)
cat "$latest/summary.txt"
```

Latest successful run:

```bash
cat selfcheck_results_m2/20260331_041756/summary.txt
```

### Typical execution

```bash
sbatch m2_unroll.slurm
sbatch m2_fused.slurm
```

### CPU profiling

```bash
bash build_pg.sh
sbatch m2_cpu_gprof.slurm
cat m2_cpu_gprof_report.txt

# After CPU profiling, rebuild the normal non-pg binaries:
./run.sh build
```

### Nsight Systems

```bash
sbatch m2_m1gpu_nsys.slurm
sbatch m2_unroll_nsys.slurm
sbatch m2_fused_nsys.slurm
```

Generated reports:

```bash
ls m1_gpu_profile.nsys-rep m2_unroll_profile.nsys-rep m2_fused_profile.nsys-rep
```

### Nsight Compute

```bash
sbatch m2_unroll_ncu.slurm
sbatch m2_fused_ncu.slurm
```

Current blocker:

```text
ERR_NVGPUCTRPERM - The user does not have permission to access NVIDIA GPU Performance Counters
```

### Compute Sanitizer

```bash
sbatch m2_unroll_sanitize.slurm
sbatch m2_fused_sanitize.slurm
```

## Profiling Notes

### Basic GPU Convolution

Things to capture:

- dominant kernels
- memory copy share
- whether Conv1 or Conv2 dominates
- whether kernel launch structure matches expectation

Notes:

- M1 baseline is already correctness-stable.
- Batch `10000` total `Op Time` baseline is `57.1786 ms`.
- `nsys` at batch `100` shows one student kernel, `conv_forward_kernel`, accounting for essentially all GPU kernel time.
- In the first `nsys` run, Conv2 `Op Time` (`0.483265 ms`) is larger than Conv1 (`0.222065 ms`).

### Current status

- The first implementation pass is complete for both unroll and fused.
- Delta validation completed successfully for both first-pass kernels.
- The new `m2_selfcheck.sh` should be the default correctness gate before any profiling.

### Unrolled Version

Things to capture:

- cost of unrolling
- cost of matmul
- cost of permutation
- whether unrolling dominates more than expected

Notes:

- First correctness pass is successful on public and edge batches.
- Batch `10000` total `Op Time` from self-check: `163.8049 ms`.
- This is comfortably below the `< 1200 ms` threshold and reasonably close to the README guidance of about `200 ms`.
- `compute-sanitizer` finished with `ERROR SUMMARY: 0 errors`.
- `nsys` at batch `100` shows three student kernels:
  - `matrix_unrolling_kernel`: `55.3%` of GPU kernel time
  - `matrixMultiplyShared`: `40.4%`
  - `matrix_permute_kernel`: `4.1%`
- This matches the expected structure of the unrolled pipeline: extra kernel launches and explicit unroll work add noticeable overhead.

### Fused Version

Things to capture:

- whether fusion reduces global memory traffic
- whether kernel count is reduced
- whether runtime drops as expected

Notes:

- First correctness pass is successful on public and edge batches.
- Batch `10000` total `Op Time` from self-check: `81.5453 ms`.
- Fusion is already materially faster than the unrolled pipeline and below the `< 200 ms` full-credit threshold.
- The first measured speedup is roughly `163.8049 / 81.5453 ≈ 2.01x` versus unroll.
- `compute-sanitizer` finished with `ERROR SUMMARY: 0 errors`.
- `nsys` at batch `100` shows a single dominant student kernel, `matmul_conv_fused`, accounting for `99.5%` of GPU kernel time.
- Compared with unroll at batch `100`, fused reduces the measured `Op Time` sum from about `3.093 ms` to about `0.961 ms`, which is roughly a `3.2x` speedup in the profiled run.

## Bugs and Debugging Notes

Use this section for bugs that are easy to forget but expensive to rediscover.

### Bug Template

```text
Symptom:
Root cause:
How I detected it:
Fix:
How I confirmed the fix:
```

### Known M1 carry-over lesson

Symptom:

- correctness can fail only at large batch sizes

Root cause:

- unsafe launch configuration can exceed CUDA grid limits

How I detected it:

- boundary batch tests and explicit CUDA error checking

Fix:

- safer launch mapping and `CUDA_CHECK(...)`

How I confirmed the fix:

- self-check across public and edge batch sizes

## What I Want to Compare at the End

When M2 is finished, I want to compare:

1. basic GPU convolution vs unroll
2. unroll vs fused
3. correctness stability across batch sizes
4. total Op Time at batch `10000`
5. profiler evidence for where the speedup came from

## Final Reflection

Fill this section after M2 is done.

### What worked well

- 

### What was harder than expected

- 

### Most important performance lesson

- 

### What I would do differently next time

- 

### What carries directly into later milestones

- 
