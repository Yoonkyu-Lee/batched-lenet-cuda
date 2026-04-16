# ECE-408 Profiling-Lecture

This folder is a profiling lecture example built around CUDA matrix multiplication. It compares three GEMM kernels, records NVTX ranges, and generates Nsight Systems and Nsight Compute reports so you can study both performance results and profiler workflow.

For the full lecture-style explanation, read [`lecture_notes.md`](./lecture_notes.md).

## Quick Start

Build:

```bash
bash build.sh
```

Run the executable:

```bash
sbatch run.slurm
```

Run Nsight Systems:

```bash
sbatch nsys.slurm
```

Run Nsight Compute:

```bash
sbatch ncu.slurm
```

## Main Files

- [`matmul_pageable.cu`](./matmul_pageable.cu): driver that runs all three kernels
- [`kernels/matmul`](./kernels/matmul): simple, shared, and improved shared GEMM kernels
- [`run.slurm`](./run.slurm), [`nsys.slurm`](./nsys.slurm), [`ncu.slurm`](./ncu.slurm): job scripts
- [`profile_nsys`](./profile_nsys): generated Nsight Systems report output
- [`profile_ncu-rep`](./profile_ncu-rep): generated Nsight Compute report output

## Current Outputs

- Regular run logs: [`run.out`](./run.out), [`run.err`](./run.err)
- Nsight Systems logs: [`nsys.out`](./nsys.out), [`nsys.err`](./nsys.err)
- Nsight Compute logs: [`ncu.out`](./ncu.out), [`ncu.err`](./ncu.err)

The detailed note explains the folder structure, lecture takeaway, kernel differences, actual measured performance, and how to interpret the profiler outputs.
