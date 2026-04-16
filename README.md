# batched-lenet-cuda

> 10,000-image LeNet-5 forward pass in **~28 ms** on a single NVIDIA A40 via fused convolution and Tensor Cores (TF32).

A study in CUDA kernel optimization: how to take a textbook convolutional layer
forward pass from a naïve one-thread-per-output kernel down to a Tensor-Core
fused matmul with N-coarsened register tiling, and what each step actually
buys you.

## Headline result

| Variant | Conv1 (ms) | Conv2 (ms) | Total (ms) | Speedup vs naïve |
|---|---:|---:|---:|---:|
| Naïve (one thread / output)             | 12.11 | 45.46 | 57.57 | 1.00× |
| Fused im2col + tiled matmul             | 41.77 | 39.12 | 80.89 | 0.71× |
| + Tensor Cores (TF32 WMMA)              | 21.33 | 47.51 | 68.84 | 0.84× |
| **+ N-coarsened register tiling (final)** | **16.40** | **11.67** | **28.07** | **2.05×** |

LeNet-5 variant — Conv1: 1→4 channels, K=7 over 86×86; Conv2: 4→16 channels,
K=7 over 40×40. Batch size 10,000. Single A40 (sm_86). Numbers are median Op
Time over 10 measured rounds after 5 warmup rounds. Reproduce with
`bench/run_all.sh`; reference data in [`bench/results.csv`](bench/results.csv).
See [`docs/OPTIMIZATION_JOURNEY.md`](docs/OPTIMIZATION_JOURNEY.md) for the
intermediate trial-and-error steps between "fused" and "register tiling".

## What's interesting here

- The "obvious" fused matmul is **slower** than the naïve kernel because of
  the integer-division overhead from on-the-fly im2col. Fixing that needs
  precomputed shared-memory lookup tables.
- Tensor Cores buy you almost nothing on Conv2 alone, because the kernel is
  memory-bound, not compute-bound.
- The 4× speedup on Conv2 only shows up after you raise arithmetic intensity
  by reusing the A-fragment across multiple N-tiles (register tiling).
- A40 measured bandwidth on Conv2 ≈ 745 GB/s — within 8% of the spec peak
  (696 GB/s effective). The kernel is pinned against a memory wall.

## Repo structure

```
batched-lenet-cuda/
├─ src/
│   ├─ main.cu              # minimal driver
│   ├─ conv/
│   │   ├─ baseline.cu      # one-thread-per-output
│   │   ├─ fused.cu         # fused im2col + tiled matmul
│   │   ├─ tensor_cores.cu  # WMMA TF32
│   │   └─ register_tiled.cu # FINAL: N-coarsened, register-blocked
│   └─ utils/
├─ docs/
│   ├─ OPTIMIZATION_JOURNEY.md  # phase-by-phase walkthrough with numbers
│   ├─ ARCHITECTURE.md          # network + workload analysis
│   └─ ROOFLINE.md              # bandwidth ceiling discussion
├─ bench/
│   ├─ run_all.sh
│   └─ results.csv
├─ tests/
│   └─ correctness.cu
├─ data/
│   └─ download.sh
└─ Makefile
```

## How to run

> Requires CUDA 11+ and an Ampere-class GPU (sm_80 or newer; tested on sm_86).

```bash
make                 # builds all variants
./bin/baseline 10000
./bin/register_tiled 10000
bench/run_all.sh     # runs all 4 variants, prints comparison table
```

## Documentation

- [`docs/OPTIMIZATION_JOURNEY.md`](docs/OPTIMIZATION_JOURNEY.md) — every trial,
  why it was tried, what happened, what got dropped.
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — the LeNet-5 variant and why
  the convolution is the bottleneck.
- [`docs/ROOFLINE.md`](docs/ROOFLINE.md) — why this kernel is memory-bound and
  what that implies for further work.

## Status

Work in progress — see `PORTFOLIO_PLAN.md` for the refactor roadmap. The CUDA
kernels themselves are stable and benchmarked; documentation and the public
driver are being lifted out of the original development tree.

## License

MIT — see `LICENSE`. The convolution kernels and surrounding harness in this
repository are my own work; concepts are referenced where they originate from
public sources (CUDA C++ Programming Guide, NVIDIA Tensor Core docs, the
*Programming Massively Parallel Processors* textbook).
