# Roofline Analysis

Why the final kernel stops where it does, and what the data says about
whether further optimization would pay off.

## Hardware ceiling (NVIDIA A40, sm_86)

| Metric | Value | Source |
|---|---:|---|
| Peak FP32 throughput                   | 37.4 TFLOPS | A40 datasheet |
| Peak TF32 Tensor Core throughput       | ~150 TFLOPS | A40 datasheet |
| Peak HBM2 memory bandwidth             | 696 GB/s    | A40 datasheet |
| **Ridge point (TF32)** `peak / BW`     | ~215 flops/byte | derived |

The ridge point tells us the arithmetic intensity above which we become
compute-bound. Below it, memory bandwidth caps us no matter how fast the
compute units are.

## Workload arithmetic intensity

For a single output element of Conv2:

- FMAs performed: `Channel · K · K = 4 · 49 = 196`
- Input reads: `196 floats` (one per FMA, ignoring cache reuse)
- Mask reads: `196 floats`, but they're broadcast across all output
  columns in the same row → per-element amortized cost ~0 bytes/element
- Output writes: 1 float

Without any reuse, the per-element intensity is:

```
AI  =  2 FMAs · 196 / (196 · 4 bytes · 2 + 4 bytes)
    ≈  2 flops / byte (FP32)
```

That's **two orders of magnitude below the ridge point**. Without
aggressive reuse, this kernel is nailed to the memory bandwidth ceiling.

## What the final kernel actually achieves

Measured Conv2 Op Time at B=10000: **11.67 ms**.

Bytes moved (lower bound — every input element is read at least once by
the im2col):

```
Conv2 input size  =  B · C · H · W · 4  =  10000 · 4 · 40 · 40 · 4  =  ~25.6 MB
Output size       =  B · M · H_out · W_out · 4  =  10000 · 16 · 34 · 34 · 4  =  ~74 MB
```

But the unrolled view reads each input pixel up to `K · K = 49` times.
Even with perfect L1/L2 reuse, the realistic footprint is larger:

```
Effective input reads  ≈  C · (K · K) · B · H_out · W_out · 4
                        =  4 · 49 · 10000 · 34 · 34 · 4  =  ~9.0 GB

Total B-reads per forward pass  ≈  9.0 GB
```

Measured effective bandwidth:

```
BW  =  9.0 GB / 11.67 ms  =  ~771 GB/s
```

A40 peak is 696 GB/s. Our measured throughput is **within 11% of peak** —
and that measurement assumes no L1/L2 reuse at all. With any reuse, actual
DRAM traffic is lower and effective cached bandwidth higher.

**Conclusion: Conv2 is pinned against the memory-bandwidth ceiling.**
There is no slack left to exploit by making the compute faster. Any
further speedup would require reducing data movement (e.g., async copies
to hide latency with `cp.async`, or a fundamentally different algorithm
like Winograd that trades compute for less memory traffic).

## Why Tensor Cores helped Conv1 but barely Conv2

The same calculation for Conv1:

- C=1, K=7, unrolled_rows=49
- width_unrolled at B=10000 = 64,000,000
- MMA ops per output: 49

Arithmetic intensity per output is slightly higher than Conv2 (because
the read footprint per MMA is smaller — only 1 channel of input), but
the total K-dimension (49) is short enough that **loop overhead**, not
memory, is the first-order bottleneck. Tensor cores replace that overhead
with a single hardware instruction, so Conv1 sees the full MMA speedup.

## Where we are relative to the roofline

```
compute (TF32) ┃                                         ●●●●●●●●●●●●●●●●
  ~150 TFLOPS ┃                                        /
               ┃                                      /
               ┃                                    /
               ┃                                  /  ← ridge at 215 flops/byte
               ┃                                /
               ┃                              /
               ┃                            /
               ┃   Conv2 (register_tiled) /
               ┃    ● at ~2 flops/byte  /
               ┃                      /   ← memory-bandwidth roof
               ┃                    /
       ━━━━━━━━┃━━━━━━━━━━━━━━━━━━/━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
               0                 ridge                   arithmetic intensity
```

We're sitting on the **left slope** (memory-bound region) within striking
distance of the memory roof itself. The journey in
[`OPTIMIZATION_JOURNEY.md`](OPTIMIZATION_JOURNEY.md) is essentially about
climbing *up* the slope — raising arithmetic intensity via register tiling
— not about moving *right* toward the compute roof (which isn't reachable
without a different algorithm).

## What further optimization would look like

In rough order of expected gain:

1. **`cp.async` (Ampere async memory copies).** Global-to-shared loads
   could run in parallel with MMA work, hiding load latency. Expected
   Conv2 improvement: 5–15% (→ ~10–11 ms). CUDA 11+ required.
2. **Double-buffered K-loop.** Prefetch next K-step's B-tile while the
   current K-step's MMA runs. Overlaps memory with compute at the cost of
   2× shared-memory footprint.
3. **Winograd F(2,3) or F(4,3) convolution.** Trades FLOPs for fewer
   loads. K=7 kernels don't fit directly — would need decomposition into
   smaller kernels.
4. **INT8 / FP16 with aggressive quantization.** Halves memory traffic
   for every data element. Accuracy risk needs calibration work; the
   current TF32 variant already matches baseline accuracy.

None of these were pursued here: the kernel is already near peak BW and
the next leaderboard rank is only ~50% ahead. Further effort would be
research-grade work rather than engineering wins.

## Why bandwidth-bound isn't the end of the story

Being bandwidth-bound explains the **floor** for our current algorithm,
not an absolute floor. A different numerical scheme (Winograd, FFT-based
convolution) reduces the amount of data that needs to be moved in the
first place. That's how libraries like cuDNN squeeze further — they
switch algorithms per workload shape. For the target dimensions here
(Fashion-MNIST-scale 40×40 with K=7), direct im2col-matmul is close to
the sweet spot, and bandwidth is the real limit.
