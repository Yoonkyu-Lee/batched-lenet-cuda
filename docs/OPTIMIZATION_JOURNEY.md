# Optimization Journey

From a naïve 58 ms baseline to 28 ms with Tensor Cores and register tiling —
and why every step along the way made sense.

All numbers in this document are Op Time for the two-layer forward pass at
**B=10000** on a single **NVIDIA A40 (sm_86)**, median of 10 measured rounds
after 5 warmup rounds. Reproducible via `bench/run_all.sh`.

| Variant | Conv1 (ms) | Conv2 (ms) | Total (ms) |
|---------|-----------:|-----------:|-----------:|
| naïve baseline                                | 12.1 | 45.5 | 57.6 |
| fused im2col + tiled matmul                   | 41.8 | 39.1 | 80.9 |
| + Tensor Cores (TF32 WMMA, 4 warps)           | 21.3 | 47.5 | 68.8 |
| + precomputed column indices + 8 warps        | 19.4 | 45.3 | 64.6 |
| + shared-memory K-index lookup                | 19.4 | 45.3 | 64.6 |
| **+ N-coarsened register tiling (final)**     | **16.4** | **11.7** | **28.1** |

The story is not monotonic — one of the early "optimizations" is slower than
the baseline, and Tensor Cores do almost nothing on Conv2 until a later fix
unlocks them. Each row below explains why the number moved.

---

## 0. Naïve baseline (57.6 ms)

One thread computes one output element. Each thread iterates
`(Channel, p, q)` and accumulates the dot product directly from global
memory. Grid maps `Map_out` into `grid.x` (folded with width tiles) and
`Batch` into `grid.z` to avoid overflowing the 65,535 limit on large
batches.

This is a surprisingly strong baseline because adjacent threads in a warp
have adjacent `w` coordinates, so their reads of `input[b, c, h+p, w+q]` are
fully coalesced, and the `mask` reads broadcast across the warp via L1.

No reuse across threads, but also very little overhead. Everything that
follows has to beat 57.6 ms.

## 1. Fused im2col + tiled matmul (80.9 ms, **slower**)

This is the textbook approach: restructure conv as a matmul with shared
memory tiles of mask (A) and unrolled input (B). The "fused" part is that
the unrolled input is computed on the fly during tile loading rather than
materialized into a separate buffer.

The result is 40% **slower** than the naïve kernel. Why?

Every tile element of B now requires decoding a linear matrix index back
into `(b, channel, p, q, h_out, w_out)`:

```cpp
int ch = input_row / (K * K);
int koff = input_row % (K * K);
int p = koff / K;
int q = koff % K;
int b = col / image_size;
int image_offset = col % image_size;
int h_out = image_offset / Width_out;
int w_out = image_offset % Width_out;
```

That's **eight integer divisions/modulos per element load**. Integer divide
is ~20–30 cycles on GPU. With 256 threads × many tiles loading B, the
division cost dominates over the memory savings from sharing tiles.

This was the big surprise: the "obvious" optimization made things worse.

## 2. Tensor Cores with WMMA TF32 (68.8 ms)

Replace the manual inner dot product with `wmma::mma_sync` on a 16×16×8
TF32 fragment. Each warp owns one 16×16 output tile, cooperatively loads
the A (mask) tile and B (unrolled input) tile into shared memory, then
runs a single MMA per K-step.

| Layer | Fused | + Tensor Cores | Δ |
|---|---:|---:|---:|
| Conv1 | 41.8 | 21.3 | **−20.5 ms** |
| Conv2 | 39.1 | 47.5 | **+8.4 ms** |

Tensor Cores help Conv1 a lot, but make Conv2 slightly *worse*. That's
because:

- Conv1's small K dimension (49) means most of the runtime is loop overhead.
  Tensor-core MMAs reduce that dramatically.
- Conv2's K dimension (196) with a tighter MMA loop still can't absorb the
  memory-bound B-tile loads. The same integer-divide overhead from step 1
  is still there — we just made the compute part faster.

Tensor Cores on their own aren't enough. The memory side has to be fixed
separately.

## 3. Precomputed column indices + 8 warps/block (64.6 ms)

Key observation: inside the K-loop, the division that decodes
`(b, h_out, w_out)` from each tile column is being re-done on every K-step.
But those values **don't change across the K-loop** — they depend only on
the column index, which is fixed per warp.

Precompute them once, store in shared memory, reuse across all K-steps:

```cpp
if (laneId < WMMA_N) {
    int gc = tile_col + laneId;
    col_b[laneId] = gc / image_size;
    int ioff = gc % image_size;
    col_h[laneId] = ioff / Width_out;
    col_w[laneId] = ioff % Width_out;
}
__syncwarp();
// K-loop now reads col_b[c], col_h[c], col_w[c] instead of dividing.
```

Simultaneously doubled `WARPS_PER_BLOCK` from 4 to 8 for better memory
latency hiding across outstanding loads.

Effect: Conv1 drops from 21.3 → 19.4 ms. Conv2 improves slightly to 45.3 ms.
Total 64.6 ms. The division-reduction matters more for Conv1 because its
K-loop is shorter.

## 4. Shared-memory K-index lookup (marginal on Total, unlocks step 5)

The remaining divisions in the K-loop decode the *row* index into
`(channel, p, q)`. Since the unrolled-row count is bounded (max 196 for
Conv2), precompute a `(ch, p, q)` lookup into shared memory at block entry:

```cpp
for (int k = threadIdx.x; k < unrolled_rows; k += BLOCK_SIZE) {
    sh_ch[k] = k / (K*K);
    int koff = k - sh_ch[k] * (K*K);
    sh_p[k]  = koff / K;
    sh_q[k]  = koff - sh_p[k] * K;
}
__syncthreads();
// K-loop now reads sh_ch[gk], sh_p[gk], sh_q[gk].
```

On its own, this is nearly a wash on Conv2 (45.3 ms unchanged). But it
removes the last integer-division bottleneck, which is what lets the next
step actually deliver its benefit.

### Why not use `__constant__` memory instead?

An earlier trial put the `(ch, p, q)` lookup in constant memory. Result:
**Conv2 went from 47 → 69 ms** — much worse.

Constant memory broadcasts efficiently only when every lane in a warp
reads the **same** address. Our K-loop load pattern has different lanes
reading different `gk` values → constant memory serializes the accesses.
Shared memory has 32 banks and handles the same pattern bank-free.

This kind of negative result is why it pays to measure rather than guess.

## 5. N-coarsened register tiling (28.1 ms, **the real win**)

Here's the insight: with Tensor Cores working and the K-loop stripped of
division overhead, Conv2 is **memory-bound on the B-tile loads**. Every
K-step reads 128 floats of B from global memory through shared memory, for
a single 16×16 output tile. That's ~0.5 flops/byte for Conv2 — nowhere
close to the A40's TF32 ridge point (~53 flops/byte).

To raise arithmetic intensity, have each warp compute **multiple N-tiles**
with the **same A-fragment**:

```cpp
wmma::fragment<..., accumulator, ...> c_frag[COARSEN_N];

for (int k_start = 0; k_start < unrolled_rows; k_start += WMMA_K) {
    // Load A[16x8] once per K-step.
    wmma::load_matrix_sync(a_frag, my_a, WMMA_K);

    for (int n = 0; n < COARSEN_N; n++) {
        // Load B[8x16] for N-tile n.
        wmma::load_matrix_sync(b_frag[n], my_b_base + n * B_SIZE, WMMA_N);
        wmma::mma_sync(c_frag[n], a_frag, b_frag[n], c_frag[n]);
    }
}
```

With `COARSEN_N = 4` and `WARPS_PER_BLOCK = 8`, each warp keeps 4
accumulator fragments (= 4 × (16 × 16 / 32) = 128 floats) resident in
registers across the entire K-loop. The A-fragment is loaded from shared
memory once and reused 4 times.

Net effect: Conv2 drops from 45.3 → **11.7 ms** (−74%). Conv1 also benefits
(19.4 → 16.4 ms) from reduced launch overhead due to 4× fewer grid blocks.

### Tuning the two knobs

`(WARPS_PER_BLOCK, COARSEN_N)` was swept; 8×4 came out on top:

| WARPS | COARSEN_N | Tiles/block | Conv1 | Conv2 | Total |
|------:|----------:|------------:|------:|------:|------:|
| 4 | 4 | 16 | 16.7 | 12.2 | 28.9 |
| 4 | 8 | 32 | 18.8 | 11.8 | 30.5 |
| 4 | 2 | 8  | 17.2 | 24.4 | 41.7 |
| 8 | 2 | 16 | 17.4 | 23.0 | 40.4 |
| **8** | **4** | **32** | **16.5** | **11.7** | **28.2** |
| 2 | 8 | 16 | 18.5 | 11.7 | 30.2 |

Going below 4 in the COARSEN_N dimension re-exposes the memory wall
(Conv2 regresses sharply). Going above 4 pushes register pressure per warp
to the point where occupancy drops. 8 warps are the sweet spot for latency
hiding at this working-set size.

## 6. Things that were tried but didn't help

Worth noting because they're in the natural sequence of things to try:

- **Shared-memory tile 32×32**. Kills occupancy (1,024 threads/block on
  sm_86) and runs at 113 ms total. Bad.
- **`__constant__` memory K-index table** (described in §4 above). Conv2
  goes from 47 → 69 ms because of warp-serial access to scattered constant
  addresses.
- **16 tiles/block with 2 warps, COARSEN_N=8**. 30.2 ms — close, but the
  asymmetric warp/coarsen split pushes Conv1 up.
- **FP16 (`__half`) with FP32 accumulator**. Correct (matches accuracy at
  public checkpoints) but **88 ms** total — the host-side FP16 conversion
  kernel and extra memory traffic for the half-precision mirrors dominate
  any compute savings. Kept in the repo as a demonstration; not used in the
  final.
- **cuBLAS GEMM on a pre-materialized unrolled matrix**. 155 ms — cuBLAS
  is very fast at the matmul itself, but the im2col materialization for
  B=10000 is expensive enough to lose the race to the fused kernel.

## Key takeaways

1. **Measure every step.** Several "obvious" optimizations slowed things
   down on first attempt; the profiler and simple A/B timings caught it.
2. **Tensor Cores don't rescue a memory-bound kernel.** You need to raise
   arithmetic intensity first. Register tiling was the step that made the
   tensor core hardware earn its keep.
3. **Divisions in the hot loop are real.** A few lines of precomputation
   in shared memory were worth several ms per layer.
4. **Shared memory > constant memory for scattered lane access.** Constant
   memory's broadcast behavior works only when every lane reads the same
   address.
5. **Two different layers can want different tunings.** The (8, 4) sweep
   winner is best for Conv2 (the bottleneck) and acceptable for Conv1,
   which is what matters when Conv2 dominates total time.
