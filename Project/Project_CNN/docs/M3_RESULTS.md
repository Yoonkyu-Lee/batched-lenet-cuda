# M3 Optimization Results Log

Track every trial and its measured outcome. Append-only — never overwrite past data.

Baseline reference (M2):
- **Fused** @ B=10000: Conv1 ~37.7 ms, Conv2 ~47.9 ms → **total 81.5 ms**
- **Basic GPU (M1)** @ B=10000: Conv1 ~11.5 ms, Conv2 ~45.5 ms → **total 57 ms**

Targets:
- Full credit: ≤ 60 ms
- Competition: each layer < 40 ms

---

## Phase A — Easy optimizations (op_0 + op_1 + op_2 stacked on fused)

| Change | Description |
|--------|-------------|
| op_0 | Mask in `__constant__` memory; kernel reads from `c_mask[]` instead of global pointer |
| op_1 | `__restrict__` on kernel pointer params |
| op_2 | `#pragma unroll` on inner dot-product loop (MATMUL_TILE_WIDTH=16) |

### Result

| Layer | Fused (M2 baseline) | Phase A stacked | Change |
|-------|-----:|-----:|-----:|
| Conv1 | 37.7 ms | 37.7 ms | 0% |
| Conv2 | 47.9 ms | 47.9 ms | 0% |
| **Total** | **81.5 ms** | **85.6 ms** | **+5% (worse)** |

**Analysis**: No real improvement. These three ops are "points-only" optimizations — the compiler already auto-unrolls, infers non-aliasing, and the mask is small enough to fit in L1 cache regardless of constant memory. The slight regression likely comes from constant-memory indexing overhead (flat `row*unrolled_rows+col` vs. the multi-dim `mask[((row*Channel+c)*K+p)*K+q]` that the original fused kernel used).

**Accuracy**: All 11 batches PASS (public + edge). ✅

---

## Phase B — req_1 Tensor Cores (TF32)

Goal: Replace the tiled-matmul inner loop with WMMA `mma_sync` (16×16×8 TF32). Expecting major speedup on the matmul-intensive Conv2.

### Trial B.0 — Baseline TC (4 warps/block, direct index computation)

TC version of fused kernel: each warp loads 16×8 mask tile + 8×16 input tile into shared memory, uses `wmma::mma_sync`, stores with permutation.

| Layer | Phase A | Trial B.0 | Change |
|-------|-----:|-----:|-----:|
| Conv1 | 37.7 ms | 31.4 ms | -17% |
| Conv2 | 47.9 ms | 47.9 ms | 0% |
| **Total** | **85.6 ms** | **79.3 ms** | **-7%** |

**Analysis**: TC helped Conv1 moderately but Conv2 was unchanged. Root cause: kernel is **memory-bound**, not compute-bound. Arithmetic intensity ~3.9 flops/byte vs A40's ridge point ~53 flops/byte. Integer division/modulo for implicit unrolling (8 divs per element load) dominates. TC speeds up compute that wasn't the bottleneck.

### Trial B.1 — TC + precomputed column indices (4 warps/block)

Precompute `(b, h_out, w_out)` for each of the 16 tile columns once in shared memory before the K-loop. Eliminates 3 divs/mods per B-tile element load. Also uses `gk - ch*KK` instead of `gk % KK` to avoid modulo.

| Layer | Trial B.0 | Trial B.1 | Change |
|-------|-----:|-----:|-----:|
| Conv1 | 31.4 ms | **21.0 ms** | -33% |
| Conv2 | 47.9 ms | **47.5 ms** | -1% |
| **Total** | **79.3 ms** | **68.6 ms** | **-14%** |

**Analysis**: Conv1 improved significantly (div elimination matters more when unrolled_rows=49 → only 7 K-iterations → less work to amortize div cost). Conv2 barely changed (196 K-iterations → div cost is a smaller fraction). **Remaining bottleneck is global memory latency on scattered input reads.**

### Trial B.2 — Regular fused with TILE_WIDTH=32 (no TC)

Hypothesis: larger tiles reduce iteration count and sync overhead.

| Layer | Phase A (TW=16) | Trial B.2 (TW=32) | Change |
|-------|-----:|-----:|-----:|
| Conv1 | 37.7 ms | 58.9 ms | +56% (worse) |
| Conv2 | 47.9 ms | 54.0 ms | +13% (worse) |
| **Total** | **85.6 ms** | **113.0 ms** | **+32% (much worse)** |

**Analysis**: 32×32 = 1024 threads per block kills occupancy on sm_86. Discarded.

### Trial B.3 — TC + precomputed + 8 warps/block

Same as Trial B.1 but WARPS_PER_BLOCK=8 (256 threads). More warps → better memory latency hiding.

| Layer | Trial B.1 (4w) | Trial B.3 (8w) | Change |
|-------|-----:|-----:|-----:|
| Conv1 | 21.0 ms | 21.1 ms | 0% |
| Conv2 | 47.5 ms | 47.0 ms | -1% |
| **Total** | **68.6 ms** | **68.1 ms** | **-1%** |

**Analysis**: 8 warps gave negligible improvement. Already enough warps for latency hiding at 4/block. Discarded.

---

## Summary Table

| Variant | Conv1 (ms) | Conv2 (ms) | Total (ms) | Accuracy | Notes |
|---------|----:|----:|----:|:---:|-------|
| M1 basic GPU | 11.5 | 45.5 | 57.0 | ✅ | Reference |
| M2 fused | ~37.7 | ~47.9 | 81.5 | ✅ | Baseline for M3 |
| Phase A (op0+1+2) | 37.7 | 47.9 | 85.6 | ✅ | No real gain |
| B.0 TC baseline | 31.4 | 47.9 | 79.3 | ✅ | TC helps Conv1 only |
| B.1 TC+precomp 4w | 21.0 | 47.5 | 68.6 | ✅ | **Best so far** |
| B.2 fused TW=32 | 58.9 | 54.0 | 113.0 | ✅ | Discarded |
| **B.3 TC+precomp 8w** | **21.1** | **47.0** | **68.1** | ✅ | **Phase B winner** |

---

---

## Phase C — op_3 Parameter Sweep

Base: B.3 winner (TC + precomputed cols + 8 warps, 68.1 ms).

### Trial C.0 — K-index precompute in constant memory

Precompute `(ch, p, q)` for all k ∈ [0..unrolled_rows) in constant memory arrays. Goal: eliminate the 3 remaining divs in the B-tile inner loop.

| Layer | B.3 | C.0 | Change |
|-------|-----:|-----:|-----:|
| Conv1 | 21.1 | 24.5 | +16% (worse) |
| Conv2 | 47.0 | 69.2 | +47% (much worse) |
| **Total** | **68.1** | **93.7** | **+38%** |

**Analysis**: Constant memory serializes when warp lanes read different addresses (`c_k_ch[gk]` where gk differs per lane). Division was faster because it's purely register-local. **Discarded.**

### Trial C.1 — WARPS_PER_BLOCK=2

Fewer threads/block → more blocks in flight → potentially better SM occupancy.

| Layer | B.3 (8w) | C.1 (2w) | Change |
|-------|-----:|-----:|-----:|
| Conv1 | 21.1 | 22.4 | +6% (worse) |
| Conv2 | 47.0 | 48.0 | +2% (worse) |
| **Total** | **68.1** | **70.4** | **+3%** |

**Analysis**: Too few warps per block reduces latency hiding. 8 warps remains optimal. **Discarded.**

### Phase C Parameter Sweep Summary Table

| Config | Warps | K-index | Conv1 | Conv2 | Total | Status |
|--------|------:|---------|------:|------:|------:|--------|
| B.3 (base) | 8 | runtime div | 21.1 | 47.0 | 68.1 | **Winner** |
| C.0 | 8 | const mem LUT | 24.5 | 69.2 | 93.7 | Discarded |
| C.1 | 2 | runtime div | 22.4 | 48.0 | 70.4 | Discarded |
| B.1 | 4 | runtime div | 21.0 | 47.5 | 68.6 | Close 2nd |

### Trial C.2 — 4 warps + K-index precompute in SHARED memory

Precompute `(ch, p, q)` for all k ∈ [0..unrolled_rows) in shared memory (cooperatively loaded once per block). Unlike constant memory, shared memory broadcasts when multiple lanes read the same address.

| Layer | B.3 | C.2 | Change |
|-------|-----:|-----:|-----:|
| Conv1 | 21.1 | **19.2** | -9% |
| Conv2 | 47.0 | 48.2 | +3% |
| **Total** | **68.1** | **67.4** | **-1%** |

**Analysis**: Conv1 improved (shared mem broadcast eliminates div cost). Conv2 slightly worse — `__syncthreads()` overhead for K-index table. Mixed.

### Trial C.3 — 8 warps + K-index precompute in SHARED memory ★

Combines C.2's shared memory approach with B.3's 8 warps.

| Layer | B.3 | C.3 | Change |
|-------|-----:|-----:|-----:|
| Conv1 | 21.1 | **19.4** | -8% |
| Conv2 | 47.0 | **45.3** | **-4%** |
| **Total** | **68.1** | **64.6** | **-5%** |

**Analysis**: Best result yet! 8 warps provides latency hiding that masks the `__syncthreads()` cost. Conv2 finally moved below 47 ms. **New winner.**

### Updated Phase C Parameter Sweep Summary Table

| Config | Warps | K-index | Conv1 | Conv2 | Total | Status |
|--------|------:|---------|------:|------:|------:|--------|
| B.3 (prev best) | 8 | runtime div | 21.1 | 47.0 | 68.1 | Superseded |
| C.0 | 8 | const mem LUT | 24.5 | 69.2 | 93.7 | Discarded |
| C.1 | 2 | runtime div | 22.4 | 48.0 | 70.4 | Discarded |
| C.2 | 4 | shared mem | 19.2 | 48.2 | 67.4 | Close 2nd |
| **C.3** | **8** | **shared mem** | **19.4** | **45.3** | **64.6** | **Winner** |

**Conclusion**: Shared memory K-index precompute + 8 warps is the best configuration at 64.6 ms. 4.6 ms away from the 60 ms target. Further improvement may come from Phase D optimizations (register tiling, FP16).

---

## Key Insights So Far

1. **Fused is slower than basic GPU** (81.5 vs 57 ms) due to integer div/mod overhead for implicit unrolling. The basic kernel reads input with direct `(b,c,h+p,w+q)` indexing → coalesced, no divs.
2. **TC provides moderate improvement on Conv1** where compute fraction is higher, but Conv2 stays memory-bound at ~48 ms regardless.
3. **Precomputing column indices eliminates the biggest div overhead** — Conv1 went from 37.7→21.0 ms with this single change.
4. **Conv2 (47.5 ms) is the primary target** for reaching ≤60 ms total. Need ~8 ms reduction minimum.

---

## Phase D — op_6 Register Tiling (N-coarsening)

Each warp computes COARSEN_N=4 output tiles (16×16 each), keeping 4 c_frag accumulators in registers. The mask A-fragment is loaded once per K-step and reused across all 4 B-tiles → 4× higher arithmetic intensity.

| Layer | C.3 (prev best) | op_6 (COARSEN_N=4) | Change |
|-------|-----:|-----:|-----:|
| Conv1 | 19.4 | **16.7** | -14% |
| Conv2 | 45.3 | **12.2** | **-73%** |
| **Total** | **64.6** | **28.9** | **-55%** |

**Analysis**: Massive improvement. Conv2 went from 45.3 → 12.2 ms because:
- A-fragment reuse: 4× reduction in mask loads per output element
- Higher arithmetic intensity: pushes the kernel closer to compute-bound
- Fewer blocks launched: 4× fewer grid entries, reducing launch overhead

**Accuracy**: ✅ (0.8714 @ B=10000, 0.86 @ B=100)

### Updated Summary Table

| Variant | Conv1 | Conv2 | Total | Status |
|---------|------:|------:|------:|--------|
| M2 fused | 37.7 | 47.9 | 81.5 | Baseline |
| Phase A | 37.7 | 47.9 | 85.6 | op_0+1+2 |
| B.3 TC 8w | 21.1 | 47.0 | 68.1 | TC + precomp cols |
| C.3 TC smem kidx | 19.4 | 45.3 | 64.6 | + smem K-index |
| **op_6 coarsen4** | **16.7** | **12.2** | **28.9** | **★ Competition ready** |

### Full selfcheck validation (op_6 stacked)

Run: `selfcheck/m3/20260415_224303/`

| Batch | Accuracy | Op Time sum | Max layer | Status |
|------:|---------:|------------:|----------:|--------|
| 100 | 0.86 | 0.52 ms | 0.30 ms | ✅ |
| 1000 | 0.886 | 3.90 ms | 2.28 ms | ✅ |
| 4095 | 0.871306 | 15.73 ms | 9.07 ms | ✅ |
| 4096 | 0.871338 | 15.75 ms | 9.09 ms | ✅ |
| 8767 | 0.871564 | 33.59 ms | 19.34 ms | ✅ |
| **10000** | **0.8714** | **38.29 ms** | **22.05 ms** | **✅ Full credit + competition** |
| 1 (edge) | 1.0 | 0.16 ms | 0.08 ms | ✅ |
| 127 (edge) | 0.88189 | 0.60 ms | 0.35 ms | ✅ |
| 4097 (edge) | 0.871369 | 15.73 ms | 9.07 ms | ✅ |
| 7919 (edge prime) | 0.872585 | 30.33 ms | 17.47 ms | ✅ |
| 9999 (edge) | 0.871387 | 38.29 ms | 22.05 ms | ✅ |

**OVERALL: PASS** — 11/11 batches, sum 38.3 ms (quick-test warm run showed 28.9 ms; selfcheck's cold sequential batches produce more conservative numbers). Competition-ready across the board.

---

---

## Phase F — Competition Push (Tier 1 Parameter Sweep)

Base: op_6 stacked at 28.9 ms (quick-test) / 38.3 ms (selfcheck). Sweep the
`(WARPS_PER_BLOCK, COARSEN_N)` pair to find the best combination.

| Trial | WARPS | COARSEN_N | tiles/blk | Conv1 | Conv2 | Total | Notes |
|-------|------:|----------:|----------:|------:|------:|------:|-------|
| base (op_6) | 4 | 4 | 16 | 16.7 | 12.2 | 28.9 | Starting point |
| D.1 | 4 | 8 | 32 | 18.8 | 11.8 | 30.5 | Conv1 regresses |
| D.2 | 4 | 2 | 8 | 17.2 | 24.4 | 41.7 | Conv2 regresses (less A reuse) |
| D.3 | 8 | 2 | 16 | 17.4 | 23.0 | 40.4 | Conv2 regresses |
| **D.4** | **8** | **4** | **32** | **16.5** | **11.7** | **28.2** | **★ New best** |
| D.5 | 2 | 8 | 16 | 18.5 | 11.7 | 30.2 | Conv1 regresses |

**Winner: D.4 (WARPS=8, COARSEN_N=4)**. Both dimensions doubled vs base → 2× total
warps per block, same per-warp coarsening. Better latency hiding + same A-frag
reuse.

### D.4 selfcheck validation (job 17652592)

All 11 batches PASS. B=10000: **sum 28.15 ms**, max layer **16.48 ms**.

| Batch | Accuracy | Op Time sum | Max layer |
|------:|---------:|------------:|----------:|
| 100 | 0.86 | 0.38 ms | — |
| 1000 | 0.886 | 2.90 ms | — |
| 4095 | 0.871306 | 11.58 ms | — |
| 4096 | 0.871338 | 11.58 ms | — |
| 8767 | 0.871564 | 24.71 ms | — |
| **10000** | **0.8714** | **28.15 ms** | **16.48 ms** |
| 1 | 1.0 | 0.15 ms | — |
| 127 | 0.88189 | 0.47 ms | — |
| 4097 | 0.871369 | 11.58 ms | — |
| 7919 | 0.872585 | 22.33 ms | — |
| 9999 | 0.871387 | 28.17 ms | — |

**Δ from op_6 base**: -10.1 ms on full selfcheck, -0.7 ms on warm quick-test.
**Competition posture**: 28.15 ms vs leaderboard #1 43.84 ms → **~15.7 ms lead**.

---

## Individual Optimization Results (per-folder, quick-test @ B=10000)

Each file in `project/m3/{req_*,op_*}/m3-forward.cu` was built and run in isolation by copying it into `project/src/layer/custom/m3-forward.cu`. Quick-test reads one warmed-up run.

| Folder | Base | Conv1 (ms) | Conv2 (ms) | Total (ms) | Accuracy | Notes |
|--------|------|-----------:|-----------:|-----------:|:--------:|-------|
| `op_0` const memory | fusion | ~37 | ~48 | ~85 | ✅ | Part of Phase A run |
| `op_1` __restrict__ | fusion | ~37 | ~48 | ~85 | ✅ | Part of Phase A run |
| `op_2` loop unroll | fusion | ~37 | ~48 | ~85 | ✅ | Part of Phase A run |
| `op_3` sweep winner | fusion+TC | 19.4 | 45.3 | **64.6** | ✅ | = C.3 config |
| `op_4` cuBLAS | unfused | 92.0 | 63.4 | 155.4 | ✅ 0.8714 | Job 17652119. vs M2 unfused baseline 163.8 ms → ~5% gain. Per-call unrolled-matrix malloc dominates. |
| `op_5` FP16 (`__half`) | fusion | 44.7 | 43.3 | 88.0 | ✅ 0.8714 | Job 17651805 |
| `op_6` register tiling (N-coarsen=4) | fusion+TC | 16.7 | 12.2 | **28.9** | ✅ | Job 17649422 |
| `req_0` Streams (3 streams) | unfused | 106.3 | 87.5 | 193.7 | ✅ 0.8714 | Job 17651740. Op Time now includes async H↔D copies (they happen inside `conv_forward_gpu`), so the raw sum is larger; actual overlap is visible in `nsys` timeline. |
| `req_1` TC baseline | fusion | 31.4 | 47.9 | 79.3 | ✅ | = B.0 |

All 9 folders build cleanly (verified by `scripts/m3/verify_all.sh`).

---

## Competition Mode Result (final submission)

Run `./m3 --competition` at B=10000 (job 17651992):

```
Conv-GPU (Conv1: 1→4 channels, 7x7)==
Competition Mode: warmup=5, measured=10
Op Time: 16.6602 ms

Conv-GPU (Conv2: 4→16 channels, 7x7)==
Competition Mode: warmup=5, measured=10
Op Time: 12.2047 ms

Test Accuracy: 0.8714
```

**Total: 28.86 ms** — median of 10 measured rounds after 5 warmup rounds.

Leaderboard (from `docs/260413_cnn_competition_leaderboard.md`, pulled ~2 days ago): top-3 ranked at 43.84 / 43.87 / 44.03 ms. Our 28.86 ms would place **#1 by a ~15 ms margin**.

## Ranking notes

- The `--competition` abort threshold is **40 ms per layer in the first warmup round**; our max is 16.66 ms, clears with wide margin.
- Competition requires: matrix unrolling + tiled matmul + single stream + matched baseline accuracy + all kernels in `conv_forward_gpu()`. Our final meets all of these (see spec check in the report / `CLAUDE.md`).
