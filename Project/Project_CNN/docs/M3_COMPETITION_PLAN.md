# M3 Competition Push Plan

**Goal**: minimize sum-of-OP-time for `./m3 --competition` @ B=10000 while keeping
accuracy exactly matched to basic GPU convolution.

## Current state

- Competition run (job 17651992): Conv1 16.66 ms, Conv2 12.20 ms → **total 28.86 ms**
- Leaderboard snapshot (2 days old, `docs/260413_cnn_competition_leaderboard.md`):
  - #1: 43.84 ms
  - #2: 43.88 ms
  - #3: 44.03 ms
- Our lead: **~15 ms** (≈35% margin)
- Competition warmup abort threshold: 40 ms / layer. Our max is 16.66 ms.

**Constraints we must preserve**:
1. matrix unrolling (fused is fine)
2. tiled matmul (WMMA counts)
3. single stream
4. all GPU kernel calls inside `conv_forward_gpu()`
5. accuracy must match basic GPU convolution (currently does: 0.8714 @ B=10000)

## Risk framing

- Current lead is already comfortable. Any change must be **reversible** and tested
  before committing to push.
- The leaderboard updates "at least weekly" — new submissions from others could
  appear between now and May 5. Keep some headroom.
- TF32 precision drift is the biggest accuracy risk; FP16 is even worse. Do not
  ship FP16 in the final.
- Always keep a known-good backup of `project/src/layer/custom/m3-forward.cu`
  before each trial (a stable `/tmp` copy + git).

## Trial roadmap (ordered by reward/effort)

### Tier 1 — Parameter sweep (low risk, ~15 min total)

Base: current op_6 stacked (4 warps × COARSEN_N=4).

| Trial | WARPS | COARSEN_N | tiles/block | Hypothesis |
|-------|------:|----------:|------------:|------------|
| D.1 | 4 | 8 | 32 | More A reuse per K-step |
| D.2 | 4 | 2 | 8 | Baseline for reuse sensitivity |
| D.3 | 8 | 2 | 16 | Same tiles/block as current, more warps |
| D.4 | 8 | 4 | 32 | Double everything |
| D.5 | 2 | 8 | 16 | Same tiles/block as current, fewer warps |

Each trial: edit two `#define`s, `./run.sh build`, `sbatch scripts/m3/m3_quick.slurm`,
record in `docs/M3_RESULTS.md`. Pick the best; revert if no gain.

**Stop condition**: best of D.1–D.5 beats 28.9 ms by ≥ 0.5 ms AND accuracy matches.

### Tier 2 — Bank-conflict / layout audit (low risk, ~20 min)

Use `ncu --set full` on current competition config:
- Check `smsp__sass_average_data_bytes_per_wavefront_mem_shared` — shared-mem
  bank conflicts on `my_a` / `my_b` / `sh_ch` / `col_*` accesses?
- Add padding (`[WMMA_K+1]` style) if conflicts > 10%.
- Revert if no improvement (padding costs shared-mem capacity).

**Stop condition**: ncu reports < 5% bank conflicts. If already low, skip.

### Tier 3 — cp.async for B-tile load (moderate risk, ~1 hr)

Only attempted if Tiers 1–2 stall.

- A40 is sm_86 (Ampere) → supports `cp.async` via `cuda::memcpy_async` or PTX.
- Replace the 32-thread per-element B-tile load with bulk cp.async into shared
  memory, followed by `cp.async.wait`. Skips the register round-trip.
- Expected gain: 5–15% on memory-bound Conv2.
- Risks:
  - Correctness (async semantics, alignment).
  - Compiler/driver edge cases — keep a revert path.

**Stop condition**: Conv2 drops below 10 ms OR no measurable gain after implementation.

### Tier 4 — Convolution-specific specialization (moderate risk, experimental)

Only if more gain still needed.

- **Conv1 specialization**: Map_out=4, unrolled_rows=49. The 16×16 WMMA output
  tile wastes 75% of rows. Try a non-WMMA specialized kernel for Conv1 only
  (reuse the op_6 kernel just for Conv2). Could also just let both layers keep
  WMMA since Conv1 is already 16 ms — not a big lever.
- **K-loop prefetch (software double buffer)**: load next K's B-tile into shared
  memory while current K's mma_sync runs. Overlaps gmem latency with compute.

## Do-not-do list

- **No FP16 / FP32 mixed for the final**: accuracy drift risk. FP16 variant
  stays in `project/m3/op_5/` only.
- **No streams in final**: `req_0` folder only. Competition forbids streams.
- **No moving kernels to prolog/epilog**: competition spec explicitly forbids it.
- **No changing `matmul.cu` / `matmul.h`**: not a graded file; grader uses the
  reference.
- **No modifying `run.sh`**: byte-identical to `release/main` required (CLAUDE.md).
- **No aggressive FP truncation** (e.g. pretending FP16 everywhere): breaks
  "match baseline accuracy" rule.

## Safety workflow per trial

1. `cp project/src/layer/custom/m3-forward.cu /tmp/m3_safe.cu` — backup
2. Make the smallest change possible (one `#define`, one loop tweak, etc.)
3. `./run.sh build` — must succeed
4. `TRIAL=<name> sbatch scripts/m3/m3_quick.slurm` — check accuracy first
5. If accuracy ≠ 0.8714 @ B=10000, **revert immediately** (`cp /tmp/m3_safe.cu ...`)
6. If accuracy OK and timing better: run full `sbatch scripts/m3/m3_selfcheck.slurm`
   to confirm edge batches
7. Record in `docs/M3_RESULTS.md` (append-only)
8. Promote best trial to the permanent baseline only after step 6 passes

## Final ship checklist (before git push)

- [ ] `./m3 --competition` completes without abort
- [ ] Accuracy 0.8714 @ B=10000
- [ ] `sbatch scripts/m3/m3_selfcheck.slurm` — all 11 batches PASS
- [ ] `./run.sh clean` before commit (remove outputs / binaries)
- [ ] `run.sh` byte-identical to release/main
- [ ] All 9 `project/m3/*/m3-forward.cu` files build (verify with `bash scripts/m3/verify_all.sh`)
- [ ] Git status clean; push

## Decision points

- After Tier 1: if best is already at or above current 28.9 ms, stop and ship.
- After Tier 2: ncu data feeds the report regardless. If no conflict, also stop.
- Tier 3/4 only if motivated (margin shrinks on next leaderboard update, or
  personal goal to break 20 ms).

## Reference budgets

A40 memory bandwidth: ~696 GB/s. For Conv2 at B=10000:
- Input reads dominate (196 × 11.56M = 2.27 G reads per warp across all blocks).
- With 12.2 ms → effective bandwidth ≈ 2.27e9 × 4 / 12.2e-3 / 1e9 ≈ 745 GB/s
  (already near peak).

This suggests **Conv2 is close to a memory-bandwidth floor** — further gains likely
require reducing data movement (cp.async, bigger tile reuse) rather than more FLOPs.
Conv1 has more slack but its absolute cost (16 ms) is smaller.
