# M3 Master Plan

Goal: **만점 + extra credit + competition** (각 layer < 40 ms).

Baseline (from M2):
- fused @ B=10000: **81.5 ms total** (Conv1 ~11.5 ms, Conv2 ~45.5 ms on basic GPU — need to re-measure for fused per-layer)
- Target for M3 full credit: ≤ 60 ms
- Target for competition: each layer < 40 ms

## Scoring Map

| ID | Name | Base | Points | Priority |
|----|------|------|-------:|----------|
| req_0 | Streams | unfused | — (10% report) | P2 |
| req_1 | Tensor Cores (TF32) | fused | — (10% report) | **P0** (biggest single gain) |
| op_0 | Constant memory weights | fused | 1 | P1 |
| op_1 | `__restrict__` | fused | 1 | P1 |
| op_2 | Loop unroll | fused | 1 | P1 |
| op_3 | Parameter sweep | fused | 3 | P2 |
| op_4 | cuBLAS | unfused | 3 | P3 |
| op_5 | FP16 `__half2` | fused | 4 | P3 (risky for final) |
| op_6 | Joint register + shared tiling | fused | 4 | P3 |

Total optional: **17 points** (need 10, surplus = +7% extra credit).

## Phased Execution

### Phase A — Easy wins (target: +3 pts, quick baseline lift)

1. **op_0 constant memory** — mask → `__constant__`, `cudaMemcpyToSymbol`.
2. **op_1 `__restrict__`** — keyword on input/mask/output pointers.
3. **op_2 loop unroll** — `#pragma unroll` on K inner loops (K=7).

Deliverables:
- `project/m3/op_0/m3-forward.cu` (fused + const mem only)
- `project/m3/op_1/m3-forward.cu` (fused + `__restrict__` only)
- `project/m3/op_2/m3-forward.cu` (fused + unroll only)
- Combined version stacked into `project/src/layer/custom/m3-forward.cu` as the running baseline.

### Phase B — req_1 Tensor Cores (TF32) — **the core speedup**

- `nvcuda::wmma` API, 16×16×16 fragments.
- Replace fused kernel's shared-memory GEMM inner loop with `load_matrix_sync`/`mma_sync`/`store_matrix_sync`.
- TF32 accumulator in FP32 → full points.
- Expected Conv2 drop from ~45 ms to ~10 ms range.

Deliverables: `project/m3/req_1/m3-forward.cu`, then stack into final.

### Phase C — op_3 parameter sweep (3 pts)

On top of req_1:
- Tile size ∈ {8, 16, 32}
- Thread coarsening ∈ {1, 2, 4}
- Block configs (shapes).

Deliverables: table + graph in report, `project/m3/op_3/m3-forward.cu` with best config.

### Phase D — op_4 cuBLAS, op_5 FP16, op_6 joint register tiling

Each independent folder for individual evaluation only.
- op_4: unfused matmul replaced with cuBLAS `gemm`.
- op_5: `__half2` mixed precision (keep FP32 accumulator to preserve accuracy).
- op_6: register blocking on top of shared-memory tiling (508-level technique).

FP16 likely stays out of the **final stacked** submission (accuracy risk).

### Phase E — req_0 Streams

- unfused base, pinned host memory, 3-stream pipeline overlapping H→D + kernel + D→H.
- Required for report (10%), **excluded from final performance submission** (single stream mandated).

Deliverables: `project/m3/req_0/m3-forward.cu`.

### Phase F — Stacking & tuning

- Merge req_1 + op_0 + op_1 + op_2 + op_3 into `project/src/layer/custom/m3-forward.cu` (single stream).
- ncu profile each remaining bottleneck.
- Squeeze to < 40 ms/layer for competition.

### Phase G — Report + Drive upload

- Template: `docs/ECE408_SP26_netid_m3_report.docx`.
- Per optimization, 6 sections: name, theory, implementation + profile, expected vs observed, synergy, references.
- Export PDF → Gradescope.
- Copy `m3/` profile folders → Google Drive (shared to "Google Apps @ Illinois").

## Local Verification (no autograder)

Grader checks: accuracy + B=10000 Op Time. Our safety net:

1. **Batch sweep** (`scripts/m3/m3_selfcheck.sh`):
   - Standard: 100 / 1000 / 4095 / 4096 / 8767 / 10000.
   - Add edge: 1 / 127 / 4097 / 7919 / 9999.
   - Public batches ± 0.002; edge batches ≥ 0.80; B=10000 Op Time ≤ 60 ms.
2. **CPU↔GPU element-wise diff** (new utility):
   - Small batch (B=10). Dump CPU conv output and GPU conv output.
   - Report `max|diff|`, `mean|diff|`, `frac(|diff|>1e-3)`.
   - Guards against argmax-collapse bugs (numerical drift invisible to accuracy alone).
3. **Fused vs m3 diff** (regression detection):
   - Verified fused baseline as reference; diff vs current m3.
   - Catches TF32 drift and silent indexing bugs.
4. **compute-sanitizer** integrated into selfcheck (B=100 run, 0 errors required).
5. **Op time stability**: run binary 3× back-to-back, record min/avg/max. Only accept result if `worst < target`.
6. **competition mode check**: `./m3 --competition` at each stacking checkpoint.

## Immediate Next Steps

1. Create `project/m3/{req_0,req_1,op_0..op_6}/` stub folders with a working copy of the fused (or unfused for req_0/op_4) baseline as `m3-forward.cu`.
2. Phase A: implement op_0, op_1, op_2 individually + stack into the running final `m3-forward.cu`.
3. Write `scripts/m3/m3_selfcheck.sh` mirroring M2's pattern.
4. Measure Phase A result on B=10000 → log in `docs/M3_RESULTS.md` (to be created).

## Reference Lectures

- L17 Accelerating Matrix Operations → req_1.
- L16 Advanced Optimizations for Projects → op_3 / op_6 strategy.
- L08 Convolution Concept and Constant Cache → op_0.
- L20 Data Transfer and CUDA Streams → req_0.
- L13 Profiling on Nvidia GPU + `doc/profiling_lecture_notes.md` → methodology.
- L06 Data Locality and Tiled Matrix Multiply → op_3 / op_6 background.
