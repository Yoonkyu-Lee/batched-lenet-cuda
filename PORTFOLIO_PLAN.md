# Portfolio Refactor Plan — `batched-lenet-cuda`

## Ground rules

1. **`archived/` is the source of truth — do not edit or delete anything inside.**
   It is the verbatim snapshot of the school project at the time of refactor.
2. New files live at the **repo root** (or in new top-level folders). The portfolio
   structure is built up from scratch by copying / rewriting selected pieces out of
   `archived/`.
3. Avoid material that visibly identifies as **coursework deliverables**, but
   ideas and patterns can be **paraphrased into a practitioner framing**:
   - ❌ Verbatim copy of course-provided files (`README_m1/m2/m3.md`, quiz docx,
     mini-dnn-cpp framework folders, `conv_cust.cc` as-is, `weights-86.bin`)
   - ❌ Files whose names or text mention "ECE408", "milestone", "PrairieLearn",
     "Gradescope", or specific course staff
   - ✅ Reworking the same techniques (im2col + WMMA + register tiling) under
     practitioner-style naming and explanations
   - ✅ Rewriting the host driver from scratch (replaces `conv_cust.cc`)
   - ✅ Reframing the "60 ms / 40 ms" grading budget as a "performance budget"
     for the README narrative
   - ✅ Citing concepts learned from textbook chapters / public papers / NVIDIA
     blogs (with a real citation)
4. Keep attribution honest: if a snippet is original, mark it; if adapted from
   a textbook / tutorial / NVIDIA sample, cite it. The point is to make the work
   stand on its own as "I built this" rather than "I submitted this for class".

## Goal

Transform the school CNN project into a recruiter-friendly portfolio repo
demonstrating deep CUDA optimization, with these signals:

- **Result-first README** (benchmark numbers, before/after table, roofline).
- **Clean, self-contained code** — only my own implementations, no course framework.
- **Reproducible benchmarks** — drop-in test harness on a public dataset.
- **Optimization journey doc** — phases A→F with measurements and reasoning.

## Target final structure

```
batched-lenet-cuda/
├─ README.md                ← top-level: value prop, results, how to run
├─ LICENSE                  ← MIT (or similar) for my own code
├─ docs/
│   ├─ OPTIMIZATION_JOURNEY.md  ← Phases A-F with numbers (rewrite of M3_RESULTS)
│   ├─ ARCHITECTURE.md          ← LeNet-5 variant, layer dims, why fused matters
│   ├─ ROOFLINE.md              ← bandwidth analysis, why kernel is memory-bound
│   └─ images/                  ← graphs, screenshots
├─ src/
│   ├─ main.cu                  ← minimal driver (load data, run conv, time it)
│   ├─ conv/
│   │   ├─ baseline.cu          ← naive, single-thread-per-output
│   │   ├─ fused.cu             ← fused unroll + tiled matmul
│   │   ├─ tensor_cores.cu      ← WMMA TF32 version
│   │   └─ register_tiled.cu    ← FINAL: N-coarsened register tiling
│   └─ utils/
│       ├─ cuda_check.h
│       └─ timing.h
├─ bench/
│   ├─ run_all.sh               ← runs all variants, prints comparison table
│   └─ results.csv              ← reference numbers from A40
├─ data/
│   └─ download.sh              ← script to fetch a public dataset (CIFAR-10? MNIST?)
├─ tests/
│   └─ correctness.cu           ← compares each variant against CPU reference
└─ archived/                    ← left as-is (private context, not on remote)
```

## Phased refactor plan

### Phase 1 — Skeleton (no code yet)

- Create `src/`, `docs/`, `bench/`, `data/`, `tests/` directories.
- Write `LICENSE` (MIT).
- Write a placeholder `README.md` listing the planned sections.
- Add `.gitignore` (build artifacts, datasets).

### Phase 2 — Pull out original kernels

Copy our own implementations from `archived/Project/Project_CNN/project/m3/` and
`archived/Project/Project_CNN/project/src/layer/custom/`. Specifically:

| Source (in archived) | Target | Notes |
|---|---|---|
| `project/m3/op_6/m3-forward.cu` (best) | `src/conv/register_tiled.cu` | Final variant; main showcase |
| `project/m3/req_1/m3-forward.cu` | `src/conv/tensor_cores.cu` | Standalone TC version |
| `project/src/layer/custom/kernel-fusion-forward.cu` | `src/conv/fused.cu` | Pre-TC fused baseline |
| `project/src/layer/custom/new-forward.cu` | `src/conv/baseline.cu` | One-thread-per-output |

Strip all references to `GPUInterface::conv_forward_gpu*` (course framework).
Convert each kernel into a standalone function with a clean signature:

```cpp
void conv_forward(
    const float* input, const float* mask, float* output,
    int Batch, int Map_out, int Channel, int Height, int Width, int K
);
```

### Phase 3 — Replace the framework with a minimal driver

Write `src/main.cu`:
- Allocate input / mask / output buffers (`cudaMalloc`)
- Generate or load test data
- Call the chosen kernel
- Verify against a CPU reference
- Print Op Time

Also write `src/utils/cuda_check.h` (CUDA_CHECK macro) and `src/utils/timing.h`
(cudaEventRecord-based timer).

### Phase 4 — Tests

`tests/correctness.cu`: compares each variant against a simple CPU loop on small
inputs (B=10, random data). Reports max absolute diff.

### Phase 5 — Benchmark harness

`bench/run_all.sh`:
- Builds `main` with each variant flag
- Runs at B ∈ {100, 1000, 10000}
- Prints a comparison table (matches OPTIMIZATION_JOURNEY.md numbers)

`bench/results.csv`: stores reference A40 numbers so a reader can compare without
running.

### Phase 6 — Docs that sell

#### `README.md` (recruiter-readable)

- One-liner: "10,000-image LeNet-5 forward pass in ~28 ms on a single A40 via
  fused convolution and Tensor Cores (TF32)."
- Result table (4 variants × 3 batch sizes)
- A diagram (input layout → fused unroll → WMMA matmul → permuted output)
- "How to run" (3 commands)
- Link to OPTIMIZATION_JOURNEY.md for the deep dive

#### `docs/OPTIMIZATION_JOURNEY.md`

Rewritten from `archived/.../docs/M3_RESULTS.md`, but:
- Removed school-specific framing (no "Phase A op_0", instead "Phase A: easy wins")
- Added context: why each step was tried, what was expected, what happened
- Kept the trial-and-error tone (negative results matter)

#### `docs/ARCHITECTURE.md`

- LeNet-5 variant dims, why the convolution is the bottleneck
- Conv1 (1→4, K=7) vs Conv2 (4→16, K=7) characteristics
- Why fused im2col matters (vs explicit unroll)

#### `docs/ROOFLINE.md`

- A40 specs (memory bandwidth, peak TF32 TFLOPS)
- Arithmetic intensity calculation for our workload
- Why this is memory-bound (and what that implies)

### Phase 7 — Optional polish

- GitHub Actions: build CUDA on a self-hosted runner, run correctness tests
- Profile screenshots from Nsight Systems / Compute (regenerated on a clean
  setup, not from school-graded runs)
- Blog post / Medium write-up linking back to the repo

## Things to NOT include (verbatim)

- `archived/` is local-only; **do not push it after the initial commit**.
  Either move it out of the repo entirely once refactor is done, or add it to
  `.gitignore` after a checkpoint commit.
- Course-specific READMEs verbatim (`README_m1.md`, `README_m2.md`, `README_m3.md`,
  `README_CNN.md`) — but the *ideas* (problem framing, network architecture) can
  be paraphrased.
- `weights-86.bin` (someone else's training output).
- Quiz answer files (`m2_quiz.md`) — but the methodology insights (gprof shows
  conv dominates CPU time, nsys shows transfer overhead, etc.) can be retold in
  the docs.
- The mini-dnn-cpp framework files (`project/third_party/`, `project/src/layer/*`
  except our own kernel files) — verbatim. Rewriting the host driver from
  scratch is fine and expected.
- Slurm scripts (Delta-specific; not portable). Replace with a shell script that
  works on any sm_80+ box.
- Any text/file that mentions "ECE408", "milestone", "PrairieLearn", "Gradescope",
  course staff names, or grading rubrics.

## Things that need rewriting (not just copying)

| Original | Why rewrite |
|----------|-------------|
| `m3-forward.cu` | Strip GPUInterface, expose plain function. Add `__device__` annotations for clarity. |
| `M3_RESULTS.md` | Remove "Phase A op_0" naming; reframe as engineering decisions. |
| `m1_cpu.cc` (CPU baseline) | Probably need a from-scratch tiny CPU loop; the school version pulls in mini-dnn-cpp. |
| Selfcheck shell scripts | Replace with portable `run_all.sh`. |

## Things that can be copied verbatim (with attribution)

- The kernel bodies of fused / TC / register-tiled variants — those are mine.
- The benchmark numbers (`docs/M3_RESULTS.md` tables).
- Roofline analysis math.

## Suggested execution order

1. Phase 1 — skeleton + `.gitignore` (~30 min)
2. Phase 2 — copy and clean kernels (~1 hr)
3. Phase 3 — driver + utils (~1-2 hr)
4. Phase 4 — correctness tests (~1 hr)
5. Phase 5 — bench harness (~30 min)
6. Phase 6 — docs (~2-3 hr; biggest impact)
7. Phase 7 — polish (optional, anytime)

Total: ~1 weekend of focused work.

## Open decisions

- **Public dataset choice** — Fashion MNIST (matches original) vs CIFAR-10
  (more recognizable) vs synthetic random data (zero dataset overhead). My
  preference: synthetic random data + a small README note about what it would
  look like with real Fashion MNIST. Keeps repo lightweight.
- **Build system** — plain `Makefile` vs CMake. Recommend plain Makefile for
  approachability; CMake if you want to demonstrate that skill too.
- **Multiple variants vs one** — should the public repo show all 4 variants, or
  only the final? My take: include all 4 for the optimization story, with
  `bench/run_all.sh` showing the progression. That's the whole narrative.

## Out-of-scope items

- **lab8** is **not yet completed** in the main school repo — intentionally excluded
  from this portfolio plan. When lab8 is finished there, copy the relevant code
  into `archived/lab8/` first, then decide whether to include any of it in the
  portfolio (likely not — sparse-matrix work is interesting on its own but
  unrelated to the LeNet story).

## Decision log (TO BE FILLED AS WE GO)

| Date | Decision | Reason |
|------|----------|--------|
| 2026-04-16 | `archived/` is read-only source of truth | Avoid accidental edits to school version |
| 2026-04-16 | Exclude lab8 (not yet done) | Will pull from main repo later if it adds portfolio value |
| 2026-04-16 | Rule 3 relaxed: paraphrasing OK | Verbatim coursework files still excluded; ideas/techniques are reusable in practitioner framing |
| | | |
