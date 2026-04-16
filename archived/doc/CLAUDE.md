# CLAUDE.md — Project CNN Harness Guidelines

## Project structure

```
Project_CNN/
├─ project/src/layer/custom/   ← GRADED files (never touch files not listed in milestone READMEs)
├─ project/m3/{req_*,op_*}/    ← individual optimization folders, each with its own m3-forward.cu
├─ scripts/{m1,m2,m3,profile}/ ← slurm scripts + selfcheck.sh
├─ docs/                       ← plans, notes, results, report template
├─ outputs/{m1,m2,m3}/         ← run outputs (.out/.err), gitignored
├─ profiles/{nsys,ncu,sanitize}/ ← profiling artifacts, gitignored
├─ selfcheck/{m1,m2,m3}/       ← timestamped selfcheck run dirs, gitignored
├─ run.sh                      ← ./run.sh build | clean
└─ weights-86.bin              ← pretrained model weights (do not modify)
```

## Build & run

- **Build**: `./run.sh build` (identical to release version — do NOT modify; grader depends on it)
- **Clean**: `./run.sh clean` (removes binaries and `*.out`/`*.err` from root)
- **Dir setup** (after fresh clone or clean): `bash scripts/setup_dirs.sh` — creates `outputs/{m1,m2,m3}/`, `profiles/{nsys,ncu,sanitize}/`, `selfcheck/{m1,m2,m3}/` that our slurm scripts write to. Safe to re-run.
- **Gprof build**: `bash scripts/build_pg.sh`
- All `sbatch` and `bash` commands must be run from `Project_CNN/` root (slurm paths are cwd-relative)

**IMPORTANT**: `run.sh` must stay byte-identical to `release/main:Project/Project_CNN/run.sh` (Campuswire explicitly instructed merging the release commit "as it contains important changes to run.sh"). Any custom directory-setup logic belongs in `scripts/setup_dirs.sh`, NOT in `run.sh`.

## M3 optimization workflow

Follow this loop for each Phase/trial:

1. **Implement** the change in `project/src/layer/custom/m3-forward.cu`
2. **Build**: `./run.sh build`
3. **Quick test**: `TRIAL=<name> sbatch scripts/m3/m3_quick.slurm`
   - Output goes to `outputs/m3/m3_quick_<jobid>.out` (timestamped by slurm %j, never overwrites)
4. **Record** Conv1 / Conv2 / Total op times + accuracy in `docs/M3_RESULTS.md` (append-only)
5. **If promising**: run full selfcheck `sbatch scripts/m3/m3_selfcheck.slurm` (results in `selfcheck/m3/<timestamp>/`)
6. **Compare** against prior best in the summary table
7. **Pick winner** for this Phase; note why alternatives were discarded
8. **Proceed** to next Phase

### Output preservation

- `m3_quick.slurm` uses `%j` (slurm job ID) in `--output`/`--error` filenames → each run creates a unique file
- `m3_selfcheck.sh` writes to `selfcheck/m3/<timestamp>/` → each run creates a unique directory
- **Never manually delete** outputs or selfcheck results during active development
- `docs/M3_RESULTS.md` is the single source of truth for performance data

### Individual optimization folders

Each `project/m3/{req_*,op_*}/m3-forward.cu` must be a **standalone** version with ONLY that optimization added on top of the M2 baseline (fused or unfused as specified). These are graded independently for correctness.

The stacked final version lives at `project/src/layer/custom/m3-forward.cu`.

## Graded files (do NOT modify anything else for grading)

| Milestone | Graded files |
|-----------|-------------|
| M1 | `project/src/layer/custom/cpu-new-forward.cc`, `project/src/layer/custom/new-forward.cu` |
| M2 | `project/src/layer/custom/unroll-new-forward.cu`, `project/src/layer/custom/kernel-fusion-forward.cu` |
| M3 | `project/src/layer/custom/m3-forward.cu` (final), `project/m3/*/m3-forward.cu` (individual) |

## Files that must NOT be committed

- `*.nsys-rep`, `*.sqlite`, `*.ncu-rep` (too large for GitHub, M2 README forbids it)
- Compiled binaries (`m1_cpu`, `m1_gpu`, `m2_unroll`, `m2_fused`, `m3`, `viz`)
- `outputs/`, `profiles/`, `selfcheck/`, `gmon.out`
- All of these are already in `.gitignore`

## Expected accuracies

| Batch | Accuracy |
|------:|----------|
| 100 | 0.86 |
| 1000 | 0.886 |
| 10000 | 0.8714 |

Tolerance: ±0.002 for public batches. Edge batches (not in table) must be ≥ 0.80.

## M3 performance targets

- **Full credit**: sum of Op Times ≤ 60 ms @ B=10000
- **Zero credit**: > 100 ms (linear between 60–100)
- **Competition**: each layer < 40 ms, run with `./m3 --competition`
- Final submission must use **single stream** (no req_0 in the performance path)

## Key reference lectures

- L17 (Accelerating Matrix Operations) → Tensor Cores / WMMA
- L16 (Advanced Optimizations) → parameter sweep, thread coarsening
- L08 (Convolution + Constant Cache) → op_0
- L20 (Data Transfer + CUDA Streams) → req_0
- L13 (Profiling) → nsys / ncu methodology
