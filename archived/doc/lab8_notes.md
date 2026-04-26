# Lab 8 — Sparse Matrix-Vector Multiplication (JDS)

## What the Lab Does

Implements SpMV (Sparse Matrix-Vector Multiplication) using the **Jagged Diagonal Storage (JDS)** transposed format. The host code converts CSR input to JDS, and the kernel computes `y = A·x` where A is sparse.

## Relevant Lectures

- **L21 (Parallel Sparse Methods)**: all sparse formats (COO, CSR, CSC, ELL, JDS), parallelization strategies, and tradeoffs.
- **L14 (Atomic Operations and Histogramming)**: background on parallel output conflicts (COO needs atomics; JDS does not).

## JDS Format Recap

1. Group nonzeros by row (like CSR).
2. Sort rows by length descending → `matRowPerm` stores original row indices.
3. Transpose to column-major → `matColStart[sec]` marks where each "section" (column of the padded table) begins in `matData`/`matCols`.
4. `matRows[row]` = number of nonzeros in sorted row `row` (i.e., how many sections this thread iterates).

### Variable Mapping (Lecture → Lab)

| Lecture name | Lab variable |
|---|---|
| `data` | `matData` |
| `col_index` | `matCols` |
| `jds_t_col_ptr` | `matColStart` |
| `jds_row_perm` | `matRowPerm` |
| section lengths | `matRows` |

## Kernel Design

- **One thread per sorted row**: thread `row` iterates over `matRows[row]` sections.
- **Index**: `idx = matColStart[sec] + row` gives position in `matData` and `matCols`.
- **Dot product**: `dot += matData[idx] * vec[matCols[idx]]`.
- **Permute output**: `out[matRowPerm[row]] = dot` writes to the original (unsorted) row position.

```c
int row = blockIdx.x * blockDim.x + threadIdx.x;
if (row < dim) {
    float dot = 0.0f;
    int numSections = matRows[row];
    for (int sec = 0; sec < numSections; sec++) {
        int idx = matColStart[sec] + row;
        dot += matData[idx] * vec[matCols[idx]];
    }
    out[matRowPerm[row]] = dot;
}
```

## Why JDS Works Well on GPUs

- **Coalesced memory access**: column-major transposition means adjacent threads access adjacent memory locations in `matData` and `matCols`.
- **Reduced control divergence**: rows sorted by length → threads in the same warp have similar iteration counts. Shorter rows drop out gracefully from the end.
- **No padding waste**: unlike ELL, JDS doesn't pad shorter rows — it just has fewer active threads in later sections.
- **No atomics needed**: each thread owns exactly one output row (via `matRowPerm`), so no write conflicts.

## Key Lessons

1. **Format choice matters more than kernel cleverness.** The same SpMV problem can be 10× faster just by picking JDS over CSR — the kernel code is nearly identical, but memory access patterns are fundamentally different.
2. **Sorting enables coalescing.** By sorting rows by length, adjacent threads do similar work (less divergence) and access adjacent memory (coalesced loads).
3. **Indirection is the cost of regularity.** `matRowPerm` adds one level of indirection on output, but the trade is worth it for coalesced input reads.
4. **The host-side conversion (CSR→JDS) is not free** — it's a preprocessing cost amortized over many SpMV calls (e.g., iterative solvers).
