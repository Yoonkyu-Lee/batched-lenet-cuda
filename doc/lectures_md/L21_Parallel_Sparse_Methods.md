# L21 — Parallel Sparse Methods

## Objectives

- Learn how to compact sparse input data to save memory bandwidth.
- Trade off **space, modifiability, accessibility, coalescing, and load balance** across storage formats.
- Core challenge: sparsity breaks regularity — formats try to recover it.

## Why Sparse Matters

- Many real workloads are sparse: PDE solvers, iterative Conjugate Gradient (built on SpMV), graph analytics (adjacency matrices), recommender systems, NLP embeddings, GNN SpMM.
- Opportunities when most entries are zero:
  - Don't **allocate** space for zeros (memory capacity).
  - Don't **load** zeros (memory bandwidth).
  - Don't **compute** with zeros (time).

## SpMV as Running Example

Sparse matrix-vector product `y = A·x`. Compared to dense matmul, SpMV is:
- Irregular / unstructured, with little data reuse.
- Benefits little from compiler transforms.
- Key to speed: **maximize regularity** (reduce divergence, balance load) and **maximize DRAM burst utilization** (layout).

Format design considers: space efficiency, modifiability (add/reorder entries), accessibility (find data by row/col/nnz), memory access pattern (coalescing), load balance (divergence).

---

## COO — Coordinate Format

Store each nonzero as `(row, col, value)` in three arrays.

```
Row:    [0 0 1 1 1 2 2 3]
Col:    [0 1 0 2 3 1 2 3]
Value:  [1 7 5 3 9 2 8 6]
```

### Parallelization

- **One thread per nonzero**: multiply `value * x[col]`, `atomicAdd` into `y[row]`. Needs atomics.
- Alternative: one thread per row (but COO is not sorted by row, so this is awkward).

```c
__global__ void spmv_coo_kernel(COOMatrix A, float* x, float* y) {
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < A.numNonzeros) {
        unsigned int row = A.rowIdxs[i];
        unsigned int col = A.colIdxs[i];
        atomicAdd(&y[row], A.values[i] * x[col]);
    }
}
```

### Tradeoffs

- ✅ Easy to add/reorder entries; nonzeros can be in any order.
- ✅ Given a nonzero, row/col are trivially available.
- ✅ Coalesced accesses to the input matrix; no control divergence.
- ❌ Given a row or column, finding all nonzeros requires search.
- ❌ Output writes need atomics.

---

## CSR — Compressed Sparse Row

Nonzeros grouped by row, plus a `rowPtr` array indexing where each row starts.

```
RowPtrs: [0 2 5 7 8]
Col:     [0 1 0 2 3 1 2 3]
Value:   [1 7 5 3 9 2 8 6]
```

### Parallelization

One thread per **row**: iterate `rowPtr[row]` → `rowPtr[row+1]`, accumulate into `y[row]`.

```c
__global__ void SpMV_CSR(int num_rows, float* data, int* col_index,
                         int* row_ptr, float* x, float* y) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < num_rows) {
        float dot = 0;
        int row_start = row_ptr[row];
        int row_end   = row_ptr[row+1];
        for (int e = row_start; e < row_end; e++)
            dot += data[e] * x[col_index[e]];
        y[row] = dot;
    }
}
```

### Tradeoffs (vs COO)

- ✅ `rowPtr` smaller than storing a full `row` array.
- ✅ Given a row, easy to find all its nonzeros.
- ✅ Output writes are coalesced; no atomics.
- ❌ Adding new entries is hard (rows are packed).
- ❌ Given a nonzero, finding its row requires search over `rowPtr`.
- ❌ **Control divergence**: threads in a warp iterate different numbers of times (row length varies).
- ❌ **Memory divergence**: thread 0 touches row 0 element 0, thread 1 touches row 1 element 0 — these are far apart in `data`, so loads are uncoalesced.

---

## CSC — Compressed Sparse Column

Same idea as CSR but grouped by column: `colPtr`, `row`, `value`.

- ✅ Given a column, easy to find all nonzeros; input-vector accesses coalesce.
- ❌ Output writes need atomics (multiple columns contribute to the same row).
- ❌ Control divergence, uncoalesced matrix reads.
- **Usually not used for SpMV.** Shines when the *vector* is very sparse — an entire column can be skipped when `x[col] == 0`.

---

## ELL (ELLPACK) — Regularized

1. Group nonzeros by row (like CSR).
2. **Pad** every row to the length of the longest row.
3. Store the padded 2D table in **column-major** order.

```
Padded rows:  value             column
1 7 *         0 1 *
5 3 9         0 2 3
2 8 *         1 2 *
6 * *         3 * *

Column-major storage:
Value:  1 5 2 6 | 7 3 8 * | * 9 * *
Col:    0 0 1 3 | 1 2 2 * | * 3 * *
```

### Parallelization

One thread per row; iterate down the padded columns. Stride = `num_rows`, so adjacent threads in the same iteration access **adjacent memory** → coalesced.

```c
__global__ void SpMV_ELL(int num_rows, float* data, int* col_index,
                         int num_elem, float* x, float* y) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < num_rows) {
        float dot = 0;
        for (int i = 0; i < num_elem; i++)
            dot += data[row + i*num_rows] *
                   x[col_index[row + i*num_rows]];
        y[row] = dot;
    }
}
```

### Tradeoffs

- ✅ Can add entries as long as row isn't full.
- ✅ Given a row or a nonzero, easy to locate row/col.
- ✅ Coalesced matrix accesses.
- ❌ Padding overhead — bad when one row is much longer than the rest.
- ❌ Given a column, hard to find all nonzeros.

---

## Hybrid ELL + COO

- ELL stores typical entries; COO handles the few exceptionally long rows.
- COO portion combined via segmented reduction (often done in sequential host code in practice).
- Benefits: less padding, easier to add entries anywhere, less control divergence than pure ELL.

---

## JDS — Jagged Diagonal Storage

Motivation: in CSR, a block's runtime is bounded by its longest row, causing divergence.

1. Group nonzeros by row (CSR-like).
2. **Sort rows by length** (descending), remember original row index in `jds_row_perm`.
3. Optionally **transpose**: store in column-major order, plus `iterPtr` (a.k.a. `matColStart`) marking where each iteration's column starts.

```
jds_row_perm: [2 0 3 1]    (original row for each sorted slot)
jds_row_ptr:  [0 3 5 7 7]
data:         [2 4 1 3 1 1 1]
col_index:    [1 2 3 0 2 0 3]
```

After JDS-T transposition:

```
Column:  0 0 0 2 | 1 0 1 | 3 2 | 4 4 | 3 5 | 5 4
Value:   b h m f | k a c | i n | g l | d j | o e
IterPtr: [0 6 11 14 15]
```

### Parallelization (JDS-T)

One thread per (sorted) row; each iteration, thread reads the element at `iterPtr[sec]+row`. Threads **drop out from the end** as the sorted rows shorten — coalesced accesses with graceful tapering.

```c
__global__ void SpMV_JDS_T(int num_rows, float* data, int* col_index,
                           int* jds_t_col_ptr, int* jds_row_perm,
                           float* x, float* y) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < num_rows) {
        float dot = 0;
        unsigned int sec = 0;
        while (jds_t_col_ptr[sec+1] - jds_t_col_ptr[sec] > row) {
            dot += data[jds_t_col_ptr[sec] + row] *
                   x[col_index[jds_t_col_ptr[sec] + row]];
            sec++;
        }
        y[jds_row_perm[row]] = dot;   // permute back to original row
    }
}
```

### Lab 8 variable names

| Slide name | Lab 8 name |
|---|---|
| data | `matData` |
| col_index | `matCols` |
| jds_t_col_ptr | `matColStart` |
| jds_row_perm | `matRowPerm` |
| jds section lengths | `matRows` |

### Tradeoffs

- ✅ No padding (unlike ELL).
- ✅ Given a row, easy to find nonzeros.
- ✅ Coalesced accesses; minimized control divergence (neighboring sorted rows have similar lengths; end-of-loop drops cleanly).
- ❌ Hard to add elements.
- ❌ Given a nonzero, row is indirect (must go through `jds_row_perm`); given a column, hard to find all nonzeros.

---

## Choosing a Format by Sparsity Pattern

| Pattern | Best format | Why |
|---|---|---|
| Roughly random | **ELL** | Padding uniformly distributed, uniform representation |
| High variance in row length | **ELL + COO** | ELL for the common case, COO for outlier rows |
| Very sparse | **COO** | Little data, compute is sparse; no point padding |
| Roughly triangular | **JDS** | Sorting exploits the structure |
| Banded | **ELL** | Rows are short and similar-length |

Other formats mentioned: **DIA** (diagonal), **PKT** (packet/reorder into diagonal submatrices), **DOK** (dictionary of keys), **Blocked CSR**, and hybrids.

## Worked Problem — CSR vs COO Identification

```
Data    = [1,2,1,1,2,3,4,1,1]
Col_idx = [0,2,3,0,1,2,3,0,3]
Row_ptr = [0,1,3,7,9]          ← CSR only
Row_idx = [0,1,1,3,3,3,3,7,7]  ← not valid COO (skips row indices)
```

Only the CSR reading is correct.

## Sparse Matrices as Foundation

- Graphs as sparse adjacency matrices → GNNs use SpMM.
- Binning with sparse matrices (ray tracing, SPH, games).
- Deeper treatment in ECE508/CS508.

## Key Takeaways

- All sparse formats trade off along the same axes — no universal winner.
- **COO**: trivial construction, atomics on output.
- **CSR**: compact, good outputs, but divergence + uncoalesced matrix reads.
- **ELL**: regularized by padding → coalesced, but padding can explode.
- **Hybrid ELL+COO**: best of both for skewed row-length distributions.
- **JDS (especially JDS-T)**: sort + transpose gives coalesced reads and low divergence without padding — the format used in Lab 8.
