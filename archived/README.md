# ECE 408 / CS 483 / CSE 408 (sp26) repo for NetID: yoonkyu2

GitHub username at initialization time: yoonkyu2

For next steps, please refer to the instructions provided by your course.

To retrieve the latest assignment:

- `git fetch release`
- `git merge release/main -m "some comment" --allow-unrelated-histories`
- `git push origin main`

## Learned Coding Skills

### Lab 0: Development Environment Setup

- **Task:** Learn how to compile and run CUDA code on Delta.
- **What I implemented:** Basic environment setup and GPU information query.
- **What I learned:** Slurm workflow, Delta job scripts, and core GPU properties.
- **Links:** [lab0/README.md](lab0/README.md)

### Lab 1: Vector Addition

- **Task:** Implement vector addition on the GPU.
- **What I implemented:** Full CUDA host flow plus a one-thread-per-element kernel.
- **What I learned:** Device memory management, indexing, bounds checks, and basic data parallelism.
- **Links:** [doc/lab1_notes.md](doc/lab1_notes.md), [lab1/README.md](lab1/README.md)

### Lab 2: Basic Matrix Multiplication

- **Task:** Implement naive dense matrix multiplication.
- **What I implemented:** 2D thread mapping and a per-output-element matrix multiply kernel.
- **What I learned:** Row-major indexing, reduction dimension handling, and why correctness does not imply performance.
- **Links:** [doc/lab2_notes.md](doc/lab2_notes.md), [lab2/README.md](lab2/README.md)

### Lab 3: Tiled Matrix Multiplication

- **Task:** Improve matrix multiplication with shared-memory tiling.
- **What I implemented:** Tile loading, synchronization, and boundary-safe shared-memory reuse.
- **What I learned:** Tiling, shared memory lifecycle, and data locality.
- **Links:** [doc/lab3_notes.md](doc/lab3_notes.md), [lab3/README.md](lab3/README.md)

### Lab 4: 3D Convolution

- **Task:** Implement 3D convolution with constant memory and 3D tiling.
- **What I implemented:** A halo-aware 3D shared-memory kernel and constant-memory mask loading.
- **What I learned:** 3D tiling, stencil access patterns, and volume boundary handling.
- **Links:** [doc/lab4_notes.md](doc/lab4_notes.md), [lab4/README.md](lab4/README.md)

### Lab 5: Histogram Equalization

- **Task:** Implement histogram equalization for RGB images.
- **What I implemented:** A pipeline from RGB conversion through histogramming, CDF computation, and equalization.
- **What I learned:** Atomic contention, privatization, and the relationship between histogram, CDF, and correction.
- **Links:** [doc/lab5_notes.md](doc/lab5_notes.md), [lab5/README.md](lab5/README.md)

### Lab 6: List Reduction

- **Task:** Implement the improved shared-memory reduction kernel from lecture.
- **What I implemented:** A per-block reduction over `2 * BLOCK_SIZE` inputs with host-side final accumulation.
- **What I learned:** Segmented reduction, shared-memory trees, improved reduction ordering, and synchronization costs.
- **Links:** [doc/lab6_notes.md](doc/lab6_notes.md), [lab6/README.md](lab6/README.md)

### Lab 7: Parallel Scan

- **Task:** Implement a hierarchical parallel scan for a 1D list.
- **What I implemented:** A block-level Brent-Kung style scan, an auxiliary block-sum scan, and a final add-back kernel.
- **What I learned:** Inclusive vs exclusive scan, hierarchical scan structure, and the tradeoff between Kogge-Stone latency and Brent-Kung work efficiency.
- **Links:** [doc/lab7_notes.md](doc/lab7_notes.md), [lab7/README.md](lab7/README.md)

### Lab 8: Sparse Matrix-Vector Multiplication (JDS)

- **Task:** Implement SpMV for a sparse matrix using the Jagged Diagonal Storage (JDS) transposed format.
- **What I implemented:** A one-thread-per-sorted-row kernel that walks `matColStart` sections and writes results back through `matRowPerm`.
- **What I learned:** Sparse storage tradeoffs (COO/CSR/ELL/JDS), how row sorting plus column-major transposition enables coalesced loads and reduces warp divergence, and why format choice can dominate kernel cleverness.
- **Links:** [doc/lab8_notes.md](doc/lab8_notes.md), [lab8/README.md](lab8/README.md)

## Study Notes

- **Doc index:** [doc/README.md](doc/README.md)
- **Glossary:** [doc/glossary.md](doc/glossary.md)
- **Quick review:** [doc/quick_review.md](doc/quick_review.md)
- **Lecture materials:** [doc/lectures_md](doc/lectures_md), [doc/lectures_pdf](doc/lectures_pdf)
- **Lab notes:** [lab1_notes.md](doc/lab1_notes.md), [lab2_notes.md](doc/lab2_notes.md), [lab3_notes.md](doc/lab3_notes.md), [lab4_notes.md](doc/lab4_notes.md), [lab5_notes.md](doc/lab5_notes.md), [lab6_notes.md](doc/lab6_notes.md), [lab7_notes.md](doc/lab7_notes.md)
- **Profiling notes:** [profiling_lecture_notes.md](doc/profiling_lecture_notes.md), [Profiling-Lecture guide](Profiling-Lecture/lecture_notes.md)
- **New lecture summaries:** [L18 Parallel Scan](doc/lectures_md/L18_Parallel_Computation_Patterns_Parallel_Scan.md), [L19 GPU Systems Architecture](doc/lectures_md/L19_GPU_Systems_Architecture.md)
