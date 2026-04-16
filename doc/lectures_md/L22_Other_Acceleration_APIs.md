# L22 — Other Acceleration APIs (Alternatives to CUDA)

## Objectives

Learn non-CUDA ways to program accelerators:

- **OpenCL** — open-standard acceleration API.
- **HIP** — Heterogeneous-Computing Interface for Portability.
- **OpenACC** — pragma-based "low-code" acceleration.
- **MPI** — multi-node parallelism (not GPU-specific, but the glue for multi-node multi-GPU systems).

## The Landscape

GPU vendors: Nvidia, AMD, Intel, Samsung, Apple, Qualcomm, ARM, …

CUDA is only one model among many:

```
OpenGL (1992)  DirectX (1995)  GPGPU (2002)
CUDA (2007)    OpenCL (2008)   OpenACC (2012)
C++AMP (2013)  RenderScript (2013)
Metal (2014)   SYCL (2014)     Vulkan (2016)
ROCm HIP (2016)                OpenMP 4.x+ (2016)
```

Existing frameworks (MPI, TBB, OpenCV) gained GPU support; newer frameworks (TensorFlow, PyTorch, PyCUDA, Caffe) natively accelerate.

## Common Traits of All Acceleration APIs

**Hardware**: hierarchy of lightweight cores, local scratchpad memories, lack of HW coherence, slow global atomics, heavy threading.

**Software**: kernel-oriented; separate device and host memory; software-managed memory; grids / blocks / threads; bulk-synchronous parallelism.

---

## OpenCL

- Framework for CPUs, GPUs, DSPs, FPGAs — *not* Nvidia-specific.
- Launched 2008 (Apple + AMD/IBM/Qualcomm/Intel/Nvidia). OpenCL 2.2 in 2017. Apple deprecated OpenCL in 2018.

### Terminology Mapping (OpenCL ↔ CUDA)

| OpenCL | CUDA |
|---|---|
| WorkGroup | Block |
| WorkItem | Thread |
| `__local` | `__shared__` |
| `barrier(CLK_LOCAL_MEM_FENCE)` | `__syncthreads()` |
| `get_local_id`, `get_group_id` | `threadIdx`, `blockIdx` |

### Tiled GEMM in OpenCL (sketch)

```c
__kernel void myGEMM2(int M, int N, int K,
                      __global float* A, __global float* B, __global float* C) {
    const int row = get_local_id(0);
    const int col = get_local_id(1);
    const int globalRow = TS*get_group_id(0) + row;
    const int globalCol = TS*get_group_id(1) + col;

    __local float Asub[TS][TS];
    __local float Bsub[TS][TS];

    float acc = 0.0f;
    const int numTiles = K/TS;
    for (int t = 0; t < numTiles; t++) {
        Asub[col][row] = A[(TS*t + col)*M + globalRow];
        Bsub[col][row] = B[globalCol*K + (TS*t + row)];
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int k = 0; k < TS; k++)
            acc += Asub[k][row] * Bsub[col][k];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    C[globalCol*M + globalRow] = acc;
}
```

Very close in shape to a CUDA tiled matmul — just different keywords.

---

## HIP (Heterogeneous-Computing Interface for Portability)

- C++ dialect intended to ease porting CUDA apps to portable C++.
- Runs on **AMD** (HCC compiler) or **Nvidia** (NVCC compiler).
- Provides both a C-style API and a C++ kernel language (templates, classes across host/kernel).
- The **HIPify** tool performs source-to-source translation from CUDA → HIP.

### HIP vector add

```c
__global__ void vecAdd(double* a, double* b, double* c, int n) {
    int id = blockIdx.x*blockDim.x + threadIdx.x;
    if (id < n) c[id] = a[id] + b[id];
}

hipMalloc(&d_a, nbytes);
hipMalloc(&d_b, nbytes);
hipMalloc(&d_c, nbytes);
hipMemcpy(d_a, h_a, nbytes, hipMemcpyHostToDevice);
hipMemcpy(d_b, h_b, nbytes, hipMemcpyHostToDevice);

hipLaunchKernelGGL(vecAdd, dim3(gridSize), dim3(blockSize), 0, 0,
                   d_a, d_b, d_c, n);
hipDeviceSynchronize();
hipMemcpy(h_c, d_c, nbytes, hipMemcpyDeviceToHost);
```

Mostly a rename of `cuda*` → `hip*`, with `hipLaunchKernelGGL` replacing `<<<...>>>`.

---

## OpenACC — "Low-Code" Acceleration

A set of:
- compiler directives (`#pragma acc ...`),
- library routines,
- environment variables,

for FORTRAN / C / C++ programs targeting GPUs and CPUs.

### Basic Pragma

```c
#pragma acc <directive> <clauses>
```

Fortran uses `!$acc ... / !$acc end ...`.

### Matmul Example

```c
void computeAcc(float* P, const float* M, const float* N, int Mh, int Mw, int Nw) {
    #pragma acc parallel loop \
        copyin(M[0:Mh*Mw]) copyin(N[0:Nw*Mw]) copyout(P[0:Mh*Nw])
    for (int i = 0; i < Mh; i++) {
        #pragma acc loop
        for (int j = 0; j < Nw; j++) {
            float sum = 0;
            for (int k = 0; k < Mw; k++)
                sum += M[i*Mw + k] * N[k*Nw + j];
            P[i*Nw + j] = sum;
        }
    }
}
```

- `parallel loop` → run the outer `i` loop on the accelerator.
- `copyin` / `copyout` → specify data transfer direction.
- Inner `#pragma acc loop` → map `j` to a second level of parallelism.

### Attractive Property

Code stays **nearly identical** to sequential C/C++ — a non-OpenACC compiler just ignores the pragmas.

### Pitfalls

- Some programs behave differently / incorrectly if pragmas are ignored.
- Pragmas are **hints**; the compiler may or may not follow them well. Performance depends heavily on compiler quality (more so than CUDA/OpenCL).
- OpenACC has **no user-specified thread synchronization** across threads.

### Execution Model: Gangs and Workers

- A `parallel` region executes on the accelerator.
- `num_gangs(G) num_workers(W)` sets the two-level parallelism.

```c
#pragma acc parallel copyout(a) num_gangs(1024) num_workers(32)
{ a = 23; }      // redundantly executed by all 1024 gang leads (usually unwanted)
```

### Gang Loop vs Redundant Loop

```c
#pragma acc parallel num_gangs(1024)
{ for (int i = 0; i < 2048; i++) { ... } }          // redundant: each gang runs all 2048 iterations

#pragma acc parallel num_gangs(1024)
{
    #pragma acc loop gang
    for (int i = 0; i < 2048; i++) { ... }          // 2048 iterations split across 1024 gangs
}
```

### Worker Loop

```c
#pragma acc parallel num_gangs(1024) num_workers(32)
{
    #pragma acc loop gang
    for (int i = 0; i < 2048; i++) {
        #pragma acc loop worker
        for (int j = 0; j < 512; j++)
            foo(i, j);
    }
}
// 32K workers each execute ~32 foo() calls
```

### Redundant vs Distributed Statements

Within `#pragma acc parallel num_gangs(32) { ... }`:
- Statements **outside** a `loop gang` construct run redundantly on all 32 gangs.
- Iterations of a `loop gang` are distributed across the 32 gangs.

### `kernels` Construct

```c
#pragma acc kernels {
    #pragma acc loop num_gangs(1024)
    for (...) { ... }
    #pragma acc loop num_gangs(512)
    for (...) { ... }
    for (...) { ... }
}
```

`kernels` is **descriptive** — a hint about programmer intent; the compiler decides details.

### Reduction

Loops that accumulate into a scalar need a **reduction** clause (otherwise parallel writes race). Floating-point parallel reduction may differ slightly from sequential due to non-associativity.

---

## Multi-GPU & Multi-Node — MPI

### Abstract CUDA-Based Node

Each node has multiple GPUs attached via PCIe to multiple CPUs sharing host memory.

CUDA multi-GPU in a single process:

```c
cudaSetDevice(0);
kernel<<<...>>>(...);
cudaMemcpyAsync(...);
cudaSetDevice(1);
kernel<<<...>>>(...);
```

- Current GPU can change while async calls are running.
- OK to queue async calls on one GPU then switch to another.

### MPI Model

- Many processes distributed across a cluster.
- Each process computes part of the output.
- Processes communicate via **message passing** (no shared memory) and synchronize via messages/barriers.

### MPI Basics

```c
mpirun -np X ./prog                // launch X processes
int MPI_Init(int* argc, char*** argv);
int MPI_Comm_rank(MPI_Comm comm, int* rank);
int MPI_Comm_size(MPI_Comm comm, int* size);
MPI_COMM_WORLD                      // default communicator with all procs
MPI_Finalize();
```

### Point-to-Point Messages

```c
int MPI_Send(void* buf, int count, MPI_Datatype dt,
             int dest,   int tag, MPI_Comm comm);

int MPI_Recv(void* buf, int count, MPI_Datatype dt,
             int source, int tag, MPI_Comm comm, MPI_Status* status);
```

### Vector-Add Pattern (Server + Compute Nodes)

```c
int main(int argc, char** argv) {
    int pid, np;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);
    MPI_Comm_size(MPI_COMM_WORLD, &np);
    if (np < 3) { MPI_Abort(MPI_COMM_WORLD, 1); return 1; }

    if (pid < np - 1) compute_node(vector_size / (np - 1));
    else              data_server(vector_size);

    MPI_Finalize();
    return 0;
}
```

- **Server** (`pid == np-1`): allocates and initializes data, `MPI_Send`s chunks to each compute node, `MPI_Barrier`s, then `MPI_Recv`s partial results back.
- **Compute node**: `MPI_Recv`s its chunk, does the work (optionally offloading to a GPU via `cudaMalloc` / `cudaMemcpy` / kernel launch), barriers, and `MPI_Send`s the result back.

This pattern generalizes to multi-node multi-GPU: within each rank, use CUDA to drive local GPUs; between ranks, use MPI to exchange boundary data.

---

## Supercomputing Context

- Top-5 supercomputers in 2025 are heavily GPU-based (e.g. **ORNL Frontier**: 9,472 AMD CPUs + 37,888 AMD GPUs).
- **Blue Waters** (UIUC, 2013–2021) and **Delta** (UIUC) provide the systems context — multiple GPU types (A100, A40, MI100), high-bandwidth Slingshot / IB interconnects, multi-PB storage, Lustre/DDN filesystems.
- App-level performance depends on the whole stack: kernels (CUDA/HIP/OpenACC), intra-node multi-GPU orchestration, and inter-node MPI.

## Key Takeaways

- All acceleration APIs share the same underlying model (device/host split, kernels, hierarchical threading, software-managed memory).
- **OpenCL**: portable across vendors; verbose; syntactic cousin of CUDA.
- **HIP**: "CUDA-looking" but runs on AMD and Nvidia; HIPify automates porting.
- **OpenACC**: pragma-based, attractive for minimal code change, but compiler-dependent and weaker on fine-grained control/sync.
- **MPI**: the standard for multi-node; pair it with CUDA/HIP/OpenACC for large multi-GPU systems.
- Choose based on portability needs, tolerance for low-level control, and target hardware mix.
