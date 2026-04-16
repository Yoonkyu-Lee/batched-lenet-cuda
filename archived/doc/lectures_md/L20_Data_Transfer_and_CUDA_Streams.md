# L20 — Data Transfer and CUDA Streams (Task Parallelism)

## Objectives

- Learn advanced CUDA APIs for data transfer and kernel launch.
- Understand task parallelism: overlapping host↔device transfers with kernel computation.
- Use **CUDA streams** to express this overlap.

## Why Streams: Serialized Transfer Is Wasteful

With a plain `cudaMemcpy` flow:

```
Trans A → Trans B → VecAdd → Trans C
```

- PCIe runs in only one direction at a time; GPU sits idle during transfers.
- Transfers and compute serialize, wasting both engines.

**Pipelined (overlapped) timing**: split the input into segments and overlap segment *N*'s transfer with segment *N−1*'s compute and segment *N−2*'s result transfer.

## Device Overlap Capability

Most CUDA devices support *device overlap* — they can execute a kernel while copying between host and device.

Check via:
```c
cudaDeviceProp prop;
cudaGetDeviceProperties(&prop, i);
if (prop.deviceOverlap) { ... }
```

## CUDA Streams

- A **stream** is a FIFO queue of operations (kernel launches + memcpys).
- The driver processes a stream strictly in order: a kernel in a stream waits for prior memcpys in the same stream.
- Operations in **different** streams can run concurrently — this is task parallelism.

Conceptual hardware picture: a GPU has separate engines that can run in parallel:
- **Copy Engine** (with PCIe Up / PCIe Down)
- **Kernel Engine**

Two streams feeding these engines let transfer and compute overlap.

## Basic Multi-Stream Host Code

```c
cudaStream_t stream0, stream1;
cudaStreamCreate(&stream0);
cudaStreamCreate(&stream1);
// cudaMalloc d_A0,d_B0,d_C0 and d_A1,d_B1,d_C1

for (int i = 0; i < n; i += SegSize*2) {
    // all of stream0's work, then all of stream1's work
    cudaMemcpyAsync(d_A0, h_A+i, ..., stream0);
    cudaMemcpyAsync(d_B0, h_B+i, ..., stream0);
    vecAdd<<<..., 0, stream0>>>(d_A0, d_B0, ...);
    cudaMemcpyAsync(h_C+i, d_C0, ..., stream0);

    cudaMemcpyAsync(d_A1, h_A+i+SegSize, ..., stream1);
    ...
}
```

**Problem on older (software-multiplexed) GPUs**: all memcpys from all streams go into a single copy-engine queue. `MemCpy C.0` blocks `MemCpy A.1` / `B.1` from making progress — not the overlap we wanted.

## Better Ordering

Issue **all input transfers first**, then kernels, then output transfers, across both streams:

```c
for (int i = 0; i < n; i += SegSize*2) {
    cudaMemcpyAsync(d_A0, ..., stream0);
    cudaMemcpyAsync(d_B0, ..., stream0);
    cudaMemcpyAsync(d_A1, ..., stream1);
    cudaMemcpyAsync(d_B1, ..., stream1);

    vecAdd<<<..., 0, stream0>>>(...);
    vecAdd<<<..., 0, stream1>>>(...);

    cudaMemcpyAsync(h_C+i,         d_C0, ..., stream0);
    cudaMemcpyAsync(h_C+i+SegSize, d_C1, ..., stream1);
}
```

Now `C.0` no longer blocks `A.1`/`B.1`. But `C.1` still blocks `A.2`/`B.2` of the next iteration because PCIe is used in only one direction at a time.

→ **Three streams** are needed for truly continuous pipelining.

## Hyper-Q (Kepler+)

- Hardware feature (Compute Capability 3.5+) giving **multiple real hardware work queues per engine**.
- Multiple CPU threads/processes can submit work to the GPU simultaneously.
- Streams execute truly in parallel at the hardware level instead of multiplexing into one queue.

| Era | Concurrency | Behavior |
|---|---|---|
| Fermi | 16-way | All streams multiplex into a single HW work queue; overlap only at stream edges. |
| Kepler | 32-way | One HW work queue per stream; full-stream concurrency with no false inter-stream dependencies. |

## Other Stream Considerations

- `cudaMemcpyAsync` **silently falls back to synchronous** if host memory is *not pinned*. Use `cudaMallocHost` / `cudaFreeHost`.
- Synchronization primitives:
  - `cudaStreamSynchronize(stream)` — wait for one stream
  - `cudaDeviceSynchronize()` — wait for all streams
  - `cudaStreamQuery(stream)` — non-blocking poll
  - `cudaEventRecord` / `cudaEventSynchronize` — event-based, also enables cross-stream dependencies
- The **default (NULL) stream** has implicit sync behavior that can silently break overlap — avoid mixing it with explicit streams.
- Destroy streams with `cudaStreamDestroy()`.

## How Small Should Segments Be?

Intuition says smaller segments reduce boundary (head/tail) effects, but real execution time is *not* linear in problem size — it has a floor:

- Small problems leave SMs idle, too few warps, partial warps, poor load balance.
- Kernel launch overhead is fixed.
- Transfers have startup cost (DMA, host side).

**Conclusion**: use **moderate** segment sizes — small enough to pipeline effectively, large enough to amortize launch/DMA overhead. Optimal size is GPU-dependent.

## Worked Pipelining Problem

Single-stream times:
- Input transfer 10 s, kernel launch overhead 1 s, compute 10 s, output transfer 10 s → **31 s** total.

With 3 hardware streams enabling continuous pipelining, splitting into more segments reduces total time (limited by the longest stage per segment + fixed per-segment launch overhead). Example from slides:

| Segments | Streams | Total |
|---|---|---|
| 1 | 1 | 31 s |
| 2 | 2 | 22 s |

The optimal segment count balances pipeline fill/drain against per-segment launch overhead.

## Key Takeaways

- Streams turn serial `copy → compute → copy` into a pipeline using two hardware engines.
- Order matters: issue transfers first, then kernels, then results, across streams.
- Use **pinned memory** with `cudaMemcpyAsync` or you silently lose overlap.
- **Three streams** are needed to hide both input and output transfers simultaneously.
- Hyper-Q makes streams cheap and truly concurrent on modern GPUs.
- Segment sizing is a tradeoff: boundary effects vs. per-launch/DMA overhead.
