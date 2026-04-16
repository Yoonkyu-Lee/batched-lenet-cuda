# Architecture

## The network

We target a two-layer convolutional feature extractor that matches the
convolutional stages of a LeNet-5 variant. For the purposes of benchmarking,
the surrounding pooling / fully-connected / softmax layers are **not**
implemented — they run fast enough on a GPU that they never appear in the
optimization budget. Every number in this repo refers to the forward pass of
the two convolutions alone.

### Layer dimensions

| Layer  | Input (C×H×W) | Mask shape (M×C×K×K) | Output (M×H_out×W_out) |
|--------|---------------|----------------------|------------------------|
| Conv1  | 1 × 86 × 86   | 4 × 1 × 7 × 7   (196 floats)   | 4 × 80 × 80 |
| Conv2  | 4 × 40 × 40   | 16 × 4 × 7 × 7  (3,136 floats) | 16 × 34 × 34 |

No padding, stride 1. Between the two conv layers, a 2×2 max-pool halves the
spatial dimensions. In this repo we generate the `4 × 40 × 40` input for Conv2
as synthetic data; the pool stage is out of scope.

## Why the convolution dominates

gprof on a CPU baseline (B=1000, full forward pass including pooling and FCs)
attributes **88.8% of total runtime** to `conv_forward`. Every other layer is
cheap by comparison. That's why the story in this repo is almost entirely
about the two `Conv` functions.

On GPU, the picture is even more skewed — the FC and pool layers sit in the
low-microsecond regime per batch, while the conv layers are the 10s of
milliseconds that this project is optimizing.

## Convolution as matmul: "im2col"

Each output element is a dot product of a `C × K × K` patch of the input with
a single output-map filter. If you materialize all such patches as columns of
a single matrix:

```
unrolled_input[(C · K · K)  ×  (B · H_out · W_out)]
```

then the full conv forward pass for the whole batch becomes a single matrix
multiply:

```
output[M × (B · H_out · W_out)]
  = mask[M × (C · K · K)]
  × unrolled_input[(C · K · K) × (B · H_out · W_out)]
```

followed by a permute that reshapes the output from `M × B × H_out × W_out`
into the final `B × M × H_out × W_out` layout.

This is the classical **im2col** reformulation. It makes the problem look
like a GEMM — which is important because GPUs are very good at GEMMs (WMMA
tensor cores, cuBLAS, etc.) while native direct convolutions are awkward.

## Why we *fuse* im2col

A naïve im2col approach materializes `unrolled_input` into global memory
before the matmul. For this workload that buffer is:

```
Conv1 at B=10000:   49 rows × 64,000,000 cols × 4 bytes = ~12.5 GB
Conv2 at B=10000:  196 rows × 11,560,000 cols × 4 bytes = ~9.1 GB
```

That's a lot of memory and bandwidth spent just to hold intermediate data
that's read-once. The *fused* approach never writes the unrolled buffer;
each shared-memory tile of the matmul computes its im2col input element on
the fly by decoding `(channel, p, q, batch, h_out, w_out)` from the linear
tile index.

So "fused im2col + tiled matmul" means: a matmul kernel where the reads of
`B` (the unrolled input matrix) go to raw input memory with a bit of index
math, rather than to a materialized intermediate.

## Why the two conv layers behave differently

| Dimension | Conv1 | Conv2 |
|---|---:|---:|
| Map_out (M) | 4 | 16 |
| Unrolled rows (C·K·K) | 49 | 196 |
| Width_unrolled (B · H_out · W_out) at B=10000 | 64,000,000 | 11,560,000 |

This matters because:

- **Conv1 has a tiny M**. Any 16-row tensor-core tile has 75% of its rows
  wasted (padded to 0). So Conv1 doesn't benefit much from pure MMA speedup.
- **Conv2 has a large K dimension**. 196 inner-product terms per output means
  every output element does ~196 FMAs. Arithmetic intensity here is higher,
  and memory access patterns matter more.
- **The width_unrolled is huge** in both cases — >10 M columns. That's what
  makes register tiling (reusing the A-fragment across many N-tiles) so
  effective: the tall-skinny matmul rewards any reuse you can extract.

The optimization story (see [`OPTIMIZATION_JOURNEY.md`](OPTIMIZATION_JOURNEY.md))
is largely about why each technique helps one layer more than the other, and
how to get both down together.
