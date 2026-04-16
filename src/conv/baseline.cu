// Baseline: one thread per output element. Each thread iterates over
// (Channel × K × K) and accumulates the convolution. No data reuse beyond what
// the L1/L2 cache happens to provide.
//
// Grid mapping: blockDim.x folds Map_out into grid.x (so Batch can use grid.z
// without overflowing the 65,535 limit at large batch sizes).
//
// This is the entry point for the optimization story; everything that follows
// is justified against this kernel's behavior.

#include "conv.h"
#include "../utils/cuda_check.h"
#include <cuda_runtime.h>

namespace lenet_conv {

namespace {

// Per-layer state owned by this translation unit.
float* d_mask_ = nullptr;
size_t d_mask_bytes_ = 0;

__global__ void conv_baseline_kernel(
    const float* __restrict__ input,
    const float* __restrict__ mask,
    float* __restrict__ output,
    int Batch, int Map_out, int Channel,
    int Height, int Width, int K)
{
    const int Height_out = Height - K + 1;
    const int Width_out  = Width  - K + 1;

    #define out_4d(i3, i2, i1, i0) output[(i3) * (Map_out * Height_out * Width_out) + (i2) * (Height_out * Width_out) + (i1) * (Width_out) + i0]
    #define in_4d(i3, i2, i1, i0)  input [(i3) * (Channel * Height * Width)        + (i2) * (Height * Width)        + (i1) * (Width)     + i0]
    #define mk_4d(i3, i2, i1, i0)  mask  [(i3) * (Channel * K * K)                 + (i2) * (K * K)                 + (i1) * (K)         + i0]

    const int tiles_w = (Width_out + blockDim.x - 1) / blockDim.x;
    const int b       = blockIdx.z;
    const int m       = blockIdx.x / tiles_w;
    const int tile_w  = blockIdx.x % tiles_w;
    const int h       = blockIdx.y * blockDim.y + threadIdx.y;
    const int w       = tile_w * blockDim.x + threadIdx.x;

    if (b < Batch && m < Map_out && h < Height_out && w < Width_out) {
        float acc = 0.0f;
        for (int c = 0; c < Channel; c++)
            for (int p = 0; p < K; p++)
                for (int q = 0; q < K; q++)
                    acc += in_4d(b, c, h + p, w + q) * mk_4d(m, c, p, q);
        out_4d(b, m, h, w) = acc;
    }

    #undef out_4d
    #undef in_4d
    #undef mk_4d
}

}  // namespace

void prepare(const float* h_mask, int Map_out, int Channel, int K)
{
    d_mask_bytes_ = static_cast<size_t>(Map_out) * Channel * K * K * sizeof(float);
    CUDA_CHECK(cudaMalloc(&d_mask_, d_mask_bytes_));
    CUDA_CHECK(cudaMemcpy(d_mask_, h_mask, d_mask_bytes_, cudaMemcpyHostToDevice));
}

void release()
{
    if (d_mask_) {
        CUDA_CHECK(cudaFree(d_mask_));
        d_mask_ = nullptr;
        d_mask_bytes_ = 0;
    }
}

void forward(const float* d_input, float* d_output,
             int Batch, int Map_out, int Channel,
             int Height, int Width, int K)
{
    const int Height_out = Height - K + 1;
    const int Width_out  = Width  - K + 1;

    dim3 block(16, 16, 1);
    const int tiles_w = (Width_out + block.x - 1) / block.x;
    const int tiles_h = (Height_out + block.y - 1) / block.y;
    dim3 grid(tiles_w * Map_out, tiles_h, Batch);

    conv_baseline_kernel<<<grid, block>>>(
        d_input, d_mask_, d_output,
        Batch, Map_out, Channel, Height, Width, K
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

const char* name() { return "baseline (one-thread-per-output)"; }

}  // namespace lenet_conv
