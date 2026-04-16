// Fused im2col + tiled matrix multiply, all in one kernel.
//
// Idea: a convolution can be written as a matmul if you "unroll" each input
// patch into a column of a tall matrix:
//
//     C[Map_out × (B·H_out·W_out)] = mask[Map_out × (Channel·K·K)]
//                                  × unrolled_input[(Channel·K·K) × (B·H_out·W_out)]
//
// The naïve approach materializes the unrolled matrix (huge for B=10000) and
// runs a matmul on it. Fusing avoids the materialization: as we tile the
// matmul, each thread computes the unrolled-input element on the fly from the
// raw input tensor. Same arithmetic, far less memory pressure.
//
// Result: this is *correct* but ironically slower than the naïve baseline
// because of the integer division/modulo overhead from on-the-fly index
// decoding. See OPTIMIZATION_JOURNEY.md for what fixes that.

#include "conv.h"
#include "../utils/cuda_check.h"
#include <cuda_runtime.h>

#define MATMUL_TILE_WIDTH 16

namespace lenet_conv {

namespace {

float* d_mask_ = nullptr;
size_t d_mask_bytes_ = 0;

__global__ void matmul_conv_fused(
    const float* __restrict__ mask,
    const float* __restrict__ input,
    float* __restrict__ output,
    int Batch, int Map_out, int Channel,
    int Height, int Width, int K)
{
    const int Height_out     = Height - K + 1;
    const int Width_out      = Width  - K + 1;
    const int unrolled_rows  = Channel * K * K;
    const int width_unrolled = Batch * Height_out * Width_out;
    const int image_size     = Height_out * Width_out;

    __shared__ float tile_mask [MATMUL_TILE_WIDTH][MATMUL_TILE_WIDTH];
    __shared__ float tile_input[MATMUL_TILE_WIDTH][MATMUL_TILE_WIDTH];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int row = blockIdx.y * MATMUL_TILE_WIDTH + ty;
    const int col = blockIdx.x * MATMUL_TILE_WIDTH + tx;

    float acc = 0.0f;

    for (int tile = 0; tile < (unrolled_rows + MATMUL_TILE_WIDTH - 1) / MATMUL_TILE_WIDTH; tile++) {
        // Load mask tile: row-major mask[Map_out × (Channel·K·K)]
        const int mask_col = tile * MATMUL_TILE_WIDTH + tx;
        if (row < Map_out && mask_col < unrolled_rows) {
            const int c = mask_col / (K * K);
            const int kernel_offset = mask_col % (K * K);
            const int p = kernel_offset / K;
            const int q = kernel_offset % K;
            tile_mask[ty][tx] = mask[((row * Channel + c) * K + p) * K + q];
        } else {
            tile_mask[ty][tx] = 0.0f;
        }

        // Load input tile via implicit unroll (decode column index back to b,h,w
        // and channel/p/q from the row index).
        const int input_row = tile * MATMUL_TILE_WIDTH + ty;
        if (col < width_unrolled && input_row < unrolled_rows) {
            const int c = input_row / (K * K);
            const int kernel_offset = input_row % (K * K);
            const int p = kernel_offset / K;
            const int q = kernel_offset % K;

            const int b = col / image_size;
            const int image_offset = col % image_size;
            const int h_out = image_offset / Width_out;
            const int w_out = image_offset % Width_out;

            tile_input[ty][tx] = input[((b * Channel + c) * Height + (h_out + p)) * Width + (w_out + q)];
        } else {
            tile_input[ty][tx] = 0.0f;
        }

        __syncthreads();

        if (row < Map_out && col < width_unrolled) {
            #pragma unroll
            for (int i = 0; i < MATMUL_TILE_WIDTH; i++)
                acc += tile_mask[ty][i] * tile_input[i][tx];
        }

        __syncthreads();
    }

    if (row < Map_out && col < width_unrolled) {
        // Store with implicit permutation: matmul output is
        // [Map_out × Batch × Height_out × Width_out] but the network expects
        // [Batch × Map_out × Height_out × Width_out].
        const int b = col / image_size;
        const int image_offset = col % image_size;
        const int h_out = image_offset / Width_out;
        const int w_out = image_offset % Width_out;
        output[((b * Map_out + row) * Height_out + h_out) * Width_out + w_out] = acc;
    }
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
    if (d_mask_) { CUDA_CHECK(cudaFree(d_mask_)); d_mask_ = nullptr; }
}

void forward(const float* d_input, float* d_output,
             int Batch, int Map_out, int Channel,
             int Height, int Width, int K)
{
    const int Height_out = Height - K + 1;
    const int Width_out  = Width  - K + 1;
    const int width_unrolled = Batch * Height_out * Width_out;

    dim3 block(MATMUL_TILE_WIDTH, MATMUL_TILE_WIDTH, 1);
    dim3 grid(
        (width_unrolled + MATMUL_TILE_WIDTH - 1) / MATMUL_TILE_WIDTH,
        (Map_out + MATMUL_TILE_WIDTH - 1) / MATMUL_TILE_WIDTH,
        1
    );

    matmul_conv_fused<<<grid, block>>>(
        d_mask_, d_input, d_output,
        Batch, Map_out, Channel, Height, Width, K
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

const char* name() { return "fused im2col + tiled matmul"; }

}  // namespace lenet_conv
