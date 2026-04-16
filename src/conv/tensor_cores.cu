// Fused im2col + matmul, with the inner dot-product replaced by a TF32 WMMA
// fragment multiply (Tensor Cores). Same memory layout as fused.cu, just a
// different inner kernel.
//
// Setup (compared to fused.cu):
//   - 4 warps per block, each warp computes one 16×16 output tile.
//   - Mask is in __constant__ memory (broadcast across all warps).
//   - Per-tile column indices (b, h_out, w_out) are precomputed once into
//     shared memory to avoid redundant integer division inside the K-loop.
//   - WMMA shape: 16×16×8 (M×N×K), TF32 inputs, FP32 accumulator.
//
// On its own, this is ~14% faster than the fused baseline. The big win arrives
// in register_tiled.cu when each warp also reuses the loaded A-fragment across
// multiple output tiles.

#include "conv.h"
#include "../utils/cuda_check.h"
#include <cuda_runtime.h>
#include <mma.h>

using namespace nvcuda;

namespace lenet_conv {

namespace {

// Conv2 in our LeNet variant has the largest mask: 16 × 4 × 7 × 7 = 3,136 floats.
// 4096 floats = 16 KB, well under the 64 KB constant-memory budget on sm_86.
constexpr int MAX_MASK_ELEMENTS = 4096;
__constant__ float c_mask[MAX_MASK_ELEMENTS];

constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 8;
constexpr int WARPS_PER_BLOCK = 4;
constexpr int BLOCK_SIZE = WARPS_PER_BLOCK * 32;

__global__ void matmul_conv_tc(
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
    const int KK             = K * K;

    const int warpId = threadIdx.x / 32;
    const int laneId = threadIdx.x % 32;

    const int tile_row = blockIdx.y * WMMA_M;
    const int tile_col = (blockIdx.x * WARPS_PER_BLOCK + warpId) * WMMA_N;

    // Per-warp shared-memory layout: 304 floats per warp
    //   [0   ..  127] = A staging (mask tile, 16×8)
    //   [128 ..  255] = B staging (input tile, 8×16) — also reused for 16×16 store
    //   [256 ..  271] = col_b[16] (precomputed batch index per tile column)
    //   [272 ..  287] = col_h[16] (precomputed h_out per tile column)
    //   [288 ..  303] = col_w[16] (precomputed w_out per tile column)
    constexpr int PER_WARP = 304;

    extern __shared__ float smem[];
    float* my_smem = smem + warpId * PER_WARP;
    float* my_a    = my_smem;
    float* my_b    = my_smem + WMMA_M * WMMA_K;
    int*   col_b   = reinterpret_cast<int*>(my_smem + 256);
    int*   col_h   = reinterpret_cast<int*>(my_smem + 272);
    int*   col_w   = reinterpret_cast<int*>(my_smem + 288);

    if (laneId < WMMA_N) {
        const int gc = tile_col + laneId;
        if (gc < width_unrolled) {
            col_b[laneId] = gc / image_size;
            const int ioff = gc % image_size;
            col_h[laneId] = ioff / Width_out;
            col_w[laneId] = ioff % Width_out;
        } else {
            col_b[laneId] = 0;
            col_h[laneId] = 0;
            col_w[laneId] = 0;
        }
    }
    __syncwarp();

    wmma::fragment<wmma::matrix_a,    WMMA_M, WMMA_N, WMMA_K, wmma::precision::tf32, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b,    WMMA_M, WMMA_N, WMMA_K, wmma::precision::tf32, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    for (int k_start = 0; k_start < unrolled_rows; k_start += WMMA_K) {
        // Stage A[16×8] from constant memory.
        for (int idx = laneId; idx < WMMA_M * WMMA_K; idx += 32) {
            const int r  = idx / WMMA_K;
            const int c  = idx % WMMA_K;
            const int gr = tile_row + r;
            const int gk = k_start + c;
            my_a[idx] = (gr < Map_out && gk < unrolled_rows)
                        ? c_mask[gr * unrolled_rows + gk]
                        : 0.0f;
        }

        // Stage B[8×16] from input via implicit unroll. The K-loop integer
        // divisions remain here (they're still cheaper than reading them from
        // a lookup table under heavy contention — see OPTIMIZATION_JOURNEY).
        for (int idx = laneId; idx < WMMA_K * WMMA_N; idx += 32) {
            const int r  = idx / WMMA_N;
            const int c  = idx % WMMA_N;
            const int gk = k_start + r;
            const int gc = tile_col + c;
            if (gk < unrolled_rows && gc < width_unrolled) {
                const int ch   = gk / KK;
                const int koff = gk - ch * KK;
                const int p    = koff / K;
                const int q    = koff - p * K;
                my_b[idx] = input[((col_b[c] * Channel + ch) * Height + (col_h[c] + p)) * Width + (col_w[c] + q)];
            } else {
                my_b[idx] = 0.0f;
            }
        }

        __syncwarp();
        wmma::load_matrix_sync(a_frag, my_a, WMMA_K);
        wmma::load_matrix_sync(b_frag, my_b, WMMA_N);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    // Store with implicit permutation. Reuse the per-warp scratch (256 floats >=
    // 16×16 needed).
    wmma::store_matrix_sync(my_smem, c_frag, WMMA_N, wmma::mem_row_major);
    __syncwarp();

    for (int idx = laneId; idx < WMMA_M * WMMA_N; idx += 32) {
        const int r  = idx / WMMA_N;
        const int c  = idx % WMMA_N;
        const int gr = tile_row + r;
        const int gc = tile_col + c;
        if (gr < Map_out && gc < width_unrolled) {
            output[((col_b[c] * Map_out + gr) * Height_out + col_h[c]) * Width_out + col_w[c]] = my_smem[idx];
        }
    }
}

}  // namespace

void prepare(const float* h_mask, int Map_out, int Channel, int K)
{
    const size_t mask_elems = static_cast<size_t>(Map_out) * Channel * K * K;
    if (mask_elems > MAX_MASK_ELEMENTS) {
        std::fprintf(stderr, "tensor_cores: mask too large for constant memory (%zu > %d)\n",
                     mask_elems, MAX_MASK_ELEMENTS);
        std::exit(EXIT_FAILURE);
    }
    CUDA_CHECK(cudaMemcpyToSymbol(c_mask, h_mask, mask_elems * sizeof(float)));
}

void release() { /* nothing to free */ }

void forward(const float* d_input, float* d_output,
             int Batch, int Map_out, int Channel,
             int Height, int Width, int K)
{
    const int Height_out = Height - K + 1;
    const int Width_out  = Width  - K + 1;
    const int width_unrolled = Batch * Height_out * Width_out;

    dim3 block(BLOCK_SIZE);
    dim3 grid(
        (width_unrolled + WARPS_PER_BLOCK * WMMA_N - 1) / (WARPS_PER_BLOCK * WMMA_N),
        (Map_out + WMMA_M - 1) / WMMA_M
    );
    const size_t smem_bytes = WARPS_PER_BLOCK * 304 * sizeof(float);

    matmul_conv_tc<<<grid, block, smem_bytes>>>(
        d_input, d_output, Batch, Map_out, Channel, Height, Width, K
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

const char* name() { return "fused + Tensor Cores (TF32, basic)"; }

}  // namespace lenet_conv
