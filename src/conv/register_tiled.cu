// THE FINAL VARIANT: fused im2col + WMMA + N-coarsened register tiling +
// shared-memory K-index lookup.
//
// What "register tiling" buys us
// ------------------------------
// Each warp computes COARSEN_N back-to-back 16×16 output tiles instead of one.
// The mask A-fragment is loaded once per K-step and *reused* across all
// COARSEN_N B-tiles. That increases arithmetic intensity by a factor of
// COARSEN_N — without changing the total FLOPs, just by holding more partial
// sums (c_frag[]) in registers.
//
// What "shared-memory K-index lookup" buys us
// -------------------------------------------
// The K-loop's hot inner work decodes an unrolled-row index `gk` into
// (channel, p, q) via integer division. Rather than divide on every lane on
// every iteration, all warps in the block cooperate to write a precomputed
// (ch, p, q) table into shared memory once. The K-loop then reads from that
// table; lanes that share the same `gk` get a free shared-memory broadcast.
//
// Combined effect on Conv2 of our LeNet-5 variant: 47.9 → 12.2 ms (B=10000),
// dropping the total layer time from 81.5 ms (fused-only baseline) to 28.9 ms.
//
// Tunable parameters
// ------------------
// WARPS_PER_BLOCK and COARSEN_N were swept; the (8, 4) combination won on A40.
// See OPTIMIZATION_JOURNEY.md for the sweep table.

#include "conv.h"
#include "../utils/cuda_check.h"
#include <cuda_runtime.h>
#include <mma.h>

using namespace nvcuda;

namespace lenet_conv {

namespace {

constexpr int MAX_MASK_ELEMENTS = 4096;
__constant__ float c_mask[MAX_MASK_ELEMENTS];

constexpr int WMMA_M           = 16;
constexpr int WMMA_N           = 16;
constexpr int WMMA_K           = 8;
constexpr int WARPS_PER_BLOCK  = 8;   // sweep winner
constexpr int COARSEN_N        = 4;   // sweep winner
constexpr int BLOCK_SIZE       = WARPS_PER_BLOCK * 32;
constexpr int MAX_UNROLLED_ROWS = 256;  // Conv2: Channel=4 * K*K=49 = 196 ≤ 256

__global__ void matmul_conv_tc_coarsened(
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
    const int base_col = (blockIdx.x * WARPS_PER_BLOCK + warpId) * WMMA_N * COARSEN_N;

    // Per-warp shared memory layout:
    //   A staging:        WMMA_M × WMMA_K  = 128 floats
    //   B staging × N:    COARSEN_N × WMMA_K × WMMA_N = COARSEN_N × 128 floats
    //   col precomp × N:  COARSEN_N × (3 × WMMA_N) ints
    constexpr int A_SIZE   = WMMA_M * WMMA_K;
    constexpr int B_SIZE   = WMMA_K * WMMA_N;
    constexpr int COL_SIZE = WMMA_N * 3;
    constexpr int PER_WARP = A_SIZE + COARSEN_N * B_SIZE + COARSEN_N * COL_SIZE;

    extern __shared__ float smem[];
    float* my_smem    = smem + warpId * PER_WARP;
    float* my_a       = my_smem;
    float* my_b_base  = my_smem + A_SIZE;
    int*   col_base   = reinterpret_cast<int*>(my_b_base + COARSEN_N * B_SIZE);

    // Block-shared K-index lookup table (one per block; not per warp).
    int* sh_ch = reinterpret_cast<int*>(smem + WARPS_PER_BLOCK * PER_WARP);
    int* sh_p  = sh_ch + MAX_UNROLLED_ROWS;
    int* sh_q  = sh_p  + MAX_UNROLLED_ROWS;

    // Cooperative fill of K-index table.
    for (int k = threadIdx.x; k < unrolled_rows; k += BLOCK_SIZE) {
        const int ch   = k / KK;
        const int koff = k - ch * KK;
        sh_ch[k] = ch;
        sh_p[k]  = koff / K;
        sh_q[k]  = koff - sh_p[k] * K;
    }
    __syncthreads();

    // Per-warp: precompute (b, h, w) for every column this warp will touch
    // (COARSEN_N tiles × WMMA_N columns each).
    for (int n = 0; n < COARSEN_N; n++) {
        int* col_b = col_base + n * WMMA_N * 3;
        int* col_h = col_b + WMMA_N;
        int* col_w = col_h + WMMA_N;
        if (laneId < WMMA_N) {
            const int gc = base_col + n * WMMA_N + laneId;
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
    }
    __syncwarp();

    // Fragments: one A-frag, COARSEN_N B-frags, and COARSEN_N accumulators in
    // registers. The accumulators are the "register tile" — they outlive the
    // shared-memory tiles.
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, wmma::precision::tf32, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, wmma::precision::tf32, wmma::row_major> b_frag[COARSEN_N];
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag[COARSEN_N];

    #pragma unroll
    for (int n = 0; n < COARSEN_N; n++) wmma::fill_fragment(c_frag[n], 0.0f);

    for (int k_start = 0; k_start < unrolled_rows; k_start += WMMA_K) {
        // Load A[16×8] once per K-step (shared across all COARSEN_N tiles).
        for (int idx = laneId; idx < A_SIZE; idx += 32) {
            const int r  = idx / WMMA_K;
            const int c  = idx % WMMA_K;
            const int gr = tile_row + r;
            const int gk = k_start + c;
            my_a[idx] = (gr < Map_out && gk < unrolled_rows)
                        ? c_mask[gr * unrolled_rows + gk]
                        : 0.0f;
        }

        // Load B[8×16] for each of the COARSEN_N tiles, reading (ch, p, q)
        // from the shared-memory K-index table.
        for (int n = 0; n < COARSEN_N; n++) {
            float* my_b      = my_b_base + n * B_SIZE;
            int*   col_b_n   = col_base + n * WMMA_N * 3;
            int*   col_h_n   = col_b_n + WMMA_N;
            int*   col_w_n   = col_h_n + WMMA_N;
            const int tile_col_n = base_col + n * WMMA_N;

            for (int idx = laneId; idx < B_SIZE; idx += 32) {
                const int r  = idx / WMMA_N;
                const int c  = idx % WMMA_N;
                const int gk = k_start + r;
                const int gc = tile_col_n + c;
                if (gk < unrolled_rows && gc < width_unrolled) {
                    const int ch = sh_ch[gk];
                    const int p  = sh_p[gk];
                    const int q  = sh_q[gk];
                    my_b[idx] = input[((col_b_n[c] * Channel + ch) * Height + (col_h_n[c] + p)) * Width + (col_w_n[c] + q)];
                } else {
                    my_b[idx] = 0.0f;
                }
            }
        }

        __syncwarp();

        // MMA: A is loaded once, applied to each B tile.
        wmma::load_matrix_sync(a_frag, my_a, WMMA_K);
        #pragma unroll
        for (int n = 0; n < COARSEN_N; n++) {
            wmma::load_matrix_sync(b_frag[n], my_b_base + n * B_SIZE, WMMA_N);
            wmma::mma_sync(c_frag[n], a_frag, b_frag[n], c_frag[n]);
        }
    }

    // Store each C-tile with implicit permutation. Reuse my_smem as scratch.
    for (int n = 0; n < COARSEN_N; n++) {
        wmma::store_matrix_sync(my_smem, c_frag[n], WMMA_N, wmma::mem_row_major);
        __syncwarp();

        int* col_b_n = col_base + n * WMMA_N * 3;
        int* col_h_n = col_b_n + WMMA_N;
        int* col_w_n = col_h_n + WMMA_N;
        const int tile_col_n = base_col + n * WMMA_N;

        for (int idx = laneId; idx < WMMA_M * WMMA_N; idx += 32) {
            const int r  = idx / WMMA_N;
            const int c  = idx % WMMA_N;
            const int gr = tile_row + r;
            const int gc = tile_col_n + c;
            if (gr < Map_out && gc < width_unrolled) {
                output[((col_b_n[c] * Map_out + gr) * Height_out + col_h_n[c]) * Width_out + col_w_n[c]] = my_smem[idx];
            }
        }
        __syncwarp();
    }
}

}  // namespace

void prepare(const float* h_mask, int Map_out, int Channel, int K)
{
    const size_t mask_elems = static_cast<size_t>(Map_out) * Channel * K * K;
    if (mask_elems > MAX_MASK_ELEMENTS) {
        std::fprintf(stderr, "register_tiled: mask too large for constant memory (%zu > %d)\n",
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

    constexpr int A_SIZE   = WMMA_M * WMMA_K;
    constexpr int B_SIZE   = WMMA_K * WMMA_N;
    constexpr int COL_SIZE = WMMA_N * 3;
    constexpr int PER_WARP = A_SIZE + COARSEN_N * B_SIZE + COARSEN_N * COL_SIZE;

    const int cols_per_block = WARPS_PER_BLOCK * COARSEN_N * WMMA_N;

    dim3 block(BLOCK_SIZE);
    dim3 grid(
        (width_unrolled + cols_per_block - 1) / cols_per_block,
        (Map_out + WMMA_M - 1) / WMMA_M
    );
    const size_t smem_bytes = (WARPS_PER_BLOCK * PER_WARP + 3 * MAX_UNROLLED_ROWS) * sizeof(float);

    matmul_conv_tc_coarsened<<<grid, block, smem_bytes>>>(
        d_input, d_output, Batch, Map_out, Channel, Height, Width, K
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

const char* name() { return "fused + WMMA + N-coarsened register tiling"; }

}  // namespace lenet_conv
