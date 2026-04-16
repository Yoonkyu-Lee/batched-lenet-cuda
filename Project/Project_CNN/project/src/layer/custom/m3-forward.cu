// op_6: Joint register and shared memory tiling (4 pts).
// Base: kernel-fusion-forward.cu (fused unroll + tiled matmul + implicit permute)
// Change: each warp computes multiple WMMA_N-wide output tiles (N-coarsening),
// keeping multiple c_frag accumulators in registers. The mask A-fragment is loaded
// once per K-step and reused across all N-tiles → higher arithmetic intensity.
//
// COARSEN_N = number of 16-wide N-tiles per warp.

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <mma.h>
#include "gpu-new-forward.h"

using namespace nvcuda;

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err__ = (call);                                            \
        if (err__ != cudaSuccess) {                                            \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__      \
                      << ": " << cudaGetErrorString(err__) << std::endl;      \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)

#define MAX_MASK_ELEMENTS 4096
__constant__ float c_mask[MAX_MASK_ELEMENTS];

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 8
#define WARPS_PER_BLOCK 8
#define COARSEN_N 4
#define BLOCK_SIZE (WARPS_PER_BLOCK * 32)
#define MAX_UNROLLED_ROWS 256

__global__ void matmul_conv_tc_coarsened(const float * __restrict__ input,
                                         float * __restrict__ output,
                                         int Batch, int Map_out, int Channel,
                                         int Height, int Width, int K)
{
    const int Height_out = Height - K + 1;
    const int Width_out  = Width  - K + 1;
    const int unrolled_rows  = Channel * K * K;
    const int width_unrolled = Batch * Height_out * Width_out;
    const int image_size     = Height_out * Width_out;
    const int KK = K * K;

    const int warpId = threadIdx.x / 32;
    const int laneId = threadIdx.x % 32;

    const int tile_row = blockIdx.y * WMMA_M;
    const int base_col = (blockIdx.x * WARPS_PER_BLOCK + warpId) * WMMA_N * COARSEN_N;

    // Shared memory layout per warp:
    // A tile: WMMA_M * WMMA_K = 128 floats (mask, loaded once per K-step, reused COARSEN_N times)
    // B tiles: COARSEN_N * WMMA_K * WMMA_N = COARSEN_N * 128 floats
    // Col precomp: COARSEN_N * WMMA_N * 3 ints (b, h, w)
    // Output staging: WMMA_M * WMMA_N = 256 floats (reused per N-tile during store)
    const int A_SIZE = WMMA_M * WMMA_K;
    const int B_SIZE = WMMA_K * WMMA_N;
    const int COL_SIZE = WMMA_N * 3;
    const int PER_WARP = A_SIZE + COARSEN_N * B_SIZE + COARSEN_N * COL_SIZE;

    extern __shared__ float smem[];
    float *my_smem = smem + warpId * PER_WARP;
    float *my_a    = my_smem;
    float *my_b_base = my_smem + A_SIZE;
    int   *col_base  = (int*)(my_b_base + COARSEN_N * B_SIZE);

    // K-index shared tables (shared by all warps).
    extern __shared__ float smem_global[];
    int *sh_ch = (int*)(smem + WARPS_PER_BLOCK * PER_WARP);
    int *sh_p  = sh_ch + MAX_UNROLLED_ROWS;
    int *sh_q  = sh_p  + MAX_UNROLLED_ROWS;

    for (int k = threadIdx.x; k < unrolled_rows; k += BLOCK_SIZE) {
        int ch  = k / KK;
        int koff = k - ch * KK;
        sh_ch[k] = ch;
        sh_p[k]  = koff / K;
        sh_q[k]  = koff - sh_p[k] * K;
    }
    __syncthreads();

    // Precompute column indices for all COARSEN_N tiles.
    for (int n = 0; n < COARSEN_N; n++) {
        int *col_b = col_base + n * WMMA_N * 3;
        int *col_h = col_b + WMMA_N;
        int *col_w = col_h + WMMA_N;
        if (laneId < WMMA_N) {
            int gc = base_col + n * WMMA_N + laneId;
            if (gc < width_unrolled) {
                col_b[laneId] = gc / image_size;
                int ioff      = gc % image_size;
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

    // Declare fragments: 1 A-frag, COARSEN_N B-frags and C-frags (C in registers!).
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, wmma::precision::tf32, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, wmma::precision::tf32, wmma::row_major> b_frag[COARSEN_N];
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag[COARSEN_N];

    for (int n = 0; n < COARSEN_N; n++)
        wmma::fill_fragment(c_frag[n], 0.0f);

    for (int k_start = 0; k_start < unrolled_rows; k_start += WMMA_K) {
        // Load A[16×8] once.
        for (int idx = laneId; idx < A_SIZE; idx += 32) {
            int r  = idx / WMMA_K;
            int c  = idx % WMMA_K;
            int gr = tile_row + r;
            int gk = k_start + c;
            my_a[idx] = (gr < Map_out && gk < unrolled_rows)
                        ? c_mask[gr * unrolled_rows + gk]
                        : 0.0f;
        }

        // Load B[8×16] for each of the COARSEN_N tiles.
        for (int n = 0; n < COARSEN_N; n++) {
            float *my_b = my_b_base + n * B_SIZE;
            int *col_b_n = col_base + n * WMMA_N * 3;
            int *col_h_n = col_b_n + WMMA_N;
            int *col_w_n = col_h_n + WMMA_N;
            int tile_col_n = base_col + n * WMMA_N;

            for (int idx = laneId; idx < B_SIZE; idx += 32) {
                int r  = idx / WMMA_N;
                int c  = idx % WMMA_N;
                int gk = k_start + r;
                int gc = tile_col_n + c;
                if (gk < unrolled_rows && gc < width_unrolled) {
                    int ch = sh_ch[gk];
                    int p  = sh_p[gk];
                    int q  = sh_q[gk];
                    my_b[idx] = input[((col_b_n[c] * Channel + ch) * Height + (col_h_n[c] + p)) * Width + (col_w_n[c] + q)];
                } else {
                    my_b[idx] = 0.0f;
                }
            }
        }

        __syncwarp();

        // MMA: A is loaded once, applied to each B tile.
        wmma::load_matrix_sync(a_frag, my_a, WMMA_K);
        for (int n = 0; n < COARSEN_N; n++) {
            wmma::load_matrix_sync(b_frag[n], my_b_base + n * B_SIZE, WMMA_N);
            wmma::mma_sync(c_frag[n], a_frag, b_frag[n], c_frag[n]);
        }
    }

    // Store each C tile with permutation.
    float *store_buf = my_a; // reuse A staging (256 floats needed, A is 128... need more)
    // Actually we need 256 floats for 16×16 output but my_a is only 128.
    // Use the first B slot (my_b_base) which is 128 floats. Together: 256 floats.
    // But they're not contiguous... let's just use my_smem directly (PER_WARP > 256).
    // my_smem has PER_WARP floats starting at offset 0. First 256 floats are fine for store.
    for (int n = 0; n < COARSEN_N; n++) {
        wmma::store_matrix_sync(my_smem, c_frag[n], WMMA_N, wmma::mem_row_major);
        __syncwarp();

        int *col_b_n = col_base + n * WMMA_N * 3;
        int *col_h_n = col_b_n + WMMA_N;
        int *col_w_n = col_h_n + WMMA_N;
        int tile_col_n = base_col + n * WMMA_N;

        for (int idx = laneId; idx < WMMA_M * WMMA_N; idx += 32) {
            int r  = idx / WMMA_N;
            int c  = idx % WMMA_N;
            int gr = tile_row + r;
            int gc = tile_col_n + c;
            if (gr < Map_out && gc < width_unrolled) {
                output[((col_b_n[c] * Map_out + gr) * Height_out + col_h_n[c]) * Width_out + col_w_n[c]] = my_smem[idx];
            }
        }
        __syncwarp();
    }
}

__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    const int Height_out = Height - K + 1;
    const int Width_out  = Width  - K + 1;
    const size_t input_size  = static_cast<size_t>(Batch) * Channel * Height * Width * sizeof(float);
    const size_t output_size = static_cast<size_t>(Batch) * Map_out * Height_out * Width_out * sizeof(float);
    const size_t mask_elems  = static_cast<size_t>(Map_out) * Channel * K * K;

    (void)host_output;
    CUDA_CHECK(cudaMalloc((void **)device_input_ptr,  input_size));
    CUDA_CHECK(cudaMalloc((void **)device_output_ptr, output_size));
    CUDA_CHECK(cudaMalloc((void **)device_mask_ptr,   mask_elems * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(*device_input_ptr, host_input, input_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpyToSymbol(c_mask, host_mask, mask_elems * sizeof(float)));
}

__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    (void)device_mask;
    const int Height_out = Height - K + 1;
    const int Width_out  = Width  - K + 1;
    const int width_unrolled = Batch * Height_out * Width_out;

    const int tiles_per_warp = COARSEN_N;
    const int cols_per_block = WARPS_PER_BLOCK * tiles_per_warp * WMMA_N;

    dim3 block_dim(BLOCK_SIZE);
    dim3 grid_dim(
        (width_unrolled + cols_per_block - 1) / cols_per_block,
        (Map_out + WMMA_M - 1) / WMMA_M
    );

    const int A_SIZE = WMMA_M * WMMA_K;
    const int B_SIZE = WMMA_K * WMMA_N;
    const int COL_SIZE = WMMA_N * 3;
    const int PER_WARP = A_SIZE + COARSEN_N * B_SIZE + COARSEN_N * COL_SIZE;
    const size_t smem_bytes = (WARPS_PER_BLOCK * PER_WARP + 3 * MAX_UNROLLED_ROWS) * sizeof(float);

    matmul_conv_tc_coarsened<<<grid_dim, block_dim, smem_bytes>>>(
        device_input, device_output, Batch, Map_out, Channel, Height, Width, K
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    const int Height_out = Height - K + 1;
    const int Width_out  = Width  - K + 1;
    const size_t output_size = static_cast<size_t>(Batch) * Map_out * Height_out * Width_out * sizeof(float);
    CUDA_CHECK(cudaMemcpy(host_output, device_output, output_size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(device_output));
    CUDA_CHECK(cudaFree(device_input));
    CUDA_CHECK(cudaFree(device_mask));
}

__host__ void GPUInterface::get_device_properties()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    for (int dev = 0; dev < deviceCount; dev++) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);
        std::cout << "Device " << dev << " name: " << deviceProp.name << std::endl;
        std::cout << "Computational capabilities: " << deviceProp.major << "." << deviceProp.minor << std::endl;
        std::cout << "Max Global memory size: " << deviceProp.totalGlobalMem << std::endl;
        std::cout << "Max Constant memory size: " << deviceProp.totalConstMem << std::endl;
        std::cout << "Max Shared memory size per block: " << deviceProp.sharedMemPerBlock << std::endl;
        std::cout << "Max threads per block: " << deviceProp.maxThreadsPerBlock << std::endl;
        std::cout << "Max block dimensions: " << deviceProp.maxThreadsDim[0] << " x, " << deviceProp.maxThreadsDim[1] << " y, " << deviceProp.maxThreadsDim[2] << " z" << std::endl;
        std::cout << "Max grid dimensions: " << deviceProp.maxGridSize[0] << " x, " << deviceProp.maxGridSize[1] << " y, " << deviceProp.maxGridSize[2] << " z" << std::endl;
        std::cout << "Warp Size: " << deviceProp.warpSize << std::endl;
    }
}

#undef CUDA_CHECK
