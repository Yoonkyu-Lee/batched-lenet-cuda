// req_1: Tensor Cores (TF32) for matrix multiplication.
// Base: kernel-fusion-forward.cu (fused unroll + tiled matmul + implicit permute)
// Change: replace the manual dot-product inner loop with WMMA (16×16×8 TF32)
// so the matmul runs on Tensor Cores. Accumulator stays FP32.
//
// Conv1: mask[4×49]  × unrolled[49×(B*80*80)]   — Map_out=4 padded to 16
// Conv2: mask[16×196] × unrolled[196×(B*34*34)]  — Map_out=16 exact fit
//
// Design: 4 warps/block, each warp computes one 16×16 output tile independently.
// Shared memory per warp: 256 floats (mask[16×8] + input[8×16]; reused for output[16×16]).

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

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 8
#define WARPS_PER_BLOCK 4
#define BLOCK_SIZE (WARPS_PER_BLOCK * 32)
#define SMEM_PER_WARP (WMMA_M * WMMA_N)

__global__ void matmul_conv_tc(const float * __restrict__ mask,
                               const float * __restrict__ input,
                               float * __restrict__ output,
                               int Batch, int Map_out, int Channel,
                               int Height, int Width, int K)
{
    const int Height_out = Height - K + 1;
    const int Width_out  = Width  - K + 1;
    const int unrolled_rows  = Channel * K * K;
    const int width_unrolled = Batch * Height_out * Width_out;
    const int image_size     = Height_out * Width_out;

    const int warpId = threadIdx.x / 32;
    const int laneId = threadIdx.x % 32;

    const int tile_row = blockIdx.y * WMMA_M;
    const int tile_col = (blockIdx.x * WARPS_PER_BLOCK + warpId) * WMMA_N;

    extern __shared__ float smem[];
    float *my_smem = smem + warpId * SMEM_PER_WARP;
    float *my_a    = my_smem;
    float *my_b    = my_smem + WMMA_M * WMMA_K;

    wmma::fragment<wmma::matrix_a,    WMMA_M, WMMA_N, WMMA_K, wmma::precision::tf32, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b,    WMMA_M, WMMA_N, WMMA_K, wmma::precision::tf32, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    wmma::fill_fragment(c_frag, 0.0f);

    for (int k_start = 0; k_start < unrolled_rows; k_start += WMMA_K) {
        // Load mask tile A[WMMA_M × WMMA_K] = [16×8]
        for (int idx = laneId; idx < WMMA_M * WMMA_K; idx += 32) {
            int r = idx / WMMA_K;
            int c = idx % WMMA_K;
            int gr = tile_row + r;
            int gk = k_start + c;
            if (gr < Map_out && gk < unrolled_rows) {
                int ch   = gk / (K * K);
                int koff = gk % (K * K);
                int p    = koff / K;
                int q    = koff % K;
                my_a[idx] = mask[((gr * Channel + ch) * K + p) * K + q];
            } else {
                my_a[idx] = 0.0f;
            }
        }

        // Load input tile B[WMMA_K × WMMA_N] = [8×16] with implicit unrolling
        for (int idx = laneId; idx < WMMA_K * WMMA_N; idx += 32) {
            int r = idx / WMMA_N;
            int c = idx % WMMA_N;
            int gk = k_start + r;
            int gc = tile_col + c;
            if (gk < unrolled_rows && gc < width_unrolled) {
                int ch   = gk / (K * K);
                int koff = gk % (K * K);
                int p    = koff / K;
                int q    = koff % K;
                int b    = gc / image_size;
                int ioff = gc % image_size;
                int h    = ioff / Width_out;
                int w    = ioff % Width_out;
                my_b[idx] = input[((b * Channel + ch) * Height + (h + p)) * Width + (w + q)];
            } else {
                my_b[idx] = 0.0f;
            }
        }

        __syncwarp();

        wmma::load_matrix_sync(a_frag, my_a, WMMA_K);
        wmma::load_matrix_sync(b_frag, my_b, WMMA_N);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    // Store with permutation: matmul output is Map_out × width_unrolled,
    // but final layout is Batch × Map_out × Height_out × Width_out.
    wmma::store_matrix_sync(my_smem, c_frag, WMMA_N, wmma::mem_row_major);
    __syncwarp();

    for (int idx = laneId; idx < WMMA_M * WMMA_N; idx += 32) {
        int r = idx / WMMA_N;
        int c = idx % WMMA_N;
        int gr = tile_row + r;
        int gc = tile_col + c;
        if (gr < Map_out && gc < width_unrolled) {
            int b    = gc / image_size;
            int ioff = gc % image_size;
            int h    = ioff / Width_out;
            int w    = ioff % Width_out;
            output[((b * Map_out + gr) * Height_out + h) * Width_out + w] = my_smem[idx];
        }
    }
}

__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    const int Height_out = Height - K + 1;
    const int Width_out  = Width  - K + 1;
    const size_t input_size  = static_cast<size_t>(Batch) * Channel * Height * Width * sizeof(float);
    const size_t output_size = static_cast<size_t>(Batch) * Map_out * Height_out * Width_out * sizeof(float);
    const size_t mask_size   = static_cast<size_t>(Map_out) * Channel * K * K * sizeof(float);

    (void)host_output;

    CUDA_CHECK(cudaMalloc((void **)device_input_ptr,  input_size));
    CUDA_CHECK(cudaMalloc((void **)device_output_ptr, output_size));
    CUDA_CHECK(cudaMalloc((void **)device_mask_ptr,   mask_size));

    CUDA_CHECK(cudaMemcpy(*device_input_ptr, host_input, input_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(*device_mask_ptr,  host_mask,  mask_size,  cudaMemcpyHostToDevice));
}

__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    const int Height_out = Height - K + 1;
    const int Width_out  = Width  - K + 1;
    const int width_unrolled = Batch * Height_out * Width_out;

    dim3 block_dim(BLOCK_SIZE);
    dim3 grid_dim(
        (width_unrolled + WARPS_PER_BLOCK * WMMA_N - 1) / (WARPS_PER_BLOCK * WMMA_N),
        (Map_out + WMMA_M - 1) / WMMA_M
    );

    const size_t smem_bytes = WARPS_PER_BLOCK * SMEM_PER_WARP * sizeof(float);

    matmul_conv_tc<<<grid_dim, block_dim, smem_bytes>>>(
        device_mask, device_input, device_output,
        Batch, Map_out, Channel, Height, Width, K
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
