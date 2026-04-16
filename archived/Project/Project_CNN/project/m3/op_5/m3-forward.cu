// op_5: FP16 (__half) arithmetic with FP32 accumulator (2 pts).
// Base: kernel-fusion-forward.cu (fused unroll + tiled matmul + implicit permute)
// Change: mask and input are stored in __half. The inner MAC multiplies two halves
// and accumulates into FP32 (preserves precision against catastrophic cancellation).
//
// Using __half2 (4 pts) would require laying out shared-memory B-tiles with a
// consecutive-K stride; that restructuring is non-trivial for the fused kernel, so
// we take the simpler __half path here.

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <cuda_fp16.h>
#include "gpu-new-forward.h"

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err__ = (call);                                            \
        if (err__ != cudaSuccess) {                                            \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__      \
                      << ": " << cudaGetErrorString(err__) << std::endl;      \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)

#define MATMUL_TILE_WIDTH 16

static __half *d_input_h  = nullptr;
static __half *d_mask_h   = nullptr;

__global__ void float_to_half(const float *src, __half *dst, size_t n) {
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n) dst[idx] = __float2half(src[idx]);
}

__global__ void matmul_conv_fused_fp16(const __half *mask, const __half *input, float *output,
                                       int Batch, int Map_out, int Channel,
                                       int Height, int Width, int K)
{
    const int Height_out = Height - K + 1;
    const int Width_out  = Width  - K + 1;
    const int unrolled_rows  = Channel * K * K;
    const int width_unrolled = Batch * Height_out * Width_out;
    const int image_size     = Height_out * Width_out;

    __shared__ __half tile_mask [MATMUL_TILE_WIDTH][MATMUL_TILE_WIDTH];
    __shared__ __half tile_input[MATMUL_TILE_WIDTH][MATMUL_TILE_WIDTH];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int row = blockIdx.y * MATMUL_TILE_WIDTH + ty;
    const int col = blockIdx.x * MATMUL_TILE_WIDTH + tx;

    float acc = 0.0f;

    for (int tile = 0; tile < (unrolled_rows + MATMUL_TILE_WIDTH - 1) / MATMUL_TILE_WIDTH; tile++) {
        const int mask_col = tile * MATMUL_TILE_WIDTH + tx;
        if (row < Map_out && mask_col < unrolled_rows) {
            const int c = mask_col / (K * K);
            const int kernel_offset = mask_col % (K * K);
            const int p = kernel_offset / K;
            const int q = kernel_offset % K;
            tile_mask[ty][tx] = mask[((row * Channel + c) * K + p) * K + q];
        } else {
            tile_mask[ty][tx] = __float2half(0.0f);
        }

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
            tile_input[ty][tx] = __float2half(0.0f);
        }

        __syncthreads();

        if (row < Map_out && col < width_unrolled) {
            #pragma unroll
            for (int i = 0; i < MATMUL_TILE_WIDTH; i++) {
                // FP16 multiply, FP32 accumulate.
                __half prod = __hmul(tile_mask[ty][i], tile_input[i][tx]);
                acc += __half2float(prod);
            }
        }
        __syncthreads();
    }

    if (row < Map_out && col < width_unrolled) {
        const int b = col / image_size;
        const int image_offset = col % image_size;
        const int h_out = image_offset / Width_out;
        const int w_out = image_offset % Width_out;
        output[((b * Map_out + row) * Height_out + h_out) * Width_out + w_out] = acc;
    }
}

__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    const int Height_out = Height - K + 1;
    const int Width_out  = Width  - K + 1;
    const size_t input_elems  = static_cast<size_t>(Batch) * Channel * Height * Width;
    const size_t output_size  = static_cast<size_t>(Batch) * Map_out * Height_out * Width_out * sizeof(float);
    const size_t mask_elems   = static_cast<size_t>(Map_out) * Channel * K * K;

    (void)host_output;

    CUDA_CHECK(cudaMalloc((void **)device_input_ptr,  input_elems * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)device_output_ptr, output_size));
    CUDA_CHECK(cudaMalloc((void **)device_mask_ptr,   mask_elems * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(*device_input_ptr, host_input, input_elems * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(*device_mask_ptr,  host_mask,  mask_elems * sizeof(float), cudaMemcpyHostToDevice));

    // Allocate and fill __half mirrors.
    CUDA_CHECK(cudaMalloc(&d_input_h, input_elems * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_mask_h,  mask_elems  * sizeof(__half)));

    const int bs = 256;
    size_t in_grid = (input_elems + bs - 1) / bs;
    size_t mk_grid = (mask_elems  + bs - 1) / bs;
    float_to_half<<<static_cast<unsigned int>(in_grid), bs>>>(*device_input_ptr, d_input_h, input_elems);
    float_to_half<<<static_cast<unsigned int>(mk_grid), bs>>>(*device_mask_ptr,  d_mask_h,  mask_elems);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    (void)device_input; (void)device_mask;
    const int Height_out = Height - K + 1;
    const int Width_out  = Width  - K + 1;
    const int width_unrolled = Batch * Height_out * Width_out;

    dim3 block_dim(MATMUL_TILE_WIDTH, MATMUL_TILE_WIDTH, 1);
    dim3 grid_dim(
        (width_unrolled + MATMUL_TILE_WIDTH - 1) / MATMUL_TILE_WIDTH,
        (Map_out + MATMUL_TILE_WIDTH - 1) / MATMUL_TILE_WIDTH,
        1
    );

    matmul_conv_fused_fp16<<<grid_dim, block_dim>>>(
        d_mask_h, d_input_h, device_output, Batch, Map_out, Channel, Height, Width, K
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

    if (d_input_h) { CUDA_CHECK(cudaFree(d_input_h)); d_input_h = nullptr; }
    if (d_mask_h)  { CUDA_CHECK(cudaFree(d_mask_h));  d_mask_h  = nullptr; }

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
