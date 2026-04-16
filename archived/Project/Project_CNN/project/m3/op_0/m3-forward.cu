// op_0: Constant memory for the convolution mask.
// Base: kernel-fusion-forward.cu (fused unroll + tiled matmul + implicit permute)
// Change: mask is copied into __constant__ memory once in prolog and read from
// there inside the kernel (instead of the device mask pointer).

#include <cmath>
#include <cstdlib>
#include <iostream>
#include "gpu-new-forward.h"
#include "matmul.h"

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err__ = (call);                                            \
        if (err__ != cudaSuccess) {                                            \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__      \
                      << ": " << cudaGetErrorString(err__) << std::endl;      \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)

// Largest mask: Conv2 = 16*4*7*7 = 3136 floats. Fits in 64 KB constant mem.
#define MAX_MASK_ELEMENTS 4096
__constant__ float c_mask[MAX_MASK_ELEMENTS];

__global__ void matmul_conv_fused(const float *input, float *output,
                                  int Batch, int Map_out, int Channel,
                                  int Height, int Width, int K)
{
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    const int unrolled_rows = Channel * K * K;
    const int width_unrolled = Batch * Height_out * Width_out;
    const int image_size = Height_out * Width_out;

    __shared__ float tile_mask[MATMUL_TILE_WIDTH][MATMUL_TILE_WIDTH];
    __shared__ float tile_input[MATMUL_TILE_WIDTH][MATMUL_TILE_WIDTH];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int row = blockIdx.y * MATMUL_TILE_WIDTH + ty;
    const int col = blockIdx.x * MATMUL_TILE_WIDTH + tx;

    float acc = 0.0f;

    for (int tile = 0; tile < (unrolled_rows + MATMUL_TILE_WIDTH - 1) / MATMUL_TILE_WIDTH; tile++) {
        const int mask_col = tile * MATMUL_TILE_WIDTH + tx;
        if (row < Map_out && mask_col < unrolled_rows) {
            tile_mask[ty][tx] = c_mask[row * unrolled_rows + mask_col];
        } else {
            tile_mask[ty][tx] = 0.0f;
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
            tile_input[ty][tx] = 0.0f;
        }

        __syncthreads();

        if (row < Map_out && col < width_unrolled) {
            for (int i = 0; i < MATMUL_TILE_WIDTH; i++) {
                acc += tile_mask[ty][i] * tile_input[i][tx];
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
    const int Width_out = Width - K + 1;
    const size_t input_size = static_cast<size_t>(Batch) * Channel * Height * Width * sizeof(float);
    const size_t output_size = static_cast<size_t>(Batch) * Map_out * Height_out * Width_out * sizeof(float);
    const size_t mask_elems = static_cast<size_t>(Map_out) * Channel * K * K;

    (void)host_output;

    CUDA_CHECK(cudaMalloc((void **) device_input_ptr, input_size));
    CUDA_CHECK(cudaMalloc((void **) device_output_ptr, output_size));
    CUDA_CHECK(cudaMalloc((void **) device_mask_ptr, mask_elems * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(*device_input_ptr, host_input, input_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpyToSymbol(c_mask, host_mask, mask_elems * sizeof(float)));
}

__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    (void)device_mask;
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    const int width_unrolled = Batch * Height_out * Width_out;

    dim3 block_dim(MATMUL_TILE_WIDTH, MATMUL_TILE_WIDTH, 1);
    dim3 grid_dim(
        (width_unrolled + MATMUL_TILE_WIDTH - 1) / MATMUL_TILE_WIDTH,
        (Map_out + MATMUL_TILE_WIDTH - 1) / MATMUL_TILE_WIDTH,
        1
    );

    matmul_conv_fused<<<grid_dim, block_dim>>>(
        device_input, device_output, Batch, Map_out, Channel, Height, Width, K
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
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
    for(int dev = 0; dev < deviceCount; dev++) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);
        std::cout<<"Device "<<dev<<" name: "<<deviceProp.name<<std::endl;
        std::cout<<"Computational capabilities: "<<deviceProp.major<<"."<<deviceProp.minor<<std::endl;
        std::cout<<"Max Global memory size: "<<deviceProp.totalGlobalMem<<std::endl;
        std::cout<<"Max Constant memory size: "<<deviceProp.totalConstMem<<std::endl;
        std::cout<<"Max Shared memory size per block: "<<deviceProp.sharedMemPerBlock<<std::endl;
        std::cout<<"Max threads per block: "<<deviceProp.maxThreadsPerBlock<<std::endl;
        std::cout<<"Max block dimensions: "<<deviceProp.maxThreadsDim[0]<<" x, "<<deviceProp.maxThreadsDim[1]<<" y, "<<deviceProp.maxThreadsDim[2]<<" z"<<std::endl;
        std::cout<<"Max grid dimensions: "<<deviceProp.maxGridSize[0]<<" x, "<<deviceProp.maxGridSize[1]<<" y, "<<deviceProp.maxGridSize[2]<<" z"<<std::endl;
        std::cout<<"Warp Size: "<<deviceProp.warpSize<<std::endl;
    }
}

#undef CUDA_CHECK
