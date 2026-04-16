// op_4: cuBLAS for matrix multiplication (3 pts).
// Base: unfused input unrolling (unroll-new-forward.cu)
// Change: replace the hand-written tiled matmul with cublasSgemm.
// Pipeline: unroll kernel → cublasSgemm → permute kernel.

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <cublas_v2.h>
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

#define CUBLAS_CHECK(call)                                                     \
    do {                                                                       \
        cublasStatus_t st__ = (call);                                          \
        if (st__ != CUBLAS_STATUS_SUCCESS) {                                   \
            std::cerr << "cuBLAS error at " << __FILE__ << ":" << __LINE__    \
                      << ": status " << st__ << std::endl;                    \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)

#define UNROLL_BLOCK_SIZE 256
#define PERMUTE_BLOCK_SIZE 256

static cublasHandle_t cublas_handle = nullptr;

__global__ void matrix_unrolling_kernel(const float *input, float *output,
                                        const int Batch, const int Channel,
                                        const int Height, const int Width,
                                        const int K)
{
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;

    #define in_4d(i3, i2, i1, i0) input[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]

    const size_t rows_unrolled = static_cast<size_t>(Channel) * K * K;
    const size_t cols_unrolled = static_cast<size_t>(Batch) * Height_out * Width_out;
    const size_t total_elements = rows_unrolled * cols_unrolled;
    const size_t linear_idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (linear_idx < total_elements) {
        const size_t row_unroll = linear_idx / cols_unrolled;
        const size_t col_unroll = linear_idx % cols_unrolled;

        const int c = static_cast<int>(row_unroll / (K * K));
        const int kernel_offset = static_cast<int>(row_unroll % (K * K));
        const int p = kernel_offset / K;
        const int q = kernel_offset % K;

        const int b = static_cast<int>(col_unroll / (Height_out * Width_out));
        const int image_offset = static_cast<int>(col_unroll % (Height_out * Width_out));
        const int h_out = image_offset / Width_out;
        const int w_out = image_offset % Width_out;

        output[row_unroll * cols_unrolled + col_unroll] = in_4d(b, c, h_out + p, w_out + q);
    }

    #undef in_4d
}

__global__ void matrix_permute_kernel(const float *input, float *output, int Map_out,
                                      int Batch, int image_size) {
    int b = blockIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x < image_size) {
        for (int m = 0; m < Map_out; m++) {
            output[b * Map_out * image_size + m * image_size + x] =
                    input[m * Batch * image_size + b * image_size + x];
        }
    }
}

__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    const size_t input_size = static_cast<size_t>(Batch) * Channel * Height * Width * sizeof(float);
    const size_t output_size = static_cast<size_t>(Batch) * Map_out * Height_out * Width_out * sizeof(float);
    const size_t mask_size = static_cast<size_t>(Map_out) * Channel * K * K * sizeof(float);

    (void)host_output;

    CUDA_CHECK(cudaMalloc((void **)device_input_ptr, input_size));
    CUDA_CHECK(cudaMalloc((void **)device_output_ptr, output_size));
    CUDA_CHECK(cudaMalloc((void **)device_mask_ptr, mask_size));

    CUDA_CHECK(cudaMemcpy(*device_input_ptr, host_input, input_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(*device_mask_ptr, host_mask, mask_size, cudaMemcpyHostToDevice));

    if (!cublas_handle) CUBLAS_CHECK(cublasCreate(&cublas_handle));
}

__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    const int Height_unrolled = Channel * K * K;
    const int Width_unrolled = Batch * Height_out * Width_out;

    float *unrolled_matrix;
    float *matmul_output;
    CUDA_CHECK(cudaMalloc((void**)&unrolled_matrix, static_cast<size_t>(Height_unrolled) * Width_unrolled * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&matmul_output, static_cast<size_t>(Batch) * Map_out * Height_out * Width_out * sizeof(float)));

    // 1. Unroll
    const size_t unroll_elements = static_cast<size_t>(Height_unrolled) * Width_unrolled;
    const size_t unroll_grid_x = (unroll_elements + UNROLL_BLOCK_SIZE - 1) / UNROLL_BLOCK_SIZE;
    matrix_unrolling_kernel<<<static_cast<unsigned int>(unroll_grid_x), UNROLL_BLOCK_SIZE>>>(
        device_input, unrolled_matrix, Batch, Channel, Height, Width, K
    );
    CUDA_CHECK(cudaGetLastError());

    // 2. cuBLAS GEMM: C = mask × unrolled
    // mask: Map_out × Height_unrolled (row-major)
    // unrolled: Height_unrolled × Width_unrolled (row-major)
    // cuBLAS uses column-major, so we compute C^T = B^T × A^T
    const float alpha = 1.0f, beta = 0.0f;
    CUBLAS_CHECK(cublasSgemm(cublas_handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        Width_unrolled, Map_out, Height_unrolled,
        &alpha,
        unrolled_matrix, Width_unrolled,
        device_mask, Height_unrolled,
        &beta,
        matmul_output, Width_unrolled
    ));

    // 3. Permute
    const int out_image_size = Height_out * Width_out;
    dim3 permute_grid((out_image_size - 1) / PERMUTE_BLOCK_SIZE + 1, Batch, 1);
    matrix_permute_kernel<<<permute_grid, PERMUTE_BLOCK_SIZE>>>(
        matmul_output, device_output, Map_out, Batch, out_image_size
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaFree(matmul_output));
    CUDA_CHECK(cudaFree(unrolled_matrix));
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

    if (cublas_handle) { cublasDestroy(cublas_handle); cublas_handle = nullptr; }
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
#undef CUBLAS_CHECK
