// req_0: CUDA Streams — overlap data transfer with kernel execution.
// Base: unfused input unrolling (unroll-new-forward.cu)
// Strategy: split batch into segments, use 3 streams for continuous pipelining.
// Per segment: cudaMemcpyAsync H->D input -> unroll -> matmul -> permute -> cudaMemcpyAsync D->H output.
//
// NOTE: per project spec, the *final* m3-forward.cu must be single-stream.
// This file is the individual-evaluation version for the req_0 report section.

#include <cmath>
#include <cstdlib>
#include <algorithm>
#include <iostream>
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

#define UNROLL_BLOCK_SIZE 256
#define PERMUTE_BLOCK_SIZE 256
#define MATMUL_TILE_WIDTH 16
#define NUM_STREAMS 3
#define SEG_SIZE 1000

static const float *g_host_input  = nullptr;
static float       *g_host_output = nullptr;
static bool         g_pinned_in   = false;
static bool         g_pinned_out  = false;

__global__ void matrix_unrolling_kernel(const float *input, float *output,
                                        int Batch, int Channel, int Height, int Width, int K)
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

__global__ void matmul_tiled(const float *A, const float *B, float *C,
                             int numARows, int numAColumns,
                             int numBRows, int numBColumns,
                             int numCRows, int numCColumns)
{
    __shared__ float subTileA[MATMUL_TILE_WIDTH][MATMUL_TILE_WIDTH];
    __shared__ float subTileB[MATMUL_TILE_WIDTH][MATMUL_TILE_WIDTH];

    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * MATMUL_TILE_WIDTH + ty;
    int col = blockIdx.x * MATMUL_TILE_WIDTH + tx;
    float acc = 0.0f;

    for (int m = 0; m < (numAColumns + MATMUL_TILE_WIDTH - 1) / MATMUL_TILE_WIDTH; m++) {
        subTileA[ty][tx] = (row < numARows && m * MATMUL_TILE_WIDTH + tx < numAColumns)
                           ? A[row * numAColumns + m * MATMUL_TILE_WIDTH + tx] : 0.0f;
        subTileB[ty][tx] = (col < numBColumns && m * MATMUL_TILE_WIDTH + ty < numBRows)
                           ? B[(m * MATMUL_TILE_WIDTH + ty) * numBColumns + col] : 0.0f;
        __syncthreads();

        if (row < numCRows && col < numCColumns) {
            for (int k = 0; k < MATMUL_TILE_WIDTH; k++)
                acc += subTileA[ty][k] * subTileB[k][tx];
        }
        __syncthreads();
    }
    if (row < numCRows && col < numCColumns)
        C[row * numCColumns + col] = acc;
}

__global__ void matrix_permute_kernel(const float *input, float *output,
                                      int Map_out, int Batch, int image_size) {
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
    const int Width_out  = Width  - K + 1;
    const size_t input_size  = static_cast<size_t>(Batch) * Channel * Height * Width * sizeof(float);
    const size_t output_size = static_cast<size_t>(Batch) * Map_out * Height_out * Width_out * sizeof(float);
    const size_t mask_size   = static_cast<size_t>(Map_out) * Channel * K * K * sizeof(float);

    CUDA_CHECK(cudaMalloc((void **)device_input_ptr,  input_size));
    CUDA_CHECK(cudaMalloc((void **)device_output_ptr, output_size));
    CUDA_CHECK(cudaMalloc((void **)device_mask_ptr,   mask_size));

    // Mask is small and one-shot — copy synchronously.
    CUDA_CHECK(cudaMemcpy(*device_mask_ptr, host_mask, mask_size, cudaMemcpyHostToDevice));

    // Cache host pointers so conv_forward_gpu can do the streamed copies itself.
    g_host_input  = host_input;
    g_host_output = const_cast<float*>(host_output);

    // Pin host buffers so cudaMemcpyAsync is actually asynchronous.
    cudaError_t er = cudaHostRegister((void*)host_input, input_size, cudaHostRegisterDefault);
    g_pinned_in = (er == cudaSuccess);
    if (!g_pinned_in) cudaGetLastError();  // swallow "already registered" or similar

    er = cudaHostRegister((void*)host_output, output_size, cudaHostRegisterDefault);
    g_pinned_out = (er == cudaSuccess);
    if (!g_pinned_out) cudaGetLastError();
}

__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    const int Height_out = Height - K + 1;
    const int Width_out  = Width  - K + 1;
    const int image_size = Height_out * Width_out;
    const int rows_unrolled = Channel * K * K;
    const int segments = (Batch + SEG_SIZE - 1) / SEG_SIZE;

    cudaStream_t streams[NUM_STREAMS];
    for (int s = 0; s < NUM_STREAMS; s++) CUDA_CHECK(cudaStreamCreate(&streams[s]));

    // One pair of scratch buffers per stream (sized for one full segment).
    float *unrolled[NUM_STREAMS];
    float *mm_out  [NUM_STREAMS];
    for (int s = 0; s < NUM_STREAMS; s++) {
        CUDA_CHECK(cudaMalloc(&unrolled[s],
            static_cast<size_t>(rows_unrolled) * SEG_SIZE * image_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&mm_out[s],
            static_cast<size_t>(SEG_SIZE) * Map_out * image_size * sizeof(float)));
    }

    const size_t in_per_image  = static_cast<size_t>(Channel) * Height * Width;
    const size_t out_per_image = static_cast<size_t>(Map_out) * image_size;

    for (int seg = 0; seg < segments; seg++) {
        int sidx = seg % NUM_STREAMS;
        int b0 = seg * SEG_SIZE;
        int bsz = std::min(SEG_SIZE, Batch - b0);

        const float *h_in_seg = g_host_input + b0 * in_per_image;
        float       *d_in_seg = const_cast<float*>(device_input) + b0 * in_per_image;
        CUDA_CHECK(cudaMemcpyAsync(d_in_seg, h_in_seg,
            static_cast<size_t>(bsz) * in_per_image * sizeof(float),
            cudaMemcpyHostToDevice, streams[sidx]));

        size_t unroll_elems = static_cast<size_t>(rows_unrolled) * bsz * image_size;
        size_t unroll_grid  = (unroll_elems + UNROLL_BLOCK_SIZE - 1) / UNROLL_BLOCK_SIZE;
        matrix_unrolling_kernel<<<static_cast<unsigned int>(unroll_grid), UNROLL_BLOCK_SIZE, 0, streams[sidx]>>>(
            d_in_seg, unrolled[sidx], bsz, Channel, Height, Width, K);

        int mm_cols = bsz * image_size;
        dim3 mm_grid((mm_cols + MATMUL_TILE_WIDTH - 1) / MATMUL_TILE_WIDTH,
                     (Map_out + MATMUL_TILE_WIDTH - 1) / MATMUL_TILE_WIDTH, 1);
        dim3 mm_block(MATMUL_TILE_WIDTH, MATMUL_TILE_WIDTH, 1);
        matmul_tiled<<<mm_grid, mm_block, 0, streams[sidx]>>>(
            device_mask, unrolled[sidx], mm_out[sidx],
            Map_out, rows_unrolled, rows_unrolled, mm_cols, Map_out, mm_cols);

        float *d_out_seg = device_output + b0 * out_per_image;
        dim3 perm_grid((image_size + PERMUTE_BLOCK_SIZE - 1) / PERMUTE_BLOCK_SIZE, bsz, 1);
        matrix_permute_kernel<<<perm_grid, PERMUTE_BLOCK_SIZE, 0, streams[sidx]>>>(
            mm_out[sidx], d_out_seg, Map_out, bsz, image_size);

        float *h_out_seg = g_host_output + b0 * out_per_image;
        CUDA_CHECK(cudaMemcpyAsync(h_out_seg, d_out_seg,
            static_cast<size_t>(bsz) * out_per_image * sizeof(float),
            cudaMemcpyDeviceToHost, streams[sidx]));
    }

    for (int s = 0; s < NUM_STREAMS; s++) CUDA_CHECK(cudaStreamSynchronize(streams[s]));

    for (int s = 0; s < NUM_STREAMS; s++) {
        CUDA_CHECK(cudaFree(unrolled[s]));
        CUDA_CHECK(cudaFree(mm_out[s]));
        CUDA_CHECK(cudaStreamDestroy(streams[s]));
    }
}

__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // Output was already transferred back inside conv_forward_gpu via streams.
    (void)host_output;
    (void)Batch; (void)Map_out; (void)Channel; (void)Height; (void)Width; (void)K;

    if (g_pinned_in)  { cudaHostUnregister((void*)g_host_input);  g_pinned_in  = false; }
    if (g_pinned_out) { cudaHostUnregister((void*)g_host_output); g_pinned_out = false; }
    g_host_input = nullptr;
    g_host_output = nullptr;

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
