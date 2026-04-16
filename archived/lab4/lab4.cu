#include <wb.h>

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "CUDA error: ", cudaGetErrorString(err));              \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      return -1;                                                          \
    }                                                                     \
  } while (0)

//@@ Define any useful program-wide constants here
#define MASK_WIDTH 3
#define TILE_X 8
#define TILE_Y 8
#define TILE_Z 8

//@@ Define constant memory for device kernel here
__constant__ float Mc[MASK_WIDTH][MASK_WIDTH][MASK_WIDTH];

__global__ void conv3d(float *input, float *output, const int z_size,
                       const int y_size, const int x_size) {
  //@@ Insert kernel code here
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int tz = threadIdx.z;

  // Output coordinates (for threads that compute)
  int x_o = blockIdx.x * TILE_X + tx;
  int y_o = blockIdx.y * TILE_Y + ty;
  int z_o = blockIdx.z * TILE_Z + tz;

  // Input coordinates (with halo: radius 1)
  int x_i = x_o - 1;
  int y_i = y_o - 1;
  int z_i = z_o - 1;

  __shared__ float tile[TILE_Z + MASK_WIDTH - 1][TILE_Y + MASK_WIDTH - 1]
                        [TILE_X + MASK_WIDTH - 1];

  // Load input tile into shared memory (boundary = 0)
  if (z_i >= 0 && z_i < z_size && y_i >= 0 && y_i < y_size && x_i >= 0 &&
      x_i < x_size) {
    tile[tz][ty][tx] =
        input[z_i * y_size * x_size + y_i * x_size + x_i];
  } else {
    tile[tz][ty][tx] = 0.0f;
  }
  __syncthreads();

  // Only threads in the output tile compute and write
  if (tz < TILE_Z && ty < TILE_Y && tx < TILE_X) {
    if (z_o < z_size && y_o < y_size && x_o < x_size) {
      float Pvalue = 0.0f;
      for (int dz = 0; dz < MASK_WIDTH; dz++) {
        for (int dy = 0; dy < MASK_WIDTH; dy++) {
          for (int dx = 0; dx < MASK_WIDTH; dx++) {
            Pvalue += tile[tz + dz][ty + dy][tx + dx] * Mc[dz][dy][dx];
          }
        }
      }
      output[z_o * y_size * x_size + y_o * x_size + x_o] = Pvalue;
    }
  }
}

int main(int argc, char *argv[]) {
  wbArg_t args;
  int z_size;
  int y_size;
  int x_size;
  int inputLength, kernelLength;
  float *hostInput;
  float *hostKernel;
  float *hostOutput;
  //@@ Initial deviceInput and deviceOutput here.
  float *deviceInput;
  float *deviceOutput;

  args = wbArg_read(argc, argv);

  // Import data
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &inputLength);
  hostKernel =
      (float *)wbImport(wbArg_getInputFile(args, 1), &kernelLength);
  hostOutput = (float *)malloc(inputLength * sizeof(float));

  // First three elements are the input dimensions
  z_size = hostInput[0];
  y_size = hostInput[1];
  x_size = hostInput[2];
  wbLog(TRACE, "The input size is ", z_size, "x", y_size, "x", x_size);
  assert(z_size * y_size * x_size == inputLength - 3);
  assert(kernelLength == 27);


  //@@ Allocate GPU memory here
  // Recall that inputLength is 3 elements longer than the input data
  // because the first  three elements were the dimensions
  int dataLength = z_size * y_size * x_size;
  wbCheck(cudaMalloc((void **)&deviceInput, dataLength * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceOutput, dataLength * sizeof(float)));

  //@@ Copy input and kernel to GPU here
  // Recall that the first three elements of hostInput are dimensions and
  // do
  // not need to be copied to the gpu
  wbCheck(cudaMemcpy(deviceInput, hostInput + 3, dataLength * sizeof(float),
                     cudaMemcpyHostToDevice));
  wbCheck(cudaMemcpyToSymbol(Mc, hostKernel,
                             MASK_WIDTH * MASK_WIDTH * MASK_WIDTH *
                                 sizeof(float)));

  //@@ Initialize grid and block dimensions here
  dim3 dimBlock(TILE_X + MASK_WIDTH - 1, TILE_Y + MASK_WIDTH - 1,
                TILE_Z + MASK_WIDTH - 1);
  dim3 dimGrid((x_size + TILE_X - 1) / TILE_X, (y_size + TILE_Y - 1) / TILE_Y,
               (z_size + TILE_Z - 1) / TILE_Z);

  //@@ Launch the GPU kernel here
  conv3d<<<dimGrid, dimBlock>>>(deviceInput, deviceOutput, z_size, y_size,
                                 x_size);

  cudaDeviceSynchronize();



  //@@ Copy the device memory back to the host here
  // Recall that the first three elements of the output are the dimensions
  // and should not be set here (they are set below)
  wbCheck(cudaMemcpy(hostOutput + 3, deviceOutput,
                     dataLength * sizeof(float), cudaMemcpyDeviceToHost));

  // Set the output dimensions for correctness checking
  hostOutput[0] = z_size;
  hostOutput[1] = y_size;
  hostOutput[2] = x_size;
  wbSolution(args, hostOutput, inputLength);

  //@@ Free device memory
  wbCheck(cudaFree(deviceInput));
  wbCheck(cudaFree(deviceOutput));


  // Free host memory
  free(hostInput);
  free(hostOutput);
  return 0;
}

