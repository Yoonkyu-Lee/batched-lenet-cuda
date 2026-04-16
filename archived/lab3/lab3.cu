#include <wb.h>

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

#define TILE_WIDTH 16

// Compute C = A * B
__global__ void matrixMultiplyShared(float *A, float *B, float *C,
                                     int numARows, int numAColumns,
                                     int numBRows, int numBColumns,
                                     int numCRows, int numCColumns) {
  //@@ Insert code to implement matrix multiplication here
  //@@ You have to use shared memory for this MP
  __shared__ float A_s[TILE_WIDTH][TILE_WIDTH];
  __shared__ float B_s[TILE_WIDTH][TILE_WIDTH];

  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  float sum = 0.0f;

  int numTiles = (numAColumns - 1) / TILE_WIDTH + 1;
  for (int tile = 0; tile < numTiles; ++tile) {
    // Load A tile (row, tile*TILE_WIDTH + threadIdx.x)
    if (row < numARows && tile * TILE_WIDTH + threadIdx.x < numAColumns)
      A_s[threadIdx.y][threadIdx.x] =
          A[row * numAColumns + tile * TILE_WIDTH + threadIdx.x];
    else
      A_s[threadIdx.y][threadIdx.x] = 0.0f;

    // Load B tile (tile*TILE_WIDTH + threadIdx.y, col)
    if (tile * TILE_WIDTH + threadIdx.y < numBRows && col < numBColumns)
      B_s[threadIdx.y][threadIdx.x] =
          B[(tile * TILE_WIDTH + threadIdx.y) * numBColumns + col];
    else
      B_s[threadIdx.y][threadIdx.x] = 0.0f;

    __syncthreads();

    for (int k = 0; k < TILE_WIDTH; ++k)
      sum += A_s[threadIdx.y][k] * B_s[k][threadIdx.x];

    __syncthreads();
  }

  if (row < numCRows && col < numCColumns)
    C[row * numCColumns + col] = sum;
}

int main(int argc, char **argv) {
  wbArg_t args;
  float *hostA; // The A matrix
  float *hostB; // The B matrix
  float *hostC; // The output C matrix

  int numARows;    // number of rows in the matrix A
  int numAColumns; // number of columns in the matrix A
  int numBRows;    // number of rows in the matrix B
  int numBColumns; // number of columns in the matrix B
  int numCRows;    // number of rows in the matrix C (you have to set this)
  int numCColumns; // number of columns in the matrix C (you have to set
                   // this)

  args = wbArg_read(argc, argv);

  //@@ Importing data and creating memory on host
  hostA = (float *)wbImport(wbArg_getInputFile(args, 0), &numARows,
                            &numAColumns);
  hostB = (float *)wbImport(wbArg_getInputFile(args, 1), &numBRows,
                            &numBColumns);
  //@@ Set numCRows and numCColumns
  numCRows = numARows;
  numCColumns = numBColumns;

  //@@ Allocate the hostC matrix
  hostC = (float *)malloc(numCRows * numCColumns * sizeof(float));

  //@@ Allocate GPU memory here
  float *deviceA;
  float *deviceB;
  float *deviceC;
  wbCheck(cudaMalloc((void **)&deviceA, numARows * numAColumns * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceB, numBRows * numBColumns * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceC, numCRows * numCColumns * sizeof(float)));

  //@@ Copy memory to the GPU here
  wbCheck(cudaMemcpy(deviceA, hostA, numARows * numAColumns * sizeof(float),
                    cudaMemcpyHostToDevice));
  wbCheck(cudaMemcpy(deviceB, hostB, numBRows * numBColumns * sizeof(float),
                    cudaMemcpyHostToDevice));

  //@@ Initialize the grid and block dimensions here
  dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
  dim3 dimGrid((numCColumns + dimBlock.x - 1) / dimBlock.x,
               (numCRows + dimBlock.y - 1) / dimBlock.y, 1);

  //@@ Launch the GPU Kernel here
  matrixMultiplyShared<<<dimGrid, dimBlock>>>(
      deviceA, deviceB, deviceC, numARows, numAColumns, numBRows, numBColumns,
      numCRows, numCColumns);

  cudaDeviceSynchronize();

  //@@ Copy the GPU memory back to the CPU here
  wbCheck(cudaMemcpy(hostC, deviceC, numCRows * numCColumns * sizeof(float), cudaMemcpyDeviceToHost));

  //@@ Free the GPU memory here
  wbCheck(cudaFree(deviceA));
  wbCheck(cudaFree(deviceB));
  wbCheck(cudaFree(deviceC));

  wbSolution(args, hostC, numCRows, numCColumns);

  free(hostA);
  free(hostB);

  //@@ Free the hostC matrix
  free(hostC);

  return 0;
}
