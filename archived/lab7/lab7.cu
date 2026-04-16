// MP Scan
// Given a list (lst) of length n
// Output its prefix sum = {lst[0], lst[0] + lst[1], lst[0] + lst[1] + ...
// +
// lst[n-1]}

#include <wb.h>

#define BLOCK_SIZE 1024 //@@ You can change this

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

__global__ void scan(float *input, float *output, float *blockSums, int len) {
  //@@ Modify the body of this function to complete the functionality of
  //@@ the scan on the device
  //@@ You may need multiple kernel calls; write your kernels before this
  //@@ function and call them from the host
  __shared__ float temp[2 * BLOCK_SIZE];
  __shared__ float original[2 * BLOCK_SIZE];

  int tid = threadIdx.x;
  int base = 2 * blockIdx.x * blockDim.x;
  int first = base + tid;
  int second = first + blockDim.x;

  if (first < len) {
    temp[tid] = input[first];
    original[tid] = input[first];
  } else {
    temp[tid] = 0.0f;
    original[tid] = 0.0f;
  }

  if (second < len) {
    temp[tid + blockDim.x] = input[second];
    original[tid + blockDim.x] = input[second];
  } else {
    temp[tid + blockDim.x] = 0.0f;
    original[tid + blockDim.x] = 0.0f;
  }

  for (int stride = 1; stride <= blockDim.x; stride *= 2) {
    __syncthreads();
    int index = (tid + 1) * stride * 2 - 1;
    if (index < 2 * blockDim.x) {
      temp[index] += temp[index - stride];
    }
  }

  __syncthreads();
  if (tid == 0) {
    if (blockSums != nullptr) {
      blockSums[blockIdx.x] = temp[2 * blockDim.x - 1];
    }
    temp[2 * blockDim.x - 1] = 0.0f;
  }

  for (int stride = blockDim.x; stride > 0; stride /= 2) {
    __syncthreads();
    int index = (tid + 1) * stride * 2 - 1;
    if (index < 2 * blockDim.x) {
      float left = temp[index - stride];
      temp[index - stride] = temp[index];
      temp[index] += left;
    }
  }

  __syncthreads();

  if (first < len) {
    output[first] = temp[tid] + original[tid];
  }
  if (second < len) {
    output[second] = temp[tid + blockDim.x] + original[tid + blockDim.x];
  }
}

__global__ void addBlockSums(float *output, const float *scannedBlockSums, int len) {
  int tid = threadIdx.x;
  int base = 2 * blockIdx.x * blockDim.x;
  int first = base + tid;
  int second = first + blockDim.x;

  if (blockIdx.x == 0) {
    return;
  }

  float offset = scannedBlockSums[blockIdx.x - 1];
  if (first < len) {
    output[first] += offset;
  }
  if (second < len) {
    output[second] += offset;
  }
}

int main(int argc, char **argv) {
  wbArg_t args;
  float *hostInput;  // The input 1D list
  float *hostOutput; // The output list
  float *deviceInput;
  float *deviceOutput;
  int numElements; // number of elements in the list

  args = wbArg_read(argc, argv);

  // Import data and create memory on host
  // The number of input elements in the input is numElements
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &numElements);
  hostOutput = (float *)malloc(numElements * sizeof(float));


  // Allocate GPU memory.
  wbCheck(cudaMalloc((void **)&deviceInput, numElements * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceOutput, numElements * sizeof(float)));


  // Clear output memory.
  wbCheck(cudaMemset(deviceOutput, 0, numElements * sizeof(float)));

  // Copy input memory to the GPU.
  wbCheck(cudaMemcpy(deviceInput, hostInput, numElements * sizeof(float),
                     cudaMemcpyHostToDevice));

  //@@ Initialize the grid and block dimensions here
  dim3 blockDim(BLOCK_SIZE, 1, 1);
  dim3 gridDim((numElements + 2 * BLOCK_SIZE - 1) / (2 * BLOCK_SIZE), 1, 1);
  float *deviceBlockSums = nullptr;
  float *deviceScannedBlockSums = nullptr;


  //@@ Modify this to complete the functionality of the scan
  //@@ on the deivce
  if (gridDim.x > 1) {
    wbCheck(cudaMalloc((void **)&deviceBlockSums, gridDim.x * sizeof(float)));
    wbCheck(cudaMalloc((void **)&deviceScannedBlockSums, gridDim.x * sizeof(float)));
    wbCheck(cudaMemset(deviceBlockSums, 0, gridDim.x * sizeof(float)));
    wbCheck(cudaMemset(deviceScannedBlockSums, 0, gridDim.x * sizeof(float)));
  }

  scan<<<gridDim, blockDim>>>(deviceInput, deviceOutput, deviceBlockSums, numElements);
  wbCheck(cudaGetLastError());

  if (gridDim.x > 1) {
    scan<<<1, blockDim>>>(deviceBlockSums, deviceScannedBlockSums, nullptr, gridDim.x);
    wbCheck(cudaGetLastError());

    addBlockSums<<<gridDim, blockDim>>>(deviceOutput, deviceScannedBlockSums, numElements);
    wbCheck(cudaGetLastError());
  }

  wbCheck(cudaDeviceSynchronize());

  // Copying output memory to the CPU
  wbCheck(cudaMemcpy(hostOutput, deviceOutput, numElements * sizeof(float), cudaMemcpyDeviceToHost));


  //@@  Free GPU Memory
  wbCheck(cudaFree(deviceInput));
  wbCheck(cudaFree(deviceOutput));
  if (deviceBlockSums != nullptr) {
    wbCheck(cudaFree(deviceBlockSums));
  }
  if (deviceScannedBlockSums != nullptr) {
    wbCheck(cudaFree(deviceScannedBlockSums));
  }


  wbSolution(args, hostOutput, numElements);

  free(hostInput);
  free(hostOutput);

  return 0;
}
