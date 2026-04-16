// LAB 1
#include <wb.h>

__global__ void vecAdd(float *in1, float *in2, float *out, int len) {
  //@@ Insert code to implement vector addition here
  // L02 P8: Block + Thread 인덱싱 → 이 스레드가 담당하는 원소 번호
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  // L02 P15: 범위 체크 후 C[i] = A[i] + B[i]
  if (i < len) {
    out[i] = in1[i] + in2[i];
  }
}

int main(int argc, char **argv) {
  wbArg_t args;
  int inputLength;
  float *hostInput1;
  float *hostInput2;
  float *hostOutput;

  args = wbArg_read(argc, argv);
  //@@ Importing data and creating memory on host
  hostInput1 =
      (float *)wbImport(wbArg_getInputFile(args, 0), &inputLength);
  hostInput2 =
      (float *)wbImport(wbArg_getInputFile(args, 1), &inputLength);
  hostOutput = (float *)malloc(inputLength * sizeof(float));

  wbLog(TRACE, "The input length is ", inputLength);

  //@@ Allocate GPU memory here
  // L02 P12–13: 1) GPU 메모리 할당 (cudaMalloc)
  float *deviceInput1;
  float *deviceInput2;
  float *deviceOutput;
  cudaMalloc((void **)&deviceInput1, inputLength * sizeof(float));
  cudaMalloc((void **)&deviceInput2, inputLength * sizeof(float));
  cudaMalloc((void **)&deviceOutput, inputLength * sizeof(float));

  //@@ Copy memory to the GPU here
  // cudaMemcpy(dst, src, size, cudaMemcpyKind);
  // L02 P12–13: 2) Host → Device 복사 (cudaMemcpyHostToDevice)
  cudaMemcpy(deviceInput1, hostInput1, inputLength * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(deviceInput2, hostInput2, inputLength * sizeof(float), cudaMemcpyHostToDevice);

  //@@ Initialize the grid and block dimensions here
  // L02 P14: numBlocks = (n + blockSize - 1) / blockSize (올림)
  int blockSize = 256;
  int gridSize = (inputLength + blockSize - 1) / blockSize;

  //@@ Launch the GPU Kernel here to perform CUDA computation
  // L02 P7, P14: vecAddKernel<<<numBlocks, numThreadsPerBlock>>>(A_d, B_d, C_d, n);
  vecAdd<<<gridSize, blockSize>>>(deviceInput1, deviceInput2, deviceOutput, inputLength);

  cudaDeviceSynchronize();
  //@@ Copy the GPU memory back to the CPU here
  // cudaMemcpy(dst, src, size, cudaMemcpyKind);
  // L02 P12–13: 4) Device → Host 복사 (cudaMemcpyDeviceToHost)
  cudaMemcpy(hostOutput, deviceOutput, inputLength * sizeof(float), cudaMemcpyDeviceToHost);

  //@@ Free the GPU memory here
  // L02 P12–13: 5) GPU 메모리 해제 (cudaFree)
  cudaFree(deviceInput1);
  cudaFree(deviceInput2);
  cudaFree(deviceOutput);


  wbSolution(args, hostOutput, inputLength);

  free(hostInput1);
  free(hostInput2);
  free(hostOutput);

  return 0;
}