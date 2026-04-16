#pragma once

#include <cuda_runtime.h>
#include "cuda_check.h"

// RAII-ish CUDA event timer. Usage:
//   CudaTimer t;
//   t.start();
//   kernel<<<...>>>(...);
//   float ms = t.stop();   // returns elapsed milliseconds (host-blocking on stop)
class CudaTimer {
public:
    CudaTimer() {
        CUDA_CHECK(cudaEventCreate(&start_));
        CUDA_CHECK(cudaEventCreate(&stop_));
    }
    ~CudaTimer() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }

    void start(cudaStream_t stream = 0) {
        CUDA_CHECK(cudaEventRecord(start_, stream));
    }

    // Records stop, blocks host until the event completes, returns elapsed ms.
    float stop(cudaStream_t stream = 0) {
        CUDA_CHECK(cudaEventRecord(stop_, stream));
        CUDA_CHECK(cudaEventSynchronize(stop_));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_));
        return ms;
    }

private:
    cudaEvent_t start_;
    cudaEvent_t stop_;
};
