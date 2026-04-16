#pragma once

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err__ = (call);                                            \
        if (err__ != cudaSuccess) {                                            \
            std::fprintf(stderr, "CUDA error at %s:%d: %s\n",                  \
                         __FILE__, __LINE__, cudaGetErrorString(err__));       \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)
