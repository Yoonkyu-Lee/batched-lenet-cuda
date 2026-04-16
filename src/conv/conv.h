// Common interface for all convolution variants.
// Each variant (baseline.cu, fused.cu, tensor_cores.cu, register_tiled.cu)
// implements both prepare() and forward(). The Makefile picks exactly one .cu
// file per binary, so there is no link-time conflict.

#pragma once

namespace lenet_conv {

// One-time per-layer setup. Copies the mask to wherever the variant needs it
// (global device memory or __constant__). Call once per conv layer; not part
// of the timed region.
void prepare(const float* h_mask,
             int Map_out, int Channel, int K);

// Frees any per-layer state allocated by prepare(). Call before reusing the
// same handle for a different layer (different Map_out/Channel/K).
void release();

// Runs the convolution. The caller is expected to have populated d_input and
// to have allocated d_output. Time this call externally with cudaEvent.
void forward(const float* d_input, float* d_output,
             int Batch, int Map_out, int Channel,
             int Height, int Width, int K);

// Short human-readable name of the variant (for logging / bench tables).
const char* name();

}  // namespace lenet_conv
