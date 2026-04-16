// Minimal driver for the LeNet-5 conv variants.
//
// Generates synthetic input + mask, runs the selected variant on Conv1 and
// Conv2 of the LeNet-5 layout, and reports per-layer Op Time (median over
// several measured rounds after a few warmup rounds). No external dataset
// required.
//
// Each variant lives in its own translation unit; the Makefile picks one
// per binary, so `lenet_conv::forward()` resolves at link time.
//
// Usage:
//   ./bin/<variant> [batch=10000] [warmup=5] [measured=10]
//
// Example:
//   ./bin/register_tiled 10000

#include "conv/conv.h"
#include "utils/cuda_check.h"
#include "utils/timing.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace {

// LeNet-5 variant from the original project. Two conv layers with K=7.
struct LayerSpec {
    const char* name;
    int Channel;
    int Map_out;
    int Height;
    int Width;
    int K;
};

constexpr LayerSpec kLayers[] = {
    {"Conv1 (1->4 channels)",  /*Channel=*/1, /*Map_out=*/4,  /*H=*/86, /*W=*/86, /*K=*/7},
    {"Conv2 (4->16 channels)", /*Channel=*/4, /*Map_out=*/16, /*H=*/40, /*W=*/40, /*K=*/7},
};

void fill_random(std::vector<float>& v, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (float& x : v) x = dist(rng);
}

float median(std::vector<float> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

}  // namespace

int main(int argc, char** argv) {
    int Batch    = (argc > 1) ? std::atoi(argv[1]) : 10000;
    int warmup   = (argc > 2) ? std::atoi(argv[2]) : 5;
    int measured = (argc > 3) ? std::atoi(argv[3]) : 10;

    if (Batch <= 0 || warmup < 0 || measured <= 0) {
        std::fprintf(stderr, "Usage: %s [batch>0] [warmup>=0] [measured>0]\n", argv[0]);
        return 1;
    }

    std::printf("== %s ==\n", lenet_conv::name());
    std::printf("Batch=%d, warmup=%d, measured=%d\n\n", Batch, warmup, measured);

    float total_op_time_ms = 0.0f;

    for (const LayerSpec& L : kLayers) {
        const int Height_out = L.Height - L.K + 1;
        const int Width_out  = L.Width  - L.K + 1;

        const size_t in_elems   = static_cast<size_t>(Batch) * L.Channel * L.Height * L.Width;
        const size_t out_elems  = static_cast<size_t>(Batch) * L.Map_out  * Height_out * Width_out;
        const size_t mask_elems = static_cast<size_t>(L.Map_out) * L.Channel * L.K * L.K;

        // Host-side synthetic data.
        std::vector<float> h_input(in_elems);
        std::vector<float> h_mask(mask_elems);
        fill_random(h_input, /*seed=*/0xC0FFEE ^ L.Channel);
        fill_random(h_mask,  /*seed=*/0xBEEF   ^ L.Map_out);

        // Device buffers.
        float *d_input = nullptr, *d_output = nullptr;
        CUDA_CHECK(cudaMalloc(&d_input,  in_elems  * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_output, out_elems * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_input, h_input.data(),
                              in_elems * sizeof(float), cudaMemcpyHostToDevice));

        // Per-layer setup (mask copy etc.) — NOT timed.
        lenet_conv::prepare(h_mask.data(), L.Map_out, L.Channel, L.K);

        // Warmup.
        for (int i = 0; i < warmup; i++) {
            lenet_conv::forward(d_input, d_output,
                                Batch, L.Map_out, L.Channel,
                                L.Height, L.Width, L.K);
        }

        // Measured rounds. Each forward() ends with a device sync internally.
        std::vector<float> times_ms;
        times_ms.reserve(measured);
        for (int i = 0; i < measured; i++) {
            CudaTimer t;
            t.start();
            lenet_conv::forward(d_input, d_output,
                                Batch, L.Map_out, L.Channel,
                                L.Height, L.Width, L.K);
            times_ms.push_back(t.stop());
        }

        const float med_ms = median(times_ms);
        total_op_time_ms  += med_ms;

        const double imgs_per_sec = (Batch * 1000.0) / med_ms;
        std::printf("%s  in=%dx%dx%d  out=%dx%dx%d  K=%d\n",
                    L.name, L.Channel, L.Height, L.Width,
                    L.Map_out, Height_out, Width_out, L.K);
        std::printf("  Op Time:    %8.3f ms (median of %d)\n", med_ms, measured);
        std::printf("  Throughput: %8.0f images/sec\n\n", imgs_per_sec);

        lenet_conv::release();
        CUDA_CHECK(cudaFree(d_input));
        CUDA_CHECK(cudaFree(d_output));
    }

    std::printf("Total Op Time: %.3f ms\n", total_op_time_ms);
    return 0;
}
