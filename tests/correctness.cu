// Correctness test: compare a variant's GPU output against a CPU reference
// (simple 6-loop direct convolution) on a small batch.
//
// Runs at B=5 on the LeNet-5 layer dimensions with random data. Prints diff
// stats and a PASS/FAIL verdict. Tolerance is loose enough to accept TF32
// rounding drift (1e-2) but tight enough to catch real indexing bugs.
//
// Exit code 0 on PASS, 1 on FAIL.

#include "../src/conv/conv.h"
#include "../src/utils/cuda_check.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

namespace {

struct LayerSpec {
    const char* name;
    int Channel;
    int Map_out;
    int Height;
    int Width;
    int K;
};

constexpr LayerSpec kLayers[] = {
    {"Conv1", 1, 4,  86, 86, 7},
    {"Conv2", 4, 16, 40, 40, 7},
};

constexpr int kTestBatch = 5;           // keep CPU reference tractable

// Absolute tolerance. The TF32 variants drop the input mantissa to 10 bits
// before multiplying (hardware behavior), so per-element error from a single
// MMA is on the order of 2^-10 ≈ 1e-3. Summing ~200 products in Conv2
// amplifies this into the 1e-2 range. 5e-2 accepts that drift while still
// catching real indexing / accumulation bugs (which present as 10x+ diffs).
constexpr float kAbsTol  = 5e-2f;

void fill_random(std::vector<float>& v, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (float& x : v) x = dist(rng);
}

// Direct CPU reference: y[b][m][h][w] = sum_{c,p,q} x[b][c][h+p][w+q] * k[m][c][p][q].
void cpu_conv(
    const std::vector<float>& input,
    const std::vector<float>& mask,
    std::vector<float>& output,
    int Batch, int Map_out, int Channel,
    int Height, int Width, int K)
{
    const int H_out = Height - K + 1;
    const int W_out = Width  - K + 1;
    output.assign(static_cast<size_t>(Batch) * Map_out * H_out * W_out, 0.0f);

    auto in_idx = [&](int b, int c, int h, int w) {
        return ((b * Channel + c) * Height + h) * Width + w;
    };
    auto mk_idx = [&](int m, int c, int p, int q) {
        return ((m * Channel + c) * K + p) * K + q;
    };
    auto out_idx = [&](int b, int m, int h, int w) {
        return ((b * Map_out + m) * H_out + h) * W_out + w;
    };

    for (int b = 0; b < Batch; b++)
    for (int m = 0; m < Map_out; m++)
    for (int h = 0; h < H_out; h++)
    for (int w = 0; w < W_out; w++) {
        float acc = 0.0f;
        for (int c = 0; c < Channel; c++)
        for (int p = 0; p < K; p++)
        for (int q = 0; q < K; q++)
            acc += input[in_idx(b, c, h + p, w + q)] * mask[mk_idx(m, c, p, q)];
        output[out_idx(b, m, h, w)] = acc;
    }
}

struct DiffStats {
    float  max_abs;
    double mean_abs;
    size_t num_over_tol;
    size_t total;
};

DiffStats compare(const std::vector<float>& a, const std::vector<float>& b, float tol) {
    DiffStats s{0.0f, 0.0, 0, a.size()};
    double sum = 0.0;
    for (size_t i = 0; i < a.size(); i++) {
        float d = std::fabs(a[i] - b[i]);
        if (d > s.max_abs) s.max_abs = d;
        sum += d;
        if (d > tol) s.num_over_tol++;
    }
    s.mean_abs = sum / a.size();
    return s;
}

}  // namespace

int main() {
    std::printf("== Correctness test: %s ==\n", lenet_conv::name());
    std::printf("Batch=%d, tolerance=%.1e\n\n", kTestBatch, kAbsTol);

    int failed_layers = 0;

    for (const LayerSpec& L : kLayers) {
        const int H_out = L.Height - L.K + 1;
        const int W_out = L.Width  - L.K + 1;

        const size_t in_elems   = static_cast<size_t>(kTestBatch) * L.Channel * L.Height * L.Width;
        const size_t out_elems  = static_cast<size_t>(kTestBatch) * L.Map_out  * H_out * W_out;
        const size_t mask_elems = static_cast<size_t>(L.Map_out)  * L.Channel * L.K * L.K;

        std::vector<float> h_input(in_elems), h_mask(mask_elems);
        fill_random(h_input, /*seed=*/0x1234 ^ L.Channel);
        fill_random(h_mask,  /*seed=*/0x5678 ^ L.Map_out);

        // CPU reference.
        std::vector<float> cpu_out;
        cpu_conv(h_input, h_mask, cpu_out,
                 kTestBatch, L.Map_out, L.Channel, L.Height, L.Width, L.K);

        // GPU run.
        float *d_input = nullptr, *d_output = nullptr;
        CUDA_CHECK(cudaMalloc(&d_input,  in_elems  * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_output, out_elems * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_input, h_input.data(),
                              in_elems * sizeof(float), cudaMemcpyHostToDevice));

        lenet_conv::prepare(h_mask.data(), L.Map_out, L.Channel, L.K);
        lenet_conv::forward(d_input, d_output,
                            kTestBatch, L.Map_out, L.Channel, L.Height, L.Width, L.K);

        std::vector<float> gpu_out(out_elems);
        CUDA_CHECK(cudaMemcpy(gpu_out.data(), d_output,
                              out_elems * sizeof(float), cudaMemcpyDeviceToHost));

        lenet_conv::release();
        CUDA_CHECK(cudaFree(d_input));
        CUDA_CHECK(cudaFree(d_output));

        DiffStats s = compare(cpu_out, gpu_out, kAbsTol);
        const bool pass = s.num_over_tol == 0;

        std::printf("%s (in=%dx%dx%d, out=%dx%dx%d, K=%d)\n",
                    L.name, L.Channel, L.Height, L.Width,
                    L.Map_out, H_out, W_out, L.K);
        std::printf("  max|diff| = %.3e\n", s.max_abs);
        std::printf("  mean|diff| = %.3e\n", s.mean_abs);
        std::printf("  elements over tol: %zu / %zu\n", s.num_over_tol, s.total);
        std::printf("  %s\n\n", pass ? "PASS" : "FAIL");

        if (!pass) failed_layers++;
    }

    std::printf("Overall: %s\n", failed_layers == 0 ? "PASS" : "FAIL");
    return failed_layers == 0 ? 0 : 1;
}
