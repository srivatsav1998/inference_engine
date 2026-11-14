// Kernel & memory-bandwidth microbenchmarks.
//
// Measures, at real GPT-2 124M shapes:
//   1. Memory read bandwidth (STREAM-style sum) — the ceiling for weight-streaming decode.
//   2. Engine matMul2D_out vs. Apple Accelerate cblas_sgemm (reference ceiling, uses AMX).
//   3. Engine pointwise / row-wise kernels (LayerNorm, GELU, Softmax, Add).
//
// Methodology notes:
//   - Weight matrices are rotated across enough copies to exceed ~256 MB so that each timed
//     call streams B from DRAM, as in real decode (12 layers x distinct weights), instead of
//     hitting L2/SLC and overstating bandwidth.
//   - Each case: warmup, then repeat until >= MIN_TIME_S and >= MIN_REPS; report the median.
//
// Usage: kernel_bench [--quick] [--out results.json]

#include "tensorView.hpp"
#include "mathOps.hpp"

#include <algorithm>
#include <arm_neon.h>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#ifdef HAVE_ACCELERATE
#include <Accelerate/Accelerate.h>
#endif

namespace
{

using Clock = std::chrono::steady_clock;

bool gQuick = false;
double minTimeS() { return gQuick ? 0.15 : 0.5; }
int minReps() { return gQuick ? 3 : 7; }

template <typename F>
double medianSeconds(F &&fn, int warmup = 2)
{
    for (int i = 0; i < warmup; ++i)
        fn();

    std::vector<double> samples;
    double total = 0.0;
    while (total < minTimeS() || static_cast<int>(samples.size()) < minReps())
    {
        auto t0 = Clock::now();
        fn();
        double dt = std::chrono::duration<double>(Clock::now() - t0).count();
        samples.push_back(dt);
        total += dt;
        if (samples.size() >= 2000)
            break;
    }
    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2];
}

std::vector<float> randomBuffer(size_t n, uint32_t seed)
{
    std::vector<float> v(n);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto &x : v)
        x = dist(rng);
    return v;
}

// ---------------------------------------------------------------------------
// 1. Memory read bandwidth
// ---------------------------------------------------------------------------
float neonSum(const float *p, size_t n)
{
    float32x4_t a0 = vdupq_n_f32(0), a1 = vdupq_n_f32(0), a2 = vdupq_n_f32(0), a3 = vdupq_n_f32(0);
    size_t i = 0;
    for (; i + 16 <= n; i += 16)
    {
        a0 = vaddq_f32(a0, vld1q_f32(p + i));
        a1 = vaddq_f32(a1, vld1q_f32(p + i + 4));
        a2 = vaddq_f32(a2, vld1q_f32(p + i + 8));
        a3 = vaddq_f32(a3, vld1q_f32(p + i + 12));
    }
    float s = vaddvq_f32(vaddq_f32(vaddq_f32(a0, a1), vaddq_f32(a2, a3)));
    for (; i < n; ++i)
        s += p[i];
    return s;
}

volatile float gSink = 0.0f;

double readBandwidthGBs(const std::vector<float> &buf, unsigned threads)
{
    const size_t n = buf.size();
    auto run = [&]
    {
        std::vector<std::thread> pool;
        std::vector<float> partial(threads, 0.0f);
        size_t chunk = (n + threads - 1) / threads;
        for (unsigned t = 0; t < threads; ++t)
        {
            pool.emplace_back([&, t]
                              {
                size_t s = t * chunk, e = std::min(n, s + chunk);
                if (s < e) partial[t] = neonSum(buf.data() + s, e - s); });
        }
        for (auto &th : pool)
            th.join();
        float total = 0;
        for (float p : partial)
            total += p;
        gSink = total;
    };
    double t = medianSeconds(run, 1);
    return (n * sizeof(float)) / t / 1e9;
}

// ---------------------------------------------------------------------------
// 2. MatMul: engine vs Accelerate
// ---------------------------------------------------------------------------
struct MatMulCase
{
    const char *name;
    size_t M, K, N;
    bool transposeB; // B stored as [N, K] (lm_head reuses wte [50257, 768])
};

struct MatMulResult
{
    std::string name;
    size_t M, K, N;
    bool transposeB;
    double engineMs, engineGflops, engineWeightGBs;
    double accelMs = NAN, accelGflops = NAN, accelWeightGBs = NAN;
    double maxAbsDiff = NAN;
};

MatMulResult benchMatMul(const MatMulCase &c)
{
    const size_t bElems = c.K * c.N;
    const size_t bBytes = bElems * sizeof(float);
    const size_t targetBytes = gQuick ? (128ull << 20) : (256ull << 20);
    const size_t copies = std::max<size_t>(2, (targetBytes + bBytes - 1) / bBytes);

    auto A = randomBuffer(c.M * c.K, 1);
    std::vector<std::vector<float>> Bs;
    Bs.reserve(copies);
    for (size_t i = 0; i < copies; ++i)
        Bs.push_back(randomBuffer(bElems, 100 + static_cast<uint32_t>(i)));
    std::vector<float> O(c.M * c.N), Oref(c.M * c.N);

    std::vector<size_t> bShape = c.transposeB ? std::vector<size_t>{c.N, c.K} : std::vector<size_t>{c.K, c.N};
    TensorView tA(A.data(), {c.M, c.K}, 0);
    TensorView tO(O.data(), {c.M, c.N}, 0);
    std::vector<TensorView> tBs;
    for (auto &b : Bs)
        tBs.emplace_back(b.data(), bShape, 0);

    const double flops = 2.0 * c.M * c.K * c.N;
    MatMulResult r{c.name, c.M, c.K, c.N, c.transposeB, 0, 0, 0};

    size_t idx = 0;
    double te = medianSeconds([&]
                              { matMul2D_out(tA, tBs[idx++ % copies], tO, false, c.transposeB); });
    r.engineMs = te * 1e3;
    r.engineGflops = flops / te / 1e9;
    r.engineWeightGBs = bBytes / te / 1e9;

#ifdef HAVE_ACCELERATE
    auto sgemm = [&](const float *B, float *C)
    {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, c.transposeB ? CblasTrans : CblasNoTrans,
                    static_cast<int>(c.M), static_cast<int>(c.N), static_cast<int>(c.K), 1.0f,
                    A.data(), static_cast<int>(c.K),
                    B, static_cast<int>(c.transposeB ? c.K : c.N),
                    0.0f, C, static_cast<int>(c.N));
    };
    idx = 0;
    double ta = medianSeconds([&]
                              { sgemm(Bs[idx++ % copies].data(), Oref.data()); });
    r.accelMs = ta * 1e3;
    r.accelGflops = flops / ta / 1e9;
    r.accelWeightGBs = bBytes / ta / 1e9;

    // Correctness sanity check on the same B
    matMul2D_out(tA, tBs[0], tO, false, c.transposeB);
    sgemm(Bs[0].data(), Oref.data());
    double maxDiff = 0;
    for (size_t i = 0; i < O.size(); ++i)
        maxDiff = std::max(maxDiff, static_cast<double>(std::fabs(O[i] - Oref[i])));
    r.maxAbsDiff = maxDiff;
#endif
    return r;
}

// ---------------------------------------------------------------------------
// 3. Pointwise / row-wise kernels
// ---------------------------------------------------------------------------
struct EltResult
{
    std::string name;
    size_t rows, cols;
    double us, gbs;
};

EltResult benchElt(const char *name, size_t rows, size_t cols)
{
    auto X = randomBuffer(rows * cols, 7);
    auto Y = randomBuffer(rows * cols, 8);
    auto w = randomBuffer(cols, 9);
    auto b = randomBuffer(cols, 10);
    std::vector<float> O(rows * cols);

    TensorView tX(X.data(), {rows, cols}, 0), tY(Y.data(), {rows, cols}, 0), tO(O.data(), {rows, cols}, 0);
    TensorView tw(w.data(), {1, cols}, 0), tb(b.data(), {1, cols}, 0);

    std::string n(name);
    double bytes = 2.0 * rows * cols * sizeof(float); // read X + write O
    double t = 0;
    if (n == "LayerNorm")
        t = medianSeconds([&]
                          { layerNorm_out(tX, tw, tb, tO); });
    else if (n == "GELU")
        t = medianSeconds([&]
                          { gelu_out(tX, tO); });
    else if (n == "Softmax")
        t = medianSeconds([&]
                          { softmax_out(tX, tO); });
    else if (n == "Add")
    {
        bytes = 3.0 * rows * cols * sizeof(float);
        t = medianSeconds([&]
                          { add2D_out(tX, tY, tO); });
    }
    return {n, rows, cols, t * 1e6, bytes / t / 1e9};
}

std::string jnum(double v)
{
    if (std::isnan(v))
        return "null";
    std::ostringstream os;
    os.precision(6);
    os << v;
    return os.str();
}

} // namespace

int main(int argc, char **argv)
{
    std::string outPath;
    for (int i = 1; i < argc; ++i)
    {
        if (!std::strcmp(argv[i], "--quick"))
            gQuick = true;
        else if (!std::strcmp(argv[i], "--out") && i + 1 < argc)
            outPath = argv[++i];
    }

    unsigned hw = std::thread::hardware_concurrency();
    if (hw == 0)
        hw = 4;

    // ---- Bandwidth ----
    std::printf("== Memory read bandwidth ==\n");
    const size_t bwBytes = gQuick ? (512ull << 20) : (1024ull << 20);
    std::vector<float> bwBuf = randomBuffer(bwBytes / sizeof(float), 3);
    std::vector<std::pair<unsigned, double>> bw;
    std::vector<unsigned> threadCounts{1, std::max(1u, hw / 2), hw};
    threadCounts.erase(std::unique(threadCounts.begin(), threadCounts.end()), threadCounts.end());
    for (unsigned t : threadCounts)
    {
        double g = readBandwidthGBs(bwBuf, t);
        bw.push_back({t, g});
        std::printf("  %2u thread(s): %7.1f GB/s\n", t, g);
    }
    bwBuf.clear();
    bwBuf.shrink_to_fit();

    // ---- MatMul ----
    const size_t V = 50257, D = 768;
    std::vector<MatMulCase> cases;
    std::vector<size_t> Ms = gQuick ? std::vector<size_t>{1, 128} : std::vector<size_t>{1, 32, 128};
    for (size_t M : Ms)
    {
        cases.push_back({"attn.c_attn (QKV)", M, D, 3 * D, false});
        cases.push_back({"attn.c_proj", M, D, D, false});
        cases.push_back({"mlp.c_fc", M, D, 4 * D, false});
        cases.push_back({"mlp.c_proj", M, 4 * D, D, false});
        cases.push_back({"lm_head (wte^T)", M, D, V, true});
    }

    std::printf("\n== MatMul (engine matMul2D_out vs Accelerate cblas_sgemm) ==\n");
    std::printf("  %-18s %5s %5s %6s | %9s %8s %8s | %9s %8s %8s | %8s\n",
                "op", "M", "K", "N", "eng ms", "GFLOP/s", "W GB/s", "acc ms", "GFLOP/s", "W GB/s", "max|d|");
    std::vector<MatMulResult> mm;
    for (auto &c : cases)
    {
        auto r = benchMatMul(c);
        mm.push_back(r);
        std::printf("  %-18s %5zu %5zu %6zu | %9.3f %8.1f %8.1f | %9.3f %8.1f %8.1f | %8.1e\n",
                    r.name.c_str(), r.M, r.K, r.N, r.engineMs, r.engineGflops, r.engineWeightGBs,
                    r.accelMs, r.accelGflops, r.accelWeightGBs, r.maxAbsDiff);
    }

    // ---- Pointwise / row-wise ----
    std::printf("\n== Pointwise / row-wise kernels ==\n");
    std::vector<EltResult> elt;
    for (size_t M : std::vector<size_t>{1, 128})
    {
        elt.push_back(benchElt("LayerNorm", M, D));
        elt.push_back(benchElt("GELU", M, 4 * D));
        elt.push_back(benchElt("Softmax", M * 12, M == 1 ? 128 : M)); // attention scores [heads*N, ctx]
        elt.push_back(benchElt("Add", M, D));
    }
    for (auto &e : elt)
        std::printf("  %-10s [%5zu x %5zu]  %10.2f us  %7.1f GB/s\n", e.name.c_str(), e.rows, e.cols, e.us, e.gbs);

    // ---- JSON ----
    if (!outPath.empty())
    {
        std::ofstream f(outPath);
        f << "{\n  \"hardware_concurrency\": " << hw << ",\n  \"accelerate\": "
#ifdef HAVE_ACCELERATE
          << "true"
#else
          << "false"
#endif
          << ",\n  \"read_bandwidth\": [";
        for (size_t i = 0; i < bw.size(); ++i)
            f << (i ? ", " : "") << "{\"threads\": " << bw[i].first << ", \"gbps\": " << jnum(bw[i].second) << "}";
        f << "],\n  \"matmul\": [\n";
        for (size_t i = 0; i < mm.size(); ++i)
        {
            auto &r = mm[i];
            f << "    {\"op\": \"" << r.name << "\", \"M\": " << r.M << ", \"K\": " << r.K << ", \"N\": " << r.N
              << ", \"transposeB\": " << (r.transposeB ? "true" : "false")
              << ", \"engine_ms\": " << jnum(r.engineMs) << ", \"engine_gflops\": " << jnum(r.engineGflops)
              << ", \"engine_weight_gbps\": " << jnum(r.engineWeightGBs)
              << ", \"accelerate_ms\": " << jnum(r.accelMs) << ", \"accelerate_gflops\": " << jnum(r.accelGflops)
              << ", \"accelerate_weight_gbps\": " << jnum(r.accelWeightGBs)
              << ", \"max_abs_diff\": " << jnum(r.maxAbsDiff) << "}" << (i + 1 < mm.size() ? "," : "") << "\n";
        }
        f << "  ],\n  \"elementwise\": [\n";
        for (size_t i = 0; i < elt.size(); ++i)
        {
            auto &e = elt[i];
            f << "    {\"op\": \"" << e.name << "\", \"rows\": " << e.rows << ", \"cols\": " << e.cols
              << ", \"us\": " << jnum(e.us) << ", \"gbps\": " << jnum(e.gbs) << "}" << (i + 1 < elt.size() ? "," : "") << "\n";
        }
        f << "  ]\n}\n";
    }
    return 0;
}
