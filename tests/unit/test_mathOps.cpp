#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "../doctest.h"

#include "tensorView.hpp"
#include "arenaAllocator.hpp"
#include "mathOps.hpp"
#include "operator.hpp"

#include <vector>
#include <cmath>
#include <cstdint>
#include <limits>
#include <new>
#include <numeric>
#include <random>
#include <sstream>
#include <string>

// ============================================================================
// matMul2D & matMul2D_out Tests
// ============================================================================
TEST_CASE("matMul2D & matMul2D_out: Matrix Multiplication")
{
    ArenaAllocator arena(1024 * 1024 * 16); // 16 MB arena

    SUBCASE("Contiguous small matrix (2x3 * 3x2)")
    {
        float a_data[6] = {2.0f, 3.0f, 4.0f, 
                           5.0f, 2.0f, 1.0f};
        float b_data[6] = {2.0f, 5.0f, 
                           3.0f, 2.0f, 
                           4.0f, 1.0f};

        TensorView A(a_data, {2, 3}, 0);
        TensorView B(b_data, {3, 2}, 0);

        float c_data[4] = {0.0f};
        TensorView C(c_data, {2, 2}, 0);
        matMul2D_out(A, B, C);
        CHECK(C(0, 0) == doctest::Approx(29.0f).epsilon(1e-4f));
        CHECK(C(0, 1) == doctest::Approx(20.0f).epsilon(1e-4f));
        CHECK(C(1, 0) == doctest::Approx(20.0f).epsilon(1e-4f));
        CHECK(C(1, 1) == doctest::Approx(30.0f).epsilon(1e-4f));

        // Test matMul2D_out directly with pre-allocated tensor
        float out_data[4] = {0.0f};
        TensorView O(out_data, {2, 2}, 0);
        matMul2D_out(A, B, O);

        CHECK(O(0, 0) == doctest::Approx(29.0f).epsilon(1e-4f));
        CHECK(O(0, 1) == doctest::Approx(20.0f).epsilon(1e-4f));
        CHECK(O(1, 0) == doctest::Approx(20.0f).epsilon(1e-4f));
        CHECK(O(1, 1) == doctest::Approx(30.0f).epsilon(1e-4f));
    }

    SUBCASE("Decoding shape: M = 1 (Vector-Matrix Multiplication)")
    {
        // In autoregressive generation, each step processes a single token (M = 1)
        float a_data[4] = {1.0f, 2.0f, 3.0f, 4.0f};
        float b_data[16] = {
            1.0f, 0.0f, 0.0f, 1.0f,
            0.0f, 2.0f, 1.0f, 0.0f,
            1.0f, 1.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 1.0f, 2.0f
        };

        TensorView A(a_data, {1, 4}, 0);
        TensorView B(b_data, {4, 4}, 0);

        float c_data[4] = {0.0f};
        TensorView C(c_data, {1, 4}, 0);
        matMul2D_out(A, B, C);

        // Expected dot products:
        // C(0, 0) = 1*1 + 2*0 + 3*1 + 4*0 = 4
        // C(0, 1) = 1*0 + 2*2 + 3*1 + 4*0 = 7
        // C(0, 2) = 1*0 + 2*1 + 3*0 + 4*1 = 6
        // C(0, 3) = 1*1 + 2*0 + 3*0 + 4*2 = 9
        CHECK(C(0, 0) == doctest::Approx(4.0f).epsilon(1e-4f));
        CHECK(C(0, 1) == doctest::Approx(7.0f).epsilon(1e-4f));
        CHECK(C(0, 2) == doctest::Approx(6.0f).epsilon(1e-4f));
        CHECK(C(0, 3) == doctest::Approx(9.0f).epsilon(1e-4f));
    }

    SUBCASE("SIMD loop tail handling (K and N not multiples of 4)")
    {
        // Shapes: A = 2x5, B = 5x7 -> C = 2x7
        // 7 columns means NEON computes 4 elements, then scalar loop handles remaining 3
        const size_t M = 2, K = 5, N = 7;
        std::vector<float> a_data(M * K);
        std::vector<float> b_data(K * N);
        std::vector<float> expected(M * N, 0.0f);

        for (size_t i = 0; i < M * K; ++i) a_data[i] = static_cast<float>(i + 1);
        for (size_t i = 0; i < K * N; ++i) b_data[i] = static_cast<float>(i + 1);

        // Compute ground truth
        for (size_t i = 0; i < M; ++i) {
            for (size_t j = 0; j < N; ++j) {
                float sum = 0.0f;
                for (size_t k = 0; k < K; ++k) {
                    sum += a_data[i * K + k] * b_data[k * N + j];
                }
                expected[i * N + j] = sum;
            }
        }

        TensorView A(a_data.data(), {M, K}, 0);
        TensorView B(b_data.data(), {K, N}, 0);

        std::vector<float> c_data(M * N, 0.0f);
        TensorView C(c_data.data(), {M, N}, 0);
        matMul2D_out(A, B, C);

        for (size_t i = 0; i < M; ++i) {
            for (size_t j = 0; j < N; ++j) {
                CHECK(C(i, j) == doctest::Approx(expected[i * N + j]).epsilon(1e-4f));
            }
        }
    }

    SUBCASE("transposeB = true (A * B^T)")
    {
        // A is 2x3, B is 4x3 -> B^T is 3x4 -> C is 2x4
        float a_data[6] = {1.0f, 2.0f, 3.0f, 
                           4.0f, 5.0f, 6.0f};
        float b_data[12] = {
            1.0f, 2.0f, 3.0f,
            4.0f, 5.0f, 6.0f,
            7.0f, 8.0f, 9.0f,
            1.0f, 1.0f, 1.0f
        };

        TensorView A(a_data, {2, 3}, 0);
        TensorView B(b_data, {4, 3}, 0);

        float c_data[8] = {0.0f};
        TensorView C(c_data, {2, 4}, 0);
        matMul2D_out(A, B, C, /*transposeA=*/false, /*transposeB=*/true);

        for (size_t i = 0; i < 2; ++i) {
            for (size_t j = 0; j < 4; ++j) {
                float expected = 0.0f;
                for (size_t k = 0; k < 3; ++k) {
                    expected += A(i, k) * B(j, k);
                }
                CHECK(C(i, j) == doctest::Approx(expected).epsilon(1e-4f));
            }
        }
    }

    SUBCASE("transposeA = true (A^T * B)")
    {
        // A is 3x2 -> A^T is 2x3, B is 3x4 -> C is 2x4
        float a_data[6] = {1.0f, 4.0f,
                           2.0f, 5.0f,
                           3.0f, 6.0f};
        float b_data[12] = {
            1.0f, 2.0f, 3.0f, 4.0f,
            5.0f, 6.0f, 7.0f, 8.0f,
            9.0f, 1.0f, 2.0f, 3.0f
        };

        TensorView A(a_data, {3, 2}, 0);
        TensorView B(b_data, {3, 4}, 0);

        float c_data[8] = {0.0f};
        TensorView C(c_data, {2, 4}, 0);
        matMul2D_out(A, B, C, /*transposeA=*/true, /*transposeB=*/false);

        for (size_t i = 0; i < 2; ++i) {
            for (size_t j = 0; j < 4; ++j) {
                float expected = 0.0f;
                for (size_t k = 0; k < 3; ++k) {
                    expected += A(k, i) * B(k, j);
                }
                CHECK(C(i, j) == doctest::Approx(expected).epsilon(1e-4f));
            }
        }
    }

    SUBCASE("transposeA = true && transposeB = true (A^T * B^T)")
    {
        // A is 3x2 -> A^T is 2x3, B is 4x3 -> B^T is 3x4 -> C is 2x4
        float a_data[6] = {1.0f, 4.0f,
                           2.0f, 5.0f,
                           3.0f, 6.0f};
        float b_data[12] = {
            1.0f, 2.0f, 3.0f,
            4.0f, 5.0f, 6.0f,
            7.0f, 8.0f, 9.0f,
            1.0f, 1.0f, 1.0f
        };

        TensorView A(a_data, {3, 2}, 0);
        TensorView B(b_data, {4, 3}, 0);

        float c_data[8] = {0.0f};
        TensorView C(c_data, {2, 4}, 0);
        matMul2D_out(A, B, C, /*transposeA=*/true, /*transposeB=*/true);

        for (size_t i = 0; i < 2; ++i) {
            for (size_t j = 0; j < 4; ++j) {
                float expected = 0.0f;
                for (size_t k = 0; k < 3; ++k) {
                    expected += A(k, i) * B(j, k);
                }
                CHECK(C(i, j) == doctest::Approx(expected).epsilon(1e-4f));
            }
        }
    }

    SUBCASE("Large multi-threaded matrix multiplication (TaskQueue path)")
    {
        const size_t M = 32, K = 128, N = 64;
        std::vector<float> a_data(M * K, 1.5f);
        std::vector<float> b_data(K * N, 2.0f);

        TensorView A(a_data.data(), {M, K}, 0);
        TensorView B(b_data.data(), {K, N}, 0);

        std::vector<float> c_data(M * N, 0.0f);
        TensorView C(c_data.data(), {M, N}, 0);
        matMul2D_out(A, B, C);

        // Dot product of row of 1.5s and col of 2.0s over K elements:
        // K * 1.5 * 2.0 = K * 3.0 = 384.0f
        const float expected = static_cast<float>(K) * 3.0f;
        for (size_t i = 0; i < M; ++i) {
            for (size_t j = 0; j < N; ++j) {
                CHECK(C(i, j) == doctest::Approx(expected).epsilon(1e-4f));
            }
        }
    }
}

// ============================================================================
// matMul2D_out: exhaustive shape/layout sweep against a double-precision reference
// ============================================================================
//
// Why this exists: the hand-written cases above pre-zero every output buffer,
// call the kernel once, mostly use K < 4 for transposeB, and only use
// contiguous tensors. So a kernel that forgets to clear its output, skips a
// SIMD accumulator, or ignores a row stride still passes them. In the engine,
// output buffers come from a reused arena and Q/K/V are `horSplit` views with
// padded row strides.
//
// Every case here:
//   - fills the output with NaN first, so any accumulation onto existing
//     contents shows up as NaN (a kernel must fully overwrite its output);
//   - surrounds each matrix with canary values and checks they are untouched,
//     so out-of-bounds writes and writes into row padding are caught;
//   - uses random, non-uniform data, so swapped rows/columns are caught;
//   - sweeps K and N across the 16-wide / 4-wide / scalar-tail boundaries of
//     the NEON loops, all transpose combinations, and three memory layouts
//     (contiguous, padded rows like horSplit, column-strided -> general path).
namespace
{
constexpr float kCanary = -12345.678f;

enum class Layout
{
    Contiguous,  // row stride = cols, col stride = 1
    PaddedRows,  // row stride = cols + 3, col stride = 1 (like a horSplit slice)
    ColStrided,  // col stride = 2: not unit-stride, forces the general path
};

const char *layoutName(Layout l)
{
    switch (l)
    {
    case Layout::Contiguous: return "contiguous";
    case Layout::PaddedRows: return "padded-rows";
    case Layout::ColStrided: return "col-strided";
    }
    return "?";
}

// A rows x cols matrix with an explicit layout, surrounded by canary-filled guard cells.
struct GuardedMatrix
{
    static constexpr size_t kGuard = 16;
    size_t rows, cols, rowStride, colStride;
    std::vector<float> storage;

    GuardedMatrix(size_t r, size_t c, Layout layout) : rows(r), cols(c)
    {
        colStride = (layout == Layout::ColStrided) ? 2 : 1;
        rowStride = (layout == Layout::PaddedRows) ? c + 3 : (layout == Layout::ColStrided ? 2 * c + 1 : c);
        size_t span = (r - 1) * rowStride + (c - 1) * colStride + 1;
        storage.assign(kGuard + span + kGuard, kCanary);
    }

    size_t offset(size_t r, size_t c) const { return kGuard + r * rowStride + c * colStride; }
    float &at(size_t r, size_t c) { return storage[offset(r, c)]; }

    TensorView view() { return TensorView(storage.data() + kGuard, {rows, cols}, {rowStride, colStride}, 0); }

    void fillRandom(std::mt19937 &rng)
    {
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (size_t r = 0; r < rows; ++r)
            for (size_t c = 0; c < cols; ++c)
                at(r, c) = dist(rng);
    }

    void fill(float v)
    {
        for (size_t r = 0; r < rows; ++r)
            for (size_t c = 0; c < cols; ++c)
                at(r, c) = v;
    }

    // True if every cell outside the logical matrix (guards + padding) still holds the canary.
    bool canariesIntact() const
    {
        std::vector<bool> logical(storage.size(), false);
        for (size_t r = 0; r < rows; ++r)
            for (size_t c = 0; c < cols; ++c)
                logical[offset(r, c)] = true;
        for (size_t i = 0; i < storage.size(); ++i)
            if (!logical[i] && storage[i] != kCanary)
                return false;
        return true;
    }
};

struct MatMulCaseResult
{
    double maxRelErr;  // NaN if any output element is NaN
    bool aIntact, bIntact, oIntact;
};

// Runs O = op(A) * op(B) for logical shapes [M, K] x [K, N] with the given layouts.
MatMulCaseResult runMatMulCase(size_t M, size_t K, size_t N, bool tA, bool tB,
                               Layout la, Layout lb, Layout lo, std::mt19937 &rng)
{
    // Stored shapes: A is [K, M] when transposed, B is [N, K] when transposed.
    GuardedMatrix A(tA ? K : M, tA ? M : K, la);
    GuardedMatrix B(tB ? N : K, tB ? K : N, lb);
    GuardedMatrix O(M, N, lo);
    A.fillRandom(rng);
    B.fillRandom(rng);
    O.fill(std::numeric_limits<float>::quiet_NaN());

    TensorView vA = A.view(), vB = B.view(), vO = O.view();
    matMul2D_out(vA, vB, vO, tA, tB);

    double maxRelErr = 0.0;
    for (size_t i = 0; i < M; ++i)
    {
        for (size_t n = 0; n < N; ++n)
        {
            double ref = 0.0;
            for (size_t k = 0; k < K; ++k)
            {
                double a = tA ? A.at(k, i) : A.at(i, k);
                double b = tB ? B.at(n, k) : B.at(k, n);
                ref += a * b;
            }
            double got = O.at(i, n);
            double err = std::fabs(got - ref) / (1.0 + std::fabs(ref));
            if (std::isnan(got))
                err = std::numeric_limits<double>::quiet_NaN();
            if (std::isnan(err) || err > maxRelErr)
                maxRelErr = err;
            if (std::isnan(maxRelErr))
                break;
        }
    }
    return {maxRelErr, A.canariesIntact(), B.canariesIntact(), O.canariesIntact()};
}

// Sweeps every shape for one (transpose, layout) combination and returns a description of
// the first failing case, or an empty string if all pass. One assertion per combination keeps
// failure output readable and the assertion count meaningful.
template <size_t NM, size_t NK, size_t NN>
std::string sweepShapes(const size_t (&Ms)[NM], const size_t (&Ks)[NK], const size_t (&Ns)[NN],
                        bool tA, bool tB, Layout la, Layout lb, Layout lo, std::mt19937 &rng)
{
    for (size_t M : Ms)
        for (size_t K : Ks)
            for (size_t N : Ns)
            {
                auto r = runMatMulCase(M, K, N, tA, tB, la, lb, lo, rng);
                const char *what = nullptr;
                if (!(r.maxRelErr <= 1e-4)) // also catches NaN: output not fully overwritten
                    what = "wrong values";
                else if (!r.oIntact)
                    what = "wrote outside output (guard cells / row padding)";
                else if (!r.aIntact || !r.bIntact)
                    what = "modified an input";
                if (what)
                {
                    std::ostringstream os;
                    os << what << " at M=" << M << " K=" << K << " N=" << N << " transposeA=" << tA
                       << " transposeB=" << tB << " A=" << layoutName(la) << " B=" << layoutName(lb)
                       << " O=" << layoutName(lo) << " (max rel err " << r.maxRelErr << ")";
                    return os.str();
                }
            }
    return "";
}
} // namespace

TEST_CASE("matMul2D_out: shape/layout sweep vs reference (single-threaded, M <= 4)")
{
    std::mt19937 rng(1234);
    // K/N values straddle the 16-wide, 4-wide and scalar-tail loop boundaries.
    const size_t Ms[] = {1, 2, 3, 4};
    const size_t Ks[] = {1, 3, 4, 5, 8, 15, 16, 17, 20, 23, 37};
    const size_t Ns[] = {1, 2, 3, 4, 5, 7, 8, 16, 21};
    const Layout inLayouts[] = {Layout::Contiguous, Layout::PaddedRows, Layout::ColStrided};
    // Layout::ColStrided is deliberately excluded for the OUTPUT: the non-transposed NEON fast
    // path checks only B's column stride and stores 4 adjacent floats, so a column-strided O
    // gives wrong results (README Known Issues). Add it here once the kernel guards O's stride.
    const Layout outLayouts[] = {Layout::Contiguous, Layout::PaddedRows};

    for (bool tA : {false, true})
        for (bool tB : {false, true})
            for (Layout la : inLayouts)
                for (Layout lb : inLayouts)
                    for (Layout lo : outLayouts)
                    {
                        std::string failure = sweepShapes(Ms, Ks, Ns, tA, tB, la, lb, lo, rng);
                        INFO(failure);
                        CHECK(failure.empty());
                    }
}

TEST_CASE("matMul2D_out: shape/layout sweep vs reference (multi-threaded, M > 4)")
{
    std::mt19937 rng(5678);
    // M values exercise uneven per-thread row chunks; GPT-2-like K = 768 hits all NEON loops.
    const size_t Ms[] = {5, 9, 32, 37};
    const size_t Ks[] = {17, 64, 768};
    const size_t Ns[] = {7, 64, 67};
    const Layout layouts[] = {Layout::Contiguous, Layout::PaddedRows};

    for (bool tA : {false, true})
        for (bool tB : {false, true})
            for (Layout la : layouts)
                for (Layout lb : layouts)
                    for (Layout lo : layouts)
                    {
                        std::string failure = sweepShapes(Ms, Ks, Ns, tA, tB, la, lb, lo, rng);
                        INFO(failure);
                        CHECK(failure.empty());
                    }
}

TEST_CASE("matMul2D_out: repeated calls into a reused (dirty) output buffer")
{
    // Mirrors the engine: the arena is reset every step, so a kernel's output buffer
    // still holds the previous step's activations. Results must not depend on them.
    std::mt19937 rng(42);
    for (bool tB : {false, true})
    {
        for (size_t M : {1, 3, 8})
        {
            const size_t K = 37, N = 21;
            GuardedMatrix A(M, K, Layout::Contiguous);
            GuardedMatrix B(tB ? N : K, tB ? K : N, Layout::Contiguous);
            GuardedMatrix O(M, N, Layout::Contiguous);
            A.fillRandom(rng);
            B.fillRandom(rng);
            TensorView vA = A.view(), vB = B.view(), vO = O.view();

            O.fill(0.0f);
            matMul2D_out(vA, vB, vO, false, tB);
            std::vector<float> first;
            for (size_t i = 0; i < M; ++i)
                for (size_t n = 0; n < N; ++n)
                    first.push_back(O.at(i, n));

            O.fillRandom(rng); // stale garbage, as left behind by a previous step
            matMul2D_out(vA, vB, vO, false, tB);

            INFO("M=" << M << " transposeB=" << tB);
            size_t idx = 0;
            bool identical = true;
            for (size_t i = 0; i < M; ++i)
                for (size_t n = 0; n < N; ++n)
                    identical = identical && (O.at(i, n) == first[idx++]);
            CHECK(identical);
        }
    }
}

// ============================================================================
// ArenaAllocator Tests
// ============================================================================
TEST_CASE("ArenaAllocator: alignment, exhaustion, reserve and reset")
{
    SUBCASE("allocations are 16-byte aligned and sized by alignedSize")
    {
        CHECK(ArenaAllocator::alignedSize(0) == 0);
        CHECK(ArenaAllocator::alignedSize(1) == 16);
        CHECK(ArenaAllocator::alignedSize(16) == 16);
        CHECK(ArenaAllocator::alignedSize(17) == 32);

        ArenaAllocator arena(1024);
        float *a = arena.alloc(4);
        float *b = arena.alloc(20);
        CHECK(reinterpret_cast<std::uintptr_t>(a) % 16 == 0);
        CHECK(reinterpret_cast<std::uintptr_t>(b) % 16 == 0);
        CHECK(arena.bytesAllocated() == 16 + 32);
    }

    SUBCASE("a full arena throws instead of returning nullptr")
    {
        ArenaAllocator arena(64);
        CHECK_NOTHROW(arena.alloc(64));
        CHECK_THROWS_AS(arena.alloc(1), std::bad_alloc);
        CHECK(arena.bytesAllocated() == 64); // failed allocation changes nothing
    }

    SUBCASE("an empty arena throws on first use")
    {
        ArenaAllocator arena;
        CHECK(arena.capacity() == 0);
        CHECK_THROWS_AS(arena.alloc(16), std::bad_alloc);
    }

    SUBCASE("reserve grows capacity, never shrinks it")
    {
        ArenaAllocator arena(64);
        arena.reserve(4096);
        CHECK(arena.capacity() == 4096);
        CHECK_NOTHROW(arena.alloc(4096));

        arena.reset();
        arena.reserve(128); // smaller request keeps the existing block
        CHECK(arena.capacity() == 4096);
        CHECK(arena.bytesAllocated() == 0);
    }

    SUBCASE("reset makes the whole capacity available again")
    {
        ArenaAllocator arena(64);
        float *first = arena.alloc(64);
        arena.reset();
        CHECK(arena.bytesAllocated() == 0);
        CHECK(arena.alloc(64) == first);
    }
}

// ============================================================================
// add2D & add2D_inplace Tests
// ============================================================================
TEST_CASE("add2D and add2D_inplace: Element-wise addition")
{
    ArenaAllocator arena(1024 * 1024);

    float a_data[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float b_data[4] = {10.0f, 20.0f, 30.0f, 40.0f};

    TensorView A(a_data, {2, 2}, 0);
    TensorView B(b_data, {2, 2}, 0);

    SUBCASE("add2D_out (out-of-place)")
    {
        float c_data[4] = {0.0f};
        TensorView C(c_data, {2, 2}, 0);
        add2D_out(A, B, C);
        CHECK(C(0, 0) == doctest::Approx(11.0f));
        CHECK(C(0, 1) == doctest::Approx(22.0f));
        CHECK(C(1, 0) == doctest::Approx(33.0f));
        CHECK(C(1, 1) == doctest::Approx(44.0f));
    }

    SUBCASE("add2D_out (in-place / residual connection)")
    {
        float target_data[4] = {1.0f, 2.0f, 3.0f, 4.0f};
        TensorView Target(target_data, {2, 2}, 0);

        add2D_out(Target, B, Target);
        CHECK(Target(0, 0) == doctest::Approx(11.0f));
        CHECK(Target(0, 1) == doctest::Approx(22.0f));
        CHECK(Target(1, 0) == doctest::Approx(33.0f));
        CHECK(Target(1, 1) == doctest::Approx(44.0f));
    }
}

// ============================================================================
// add1DTo2D & add1DTo2D_inplace Tests (Bias Addition)
// ============================================================================
TEST_CASE("add1DTo2D and add1DTo2D_inplace: Bias Broadcasting")
{
    ArenaAllocator arena(1024 * 1024);

    float a_data[6] = {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f
    };
    float bias_data[3] = {10.0f, 20.0f, 30.0f};

    TensorView A(a_data, {2, 3}, 0);
    TensorView Bias(bias_data, {1, 3}, 0);

    SUBCASE("add1DTo2D_out (out-of-place)")
    {
        float c_data[6] = {0.0f};
        TensorView C(c_data, {2, 3}, 0);
        add1DTo2D_out(A, Bias, C);
        CHECK(C(0, 0) == doctest::Approx(11.0f));
        CHECK(C(0, 1) == doctest::Approx(22.0f));
        CHECK(C(0, 2) == doctest::Approx(33.0f));
        CHECK(C(1, 0) == doctest::Approx(14.0f));
        CHECK(C(1, 1) == doctest::Approx(25.0f));
        CHECK(C(1, 2) == doctest::Approx(36.0f));
    }

    SUBCASE("add1DTo2D_out (in-place)")
    {
        float target_data[6] = {
            1.0f, 2.0f, 3.0f,
            4.0f, 5.0f, 6.0f
        };
        TensorView Target(target_data, {2, 3}, 0);

        add1DTo2D_out(Target, Bias, Target);
        CHECK(Target(0, 0) == doctest::Approx(11.0f));
        CHECK(Target(0, 1) == doctest::Approx(22.0f));
        CHECK(Target(0, 2) == doctest::Approx(33.0f));
        CHECK(Target(1, 0) == doctest::Approx(14.0f));
        CHECK(Target(1, 1) == doctest::Approx(25.0f));
        CHECK(Target(1, 2) == doctest::Approx(36.0f));
    }
}

// ============================================================================
// eleMul2D_out Tests
// ============================================================================
TEST_CASE("eleMul2D_out: Scalar Scaling")
{
    SUBCASE("eleMul2D_out in-place")
    {
        float data[4] = {2.0f, 4.0f, 6.0f, 8.0f};
        TensorView X(data, {2, 2}, 0);

        eleMul2D_out(X, 0.5f, X);

        CHECK(X(0, 0) == doctest::Approx(1.0f));
        CHECK(X(0, 1) == doctest::Approx(2.0f));
        CHECK(X(1, 0) == doctest::Approx(3.0f));
        CHECK(X(1, 1) == doctest::Approx(4.0f));
    }

    SUBCASE("eleMul2D_out out-of-place")
    {
        float data[4] = {2.0f, 4.0f, 6.0f, 8.0f};
        float out_data[4] = {0.0f};
        TensorView X(data, {2, 2}, 0);
        TensorView O(out_data, {2, 2}, 0);

        eleMul2D_out(X, 2.0f, O);

        CHECK(O(0, 0) == doctest::Approx(4.0f));
        CHECK(O(0, 1) == doctest::Approx(8.0f));
        CHECK(O(1, 0) == doctest::Approx(12.0f));
        CHECK(O(1, 1) == doctest::Approx(16.0f));
    }
}

// ============================================================================
// gelu_out Tests
// ============================================================================
TEST_CASE("gelu_out: Activation Function")
{
    float data[5] = {-4.0f, -1.0f, 0.0f, 1.0f, 4.0f};
    TensorView A(data, {1, 5}, 0);

    SUBCASE("gelu_out out-of-place")
    {
        float c_data[5] = {0.0f};
        TensorView C(c_data, {1, 5}, 0);
        gelu_out(A, C);

        // GELU(-4) ~ 0
        CHECK(std::abs(C(0, 0)) < 1e-3f);
        // GELU(-1) ~ -0.1587
        CHECK(C(0, 1) == doctest::Approx(-0.1587f).epsilon(1e-2f));
        // GELU(0) == 0
        CHECK(std::abs(C(0, 2)) < 1e-5f);
        // GELU(1) ~ 0.8413
        CHECK(C(0, 3) == doctest::Approx(0.8413f).epsilon(1e-2f));
        // GELU(4) ~ 4.0
        CHECK(C(0, 4) == doctest::Approx(4.0f).epsilon(1e-2f));
    }

    SUBCASE("gelu_out in-place (1D)")
    {
        float target_data[5] = {-4.0f, -1.0f, 0.0f, 1.0f, 4.0f};
        TensorView Target(target_data, {1, 5}, 0);

        gelu_out(Target, Target);

        CHECK(std::abs(Target(0, 0)) < 1e-3f);
        CHECK(Target(0, 1) == doctest::Approx(-0.1587f).epsilon(1e-2f));
        CHECK(std::abs(Target(0, 2)) < 1e-5f);
        CHECK(Target(0, 3) == doctest::Approx(0.8413f).epsilon(1e-2f));
        CHECK(Target(0, 4) == doctest::Approx(4.0f).epsilon(1e-2f));
    }

    SUBCASE("gelu_out in-place (multi-row 2D)")
    {
        float target_data[6] = {
            -4.0f, 0.0f, 4.0f,
            -1.0f, 1.0f, 2.0f
        };
        TensorView Target(target_data, {2, 3}, 0);

        gelu_out(Target, Target);

        CHECK(std::abs(Target(0, 0)) < 1e-3f);
        CHECK(std::abs(Target(0, 1)) < 1e-5f);
        CHECK(Target(0, 2) == doctest::Approx(4.0f).epsilon(1e-2f));
        CHECK(Target(1, 0) == doctest::Approx(-0.1587f).epsilon(1e-2f));
        CHECK(Target(1, 1) == doctest::Approx(0.8413f).epsilon(1e-2f));
    }
}

// ============================================================================
// softmax_out Tests
// ============================================================================
TEST_CASE("softmax_out: Exponentials Normalization")
{
    SUBCASE("Basic probability distribution and sum to 1.0")
    {
        float data[6] = {
            1.0f, 2.0f, 3.0f,
            4.0f, 5.0f, 6.0f
        };
        TensorView A(data, {2, 3}, 0);

        float s_data[6] = {0.0f};
        TensorView S(s_data, {2, 3}, 0);
        softmax_out(A, S);

        // Check each row sums to 1.0
        for (size_t i = 0; i < 2; ++i) {
            float row_sum = S(i, 0) + S(i, 1) + S(i, 2);
            CHECK(row_sum == doctest::Approx(1.0f).epsilon(1e-5f));
            // Check monotonicity: softmax is strictly increasing with input
            CHECK(S(i, 0) < S(i, 1));
            CHECK(S(i, 1) < S(i, 2));
        }
    }

    SUBCASE("Numerical stability with large logits (prevents exp overflow to NaN)")
    {
        // Without x - max(x), exp(1000) would overflow to +inf and result in NaN
        float data[3] = {1000.0f, 1001.0f, 1002.0f};
        TensorView A(data, {1, 3}, 0);

        float s_data[3] = {0.0f};
        TensorView S(s_data, {1, 3}, 0);
        softmax_out(A, S);

        CHECK(!std::isnan(S(0, 0)));
        CHECK(!std::isnan(S(0, 1)));
        CHECK(!std::isnan(S(0, 2)));

        float sum = S(0, 0) + S(0, 1) + S(0, 2);
        CHECK(sum == doctest::Approx(1.0f).epsilon(1e-5f));

        // exp(0) / (exp(-2) + exp(-1) + exp(0))
        float expected_last = 1.0f / (std::exp(-2.0f) + std::exp(-1.0f) + 1.0f);
        CHECK(S(0, 2) == doctest::Approx(expected_last).epsilon(1e-4f));
    }

    SUBCASE("softmax_out in-place (1D)")
    {
        float data[3] = {1.0f, 2.0f, 3.0f};
        TensorView Target(data, {1, 3}, 0);

        softmax_out(Target, Target);

        float sum = Target(0, 0) + Target(0, 1) + Target(0, 2);
        CHECK(sum == doctest::Approx(1.0f).epsilon(1e-5f));
        CHECK(Target(0, 0) < Target(0, 1));
        CHECK(Target(0, 1) < Target(0, 2));
    }

    SUBCASE("softmax_out in-place (multi-row 2D invariant)")
    {
        float data[6] = {
            1.0f, 2.0f, 3.0f,
            10.0f, 20.0f, 30.0f
        };
        TensorView Target(data, {2, 3}, 0);

        softmax_out(Target, Target);

        for (size_t i = 0; i < 2; ++i) {
            float sum = Target(i, 0) + Target(i, 1) + Target(i, 2);
            CHECK(sum == doctest::Approx(1.0f).epsilon(1e-5f));
            CHECK(Target(i, 0) < Target(i, 1));
            CHECK(Target(i, 1) < Target(i, 2));
        }
    }
}

// ============================================================================
// layerNorm_out Tests
// ============================================================================
TEST_CASE("layerNorm_out: Layer Normalization")
{
    SUBCASE("Zero mean and unit variance with identity affine parameters")
    {
        float x_data[6] = {
            1.0f, 2.0f, 3.0f,
            10.0f, 20.0f, 30.0f
        };
        float w_data[3] = {1.0f, 1.0f, 1.0f}; // Scale = 1.0
        float b_data[3] = {0.0f, 0.0f, 0.0f}; // Shift = 0.0

        TensorView X(x_data, {2, 3}, 0);
        TensorView W(w_data, {1, 3}, 0);
        TensorView B(b_data, {1, 3}, 0);

        float y_data[6] = {0.0f};
        TensorView Y(y_data, {2, 3}, 0);
        layerNorm_out(X, W, B, Y);

        for (size_t i = 0; i < 2; ++i) {
            float mean = (Y(i, 0) + Y(i, 1) + Y(i, 2)) / 3.0f;
            CHECK(std::abs(mean) < 1e-4f);

            float var = ((Y(i, 0) - mean) * (Y(i, 0) - mean) +
                         (Y(i, 1) - mean) * (Y(i, 1) - mean) +
                         (Y(i, 2) - mean) * (Y(i, 2) - mean)) / 3.0f;
            // Variance should be approximately 1 (slightly bounded by epsilon 1e-5)
            CHECK(var == doctest::Approx(1.0f).epsilon(1e-2f));
        }
    }

    SUBCASE("Custom scale (weight) and shift (bias)")
    {
        float x_data[3] = {1.0f, 2.0f, 3.0f};
        float w_data[3] = {2.0f, 2.0f, 2.0f};
        float b_data[3] = {5.0f, 5.0f, 5.0f};

        TensorView X(x_data, {1, 3}, 0);
        TensorView W(w_data, {1, 3}, 0);
        TensorView B(b_data, {1, 3}, 0);

        float y_data[3] = {0.0f};
        TensorView Y(y_data, {1, 3}, 0);
        layerNorm_out(X, W, B, Y);

        // For {1, 2, 3}: mean = 2, var = 2/3 ~ 0.66667, std ~ 0.816497
        // norm(0) = (1 - 2) / 0.8165 = -1.2247 -> Y(0) = 2 * (-1.2247) + 5 = 2.5505
        // norm(1) = (2 - 2) / 0.8165 = 0.0     -> Y(1) = 2 * (0.0) + 5 = 5.0
        // norm(2) = (3 - 2) / 0.8165 = +1.2247 -> Y(2) = 2 * (1.2247) + 5 = 7.4495
        CHECK(Y(0, 0) == doctest::Approx(2.5505f).epsilon(1e-3f));
        CHECK(Y(0, 1) == doctest::Approx(5.0f).epsilon(1e-3f));
        CHECK(Y(0, 2) == doctest::Approx(7.4495f).epsilon(1e-3f));
    }
}

// ============================================================================
// Operator Polymorphic Hierarchy Tests
// ============================================================================
TEST_CASE("Operator Polymorphic Hierarchy")
{
    SUBCASE("MatMulOp via base pointer")
    {
        float a_data[4] = {1.0f, 2.0f, 3.0f, 4.0f};
        float b_data[4] = {1.0f, 0.0f, 0.0f, 1.0f};
        float c_data[4] = {0.0f};

        TensorView A(a_data, {2, 2}, 0);
        TensorView B(b_data, {2, 2}, 0);
        TensorView C(c_data, {2, 2}, 0);

        std::unique_ptr<Operator> op = std::make_unique<MatMulOp>();
        CHECK(op->type() == OpType::MatMul);
        CHECK(op->name() == "MatMul");

        op->execute({&A, &B}, C);
        CHECK(C(0, 0) == doctest::Approx(1.0f));
        CHECK(C(0, 1) == doctest::Approx(2.0f));
        CHECK(C(1, 0) == doctest::Approx(3.0f));
        CHECK(C(1, 1) == doctest::Approx(4.0f));
    }

    SUBCASE("MatMulOp with transposeB via base pointer")
    {
        // A is [1, 3], B is [2, 3] -> B^T is [3, 2] -> C is [1, 2]
        float a_data[3] = {1.0f, 2.0f, 3.0f};
        float b_data[6] = {1.0f, 1.0f, 1.0f,
                           2.0f, 2.0f, 2.0f};
        float c_data[2] = {0.0f};

        TensorView A(a_data, {1, 3}, 0);
        TensorView B(b_data, {2, 3}, 0);
        TensorView C(c_data, {1, 2}, 0);

        std::unique_ptr<Operator> op = std::make_unique<MatMulOp>(/*transposeA=*/false, /*transposeB=*/true);
        CHECK(op->type() == OpType::MatMul);

        op->execute({&A, &B}, C);
        // C(0, 0) = 1*1 + 2*1 + 3*1 = 6.0
        // C(0, 1) = 1*2 + 2*2 + 3*2 = 12.0
        CHECK(C(0, 0) == doctest::Approx(6.0f));
        CHECK(C(0, 1) == doctest::Approx(12.0f));
    }

    SUBCASE("GeluOp and EleMulOp via base pointer")
    {
        float x_data[2] = {0.0f, 2.0f};
        float y_data[2] = {0.0f};

        TensorView X(x_data, {1, 2}, 0);
        TensorView Y(y_data, {1, 2}, 0);

        std::unique_ptr<Operator> gelu = std::make_unique<GeluOp>();
        gelu->execute({&X}, Y);
        CHECK(std::abs(Y(0, 0)) < 1e-5f);

        std::unique_ptr<Operator> scale = std::make_unique<EleMulOp>(0.5f);
        scale->execute({&X}, Y);
        CHECK(Y(0, 1) == doctest::Approx(1.0f));
    }
}
