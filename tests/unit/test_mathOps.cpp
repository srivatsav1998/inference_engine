#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "../doctest.h"

#include "tensorView.hpp"
#include "arenaAllocator.hpp"
#include "mathOps.hpp"

#include <vector>
#include <cmath>
#include <numeric>

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

    SUBCASE("Transposed B matrix (stride[1] != 1 non-contiguous branch)")
    {
        // A is 2x3, B_orig is 4x3. B = transpose(B_orig) is 3x4 with stride[1] != 1.
        float a_data[6] = {1.0f, 2.0f, 3.0f, 
                           4.0f, 5.0f, 6.0f};
        float b_orig_data[12] = {
            1.0f, 2.0f, 3.0f,
            4.0f, 5.0f, 6.0f,
            7.0f, 8.0f, 9.0f,
            1.0f, 1.0f, 1.0f
        };

        TensorView A(a_data, {2, 3}, 0);
        TensorView B_orig(b_orig_data, {4, 3}, 0);
        TensorView B = transpose2D(B_orig);

        REQUIRE(B.shape()[0] == 3);
        REQUIRE(B.shape()[1] == 4);
        REQUIRE(B.stride()[1] != 1); // Triggers the non-contiguous branch

        float c_data[8] = {0.0f};
        TensorView C(c_data, {2, 4}, 0);
        matMul2D_out(A, B, C);

        // Compute expected A * B_transposed
        for (size_t i = 0; i < 2; ++i) {
            for (size_t j = 0; j < 4; ++j) {
                float expected = 0.0f;
                for (size_t k = 0; k < 3; ++k) {
                    expected += A(i, k) * B(k, j);
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
// transpose2D Tests
// ============================================================================
TEST_CASE("transpose2D: Zero-copy Transposition")
{
    float data[6] = {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f
    };
    TensorView X(data, {2, 3}, 0);

    TensorView XT = transpose2D(X);

    CHECK(XT.shape()[0] == 3);
    CHECK(XT.shape()[1] == 2);
    CHECK(XT.get() == X.get()); // Verifies zero-copy underlying buffer

    // Check transposed coordinates
    CHECK(XT(0, 0) == doctest::Approx(1.0f));
    CHECK(XT(0, 1) == doctest::Approx(4.0f));
    CHECK(XT(1, 0) == doctest::Approx(2.0f));
    CHECK(XT(1, 1) == doctest::Approx(5.0f));
    CHECK(XT(2, 0) == doctest::Approx(3.0f));
    CHECK(XT(2, 1) == doctest::Approx(6.0f));
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
