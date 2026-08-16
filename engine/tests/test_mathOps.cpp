#include "../tensorView.hpp"
#include "../arenaAllocator.hpp"
#include "../mathOps.hpp"
#include <iostream>
#include <cmath>
#include <cassert>

// A simple macro to make test failures readable
#define ASSERT_APPROX_EQUAL(expected, actual, epsilon)                       \
    do                                                                       \
    {                                                                        \
        if (std::abs((expected) - (actual)) > (epsilon))                     \
        {                                                                    \
            std::cerr << "Test failed: Expected " << (expected)              \
                      << ", got " << (actual)                                \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            std::abort();                                                    \
        }                                                                    \
    } while (0)

void test_add2D()
{
    std::cout << "Running test_add2D... ";
    ArenaAllocator arena(1024);

    float a_data[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float b_data[4] = {5.0f, 6.0f, 7.0f, 8.0f};

    TensorView A(a_data, {2, 2}, 0);
    TensorView B(b_data, {2, 2}, 0);

    TensorView C = add2D(A, B, arena);

    ASSERT_APPROX_EQUAL(6.0f, C(0, 0), 1e-5f);
    ASSERT_APPROX_EQUAL(8.0f, C(0, 1), 1e-5f);
    ASSERT_APPROX_EQUAL(10.0f, C(1, 0), 1e-5f);
    ASSERT_APPROX_EQUAL(12.0f, C(1, 1), 1e-5f);

    std::cout << "PASSED" << std::endl;
}

void test_matMul2D()
{
    std::cout << "Running test_matMul2D... ";
    ArenaAllocator arena(1024);

    float a_data[6] = {2, 3, 4, 5, 2, 1};
    float b_data[6] = {2, 5, 3, 2, 4, 1};

    TensorView A(a_data, {2, 3}, 0);
    TensorView B(b_data, {3, 2}, 0);

    auto C = matMul2D(A, B, arena);

    ASSERT_APPROX_EQUAL(29, C(0, 0), 1e-5f);
    ASSERT_APPROX_EQUAL(20, C(0, 1), 1e-5f);
    ASSERT_APPROX_EQUAL(20, C(1, 0), 1e-5f);
    ASSERT_APPROX_EQUAL(30, C(1, 1), 1e-5f);

    std::cout << "PASSED" << std::endl;
}

void test_gelu()
{
    std::cout << "Running test_gelu... ";
    ArenaAllocator arena(1024);

    float a_data[6] = {2, 3, 4, 5, 2, 1};

    TensorView A(a_data, {1, 6}, 0);

    auto C = gelu(A, arena);

    ASSERT_APPROX_EQUAL(1.9546, C(0, 0), 1e-5f);
    ASSERT_APPROX_EQUAL(2.99636, C(0, 1), 1e-5f);
    ASSERT_APPROX_EQUAL(3.99993, C(0, 2), 1e-5f);
    ASSERT_APPROX_EQUAL(5, C(0, 3), 1e-5f);
    ASSERT_APPROX_EQUAL(1.9546, C(0, 4), 1e-5f);
    ASSERT_APPROX_EQUAL(0.841192, C(0, 5), 1e-5f);

    std::cout << "PASSED" << std::endl;
}

void test_softmax()
{
    std::cout << "Running test_softmax... ";
    ArenaAllocator arena(1024);

    float a_data[6] = {2, 3, 4, 5, 2, 1};

    TensorView A(a_data, {1, 6}, 0);

    auto C = softmax(A, arena);

    ASSERT_APPROX_EQUAL(0.0307118, C(0, 0), 1e-5f);
    ASSERT_APPROX_EQUAL(0.0834834, C(0, 1), 1e-5f);
    ASSERT_APPROX_EQUAL(0.226931, C(0, 2), 1e-5f);
    ASSERT_APPROX_EQUAL(0.616863, C(0, 3), 1e-5f);
    ASSERT_APPROX_EQUAL(0.0307118, C(0, 4), 1e-5f);
    ASSERT_APPROX_EQUAL(0.0112982, C(0, 5), 1e-5f);

    std::cout << "PASSED" << std::endl;
}

void test_layerNorm()
{
    std::cout << "Running test_layerNorm... ";
    ArenaAllocator arena(1024);

    float x_data[6] = {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f};
    float w_data[3] = {1.0f, 1.0f, 1.0f};
    float b_data[3] = {0.0f, 0.0f, 0.0f};

    TensorView X(x_data, {2, 3}, 0);
    TensorView W(w_data, {1, 3}, 0);
    TensorView B(b_data, {1, 3}, 0);

    TensorView Y = layerNorm(X, W, B, arena);

    ASSERT_APPROX_EQUAL(-1.2247f, Y(0, 0), 1e-3f);
    ASSERT_APPROX_EQUAL(0.0f, Y(0, 1), 1e-3f);
    ASSERT_APPROX_EQUAL(1.2247f, Y(0, 2), 1e-3f);

    std::cout << "PASSED" << std::endl;
}

int main()
{
    std::cout << "Starting Math Ops Unit Tests..." << std::endl;

    test_add2D();
    test_matMul2D();
    test_gelu();
    test_softmax();
    test_layerNorm();

    std::cout << "All Math Ops tests passed successfully!" << std::endl;
    return 0;
}
