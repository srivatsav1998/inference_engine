#include "mathOps.hpp"

#include "tensorView.hpp"
#include "arenaAllocator.hpp"
#include "taskQueue.hpp"

#include <numbers>
#include <arm_neon.h>

//////////////// Accepts output Variants ////////////////

void matMul2D_out(const TensorView &A, const TensorView &B, TensorView &O, bool transposeA, bool transposeB)
{
    assert(O.get() != A.get() && O.get() != B.get() && "matMul2D_out: output buffer cannot alias input buffers");

    auto a_shape = A.shape();
    auto b_shape = B.shape();
    auto o_shape = O.shape();

    size_t M = transposeA ? a_shape[1] : a_shape[0];
    size_t K_A = transposeA ? a_shape[0] : a_shape[1];
    size_t K_B = transposeB ? b_shape[1] : b_shape[0];
    size_t N = transposeB ? b_shape[0] : b_shape[1];

    assert(K_A == K_B && "matMul2D_out: inner dimensions must match");
    assert(o_shape[0] == M && o_shape[1] == N && "matMul2D_out: output shape mismatch");

    size_t K = K_A;

    auto numThreads = std::thread::hardware_concurrency();
    if (numThreads == 0)
    {
        // hardware_concurrency can sometimes return 0 when not implemented correctly.
        numThreads = 4; // fall back value
    }

    auto action = [&A, &B, &O, N, K, transposeA, transposeB](size_t startI, size_t endI)
    {
        if (!transposeA && !transposeB && B.stride()[1] == 1)
        {
            // Fast contiguous NEON path
            for (size_t i = startI; i < endI; ++i)
            {
                for (size_t j = 0; j < K; ++j)
                {
                    float32x4_t a_vec = vld1q_dup_f32(&A(i, j));
                    size_t k = 0;
                    for (; k + 4 <= N; k += 4)
                    {
                        float32x4_t b_vec = vld1q_f32(&B(j, k));

                        if (j == 0)
                        {
                            float32x4_t temp = vdupq_n_f32(0.0f);
                            vst1q_f32(&O(i, k), temp);
                        }

                        float32x4_t o_vec = vld1q_f32(&O(i, k));
                        o_vec = vmlaq_f32(o_vec, a_vec, b_vec);
                        vst1q_f32(&O(i, k), o_vec);
                    }

                    if (N % 4 != 0)
                    {
                        for (; k < N; ++k)
                        {
                            if (j == 0)
                            {
                                O(i, k) = A(i, j) * B(j, k);
                            }
                            else
                            {
                                O(i, k) += A(i, j) * B(j, k);
                            }
                        }
                    }
                }
            }
        }
        else if (!transposeA && transposeB && A.stride()[1] == 1 && B.stride()[1] == 1)
        {
            // Fast contiguous NEON dot-product path:
            // O(i, k) = dot_product(row i of A, row k of B) over K elements
            for (size_t i = startI; i < endI; ++i)
            {
                for (size_t k = 0; k < N; ++k)
                {
                    float sum = 0.0f;
                    size_t j = 0;
                    float32x4_t acc = vdupq_n_f32(0.0f);
                    for (; j + 4 <= K; j += 4)
                    {
                        float32x4_t a_vec = vld1q_f32(&A(i, j));
                        float32x4_t b_vec = vld1q_f32(&B(k, j));
                        acc = vmlaq_f32(acc, a_vec, b_vec);
                    }
                    sum = vaddvq_f32(acc);

                    for (; j < K; ++j)
                    {
                        sum += A(i, j) * B(k, j);
                    }
                    O(i, k) = sum;
                }
            }
        }
        else
        {
            // General path (transposeA, or arbitrary non-contiguous strides)
            for (size_t i = startI; i < endI; ++i)
            {
                for (size_t j = 0; j < K; ++j)
                {
                    float a_val = transposeA ? A(j, i) : A(i, j);
                    for (size_t k = 0; k < N; ++k)
                    {
                        float b_val = transposeB ? B(k, j) : B(j, k);
                        if (j == 0)
                        {
                            O(i, k) = a_val * b_val;
                        }
                        else
                        {
                            O(i, k) += a_val * b_val;
                        }
                    }
                }
            }
        }
    };

    auto totalIters = o_shape[0];

    if (totalIters <= 4)
    {
        // lower totalIters value doesn't justify the massive overhead of a threadpool
        // hence execute and return early
        action(0, totalIters);
        return;
    }

    auto chunkSize = (totalIters + numThreads - 1) / numThreads;

    auto &taskQueue = TaskQueue::getInstance();

    for (unsigned int i = 0; i < numThreads; ++i)
    {
        // for every thread compute start and end
        size_t startIdx = i * chunkSize;
        size_t endIdx = (i + 1) * chunkSize;

        if (startIdx >= totalIters)
        {
            // extra threads - do nothing in this case
            continue;
        }

        if (endIdx > totalIters)
        {
            endIdx = totalIters;
        }

        taskQueue.enqueueTask([&action, startIdx, endIdx]
                              { action(startIdx, endIdx); });
    }

    std::unique_lock lck(taskQueue.tasksCntMutex_);
    taskQueue.tasksCntCv_.wait(lck, [&taskQueue]
                               { return taskQueue.tasksCnt_.load() == 0; });
}

//////////////// General Variants ////////////////

void add2D_out(const TensorView &A, const TensorView &B, TensorView &O)
{
    auto a_shape = A.shape();
    auto b_shape = B.shape();
    auto o_shape = O.shape();

    // ensure that matrix addition is possible
    assert(a_shape[0] == b_shape[0] && a_shape[1] == b_shape[1] && "Provided matrices are not compatible for add operation");
    assert(a_shape == o_shape && "add2D_out: output shape must match input shape");

    if (A.get() == O.get())
    {
        assert(A.stride() == O.stride() && "In-place add2D_out requires identical strides between A and O");
    }
    if (B.get() == O.get())
    {
        assert(B.stride() == O.stride() && "In-place add2D_out requires identical strides between B and O");
    }

    for (size_t i = 0; i < a_shape[0]; ++i)
    {
        for (size_t j = 0; j < a_shape[1]; ++j)
        {
            O(i, j) = A(i, j) + B(i, j);
        }
    }
}

void add1DTo2D_out(const TensorView &A, const TensorView &b, TensorView &O)
{
    // output shape will be same as A
    auto a_shape = A.shape();
    auto b_shape = b.shape();
    auto o_shape = O.shape();

    assert(a_shape[1] == b_shape[1] && "The columns are not equal");
    assert(a_shape == o_shape && "add1DTo2D_out: output shape must match input A shape");

    if (A.get() == O.get())
    {
        assert(A.stride() == O.stride() && "In-place add1DTo2D_out requires identical strides between A and O");
    }

    for (size_t i = 0; i < a_shape[0]; ++i)
    {
        for (size_t j = 0; j < a_shape[1]; ++j)
        {
            O(i, j) = A(i, j) + b(0, j);
        }
    }
}

void gelu_out(const TensorView &A, TensorView &O)
{
    auto a_shape = A.shape();
    auto o_shape = O.shape();

    assert(a_shape == o_shape && "gelu_out: input and output shapes must match");
    if (A.get() == O.get())
    {
        assert(A.stride() == O.stride() && "In-place gelu_out requires identical strides");
    }

    const float prefix = sqrt(2 / std::numbers::pi_v<float>);

    auto gelu_op = [&prefix](float x)
    {
        return 0.5 * x * (1.0 + tanh(prefix * (x + (0.044715 * x * x * x))));
    };

    for (size_t i = 0; i < a_shape[0]; ++i)
    {
        for (size_t j = 0; j < a_shape[1]; ++j)
        {
            O(i, j) = gelu_op(A(i, j));
        }
    }
}

void softmax_out(const TensorView &A, TensorView &O)
{
    // for softmax - the input can be a matrix
    auto a_shape = A.shape();
    auto o_shape = O.shape();

    assert(a_shape == o_shape && "softmax_out: input and output shapes must match");
    if (A.get() == O.get())
    {
        assert(A.stride() == O.stride() && "In-place softmax_out requires identical strides");
    }

    for (size_t i = 0; i < a_shape[0]; ++i)
    {
        // Apply softmax logic for every row
        float maxVal = A(i, 0);
        for (size_t j = 0; j < a_shape[1]; ++j)
        {
            maxVal = std::max(maxVal, A(i, j));
        }

        float totalSum = 0;
        for (size_t j = 0; j < a_shape[1]; ++j)
        {
            O(i, j) = std::exp(A(i, j) - maxVal);
            totalSum += O(i, j);
        }

        for (size_t j = 0; j < a_shape[1]; ++j)
        {
            O(i, j) /= totalSum;
        }
    }
}

void layerNorm_out(const TensorView &X, const TensorView &weight, const TensorView &bias, TensorView &O)
{
    auto out_shape = X.shape();
    auto o_shape = O.shape();

    assert(out_shape == o_shape && "layerNorm_out: output shape must match input X shape");
    if (X.get() == O.get())
    {
        assert(X.stride() == O.stride() && "In-place layerNorm_out requires identical strides");
    }

    for (size_t i = 0; i < out_shape[0]; ++i)
    {
        // for every row
        // find the mean value
        float rowTotal = 0;
        for (size_t j = 0; j < out_shape[1]; ++j)
        {
            rowTotal += X(i, j);
        }
        auto rowMean = rowTotal / out_shape[1];

        // find the variance
        rowTotal = 0;
        for (size_t j = 0; j < out_shape[1]; ++j)
        {
            rowTotal += ((X(i, j) - rowMean) * (X(i, j) - rowMean));
        }
        auto rowVariance = rowTotal / out_shape[1];

        // Normalizing the value
        for (size_t j = 0; j < out_shape[1]; ++j)
        {
            auto normalizedVal = (X(i, j) - rowMean) / sqrt(rowVariance + 0.00001);
            O(i, j) = weight(0, j) * normalizedVal + bias(0, j);
        }
    }
}

void eleMul2D_out(const TensorView &X, float val, TensorView &O)
{
    auto x_shape = X.shape();
    auto o_shape = O.shape();

    assert(x_shape == o_shape && "eleMul2D_out: input and output shapes must match");
    if (X.get() == O.get())
    {
        assert(X.stride() == O.stride() && "In-place eleMul2D_out requires identical strides");
    }

    for (size_t i = 0; i < x_shape[0]; ++i)
    {
        for (size_t j = 0; j < x_shape[1]; ++j)
        {
            O(i, j) = X(i, j) * val;
        }
    }
}
