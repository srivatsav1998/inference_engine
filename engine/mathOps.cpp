#include "mathOps.hpp"

#include "tensorView.hpp"
#include "arenaAllocator.hpp"
#include "taskQueue.hpp"

#include <numbers>
#include <arm_neon.h>

//////////////// Accepts output Variants ////////////////

void matMul2D_out(TensorView &A, TensorView &B, TensorView &O)
{
    auto o_shape = O.shape();
    auto a_shape = A.shape();

    auto numThreads = std::thread::hardware_concurrency();
    if (numThreads == 0)
    {
        // hardware_concurrency can sometimes return 0 when not implemented correctly.
        numThreads = 4; // fall back value
    }

    auto action = [&A, &B, &O, &a_shape, &o_shape](size_t startI, size_t endI)
    {
        for (size_t i = startI; i < endI; ++i)
        {
            for (size_t j = 0; j < a_shape[1]; ++j)
            {
                if (B.stride()[1] == 1)
                {
                    // continguous memory - fast path
                    size_t k = 0;
                    for (; k + 4 <= o_shape[1]; k += 4)
                    {
                        // duplicate a's values to all lanes
                        float32x4_t a_vec = vld1q_dup_f32(&A(i, j));

                        // load b's values to all lanes directly from contiguous memory
                        float32x4_t b_vec = vld1q_f32(&B(j, k));

                        if (j == 0)
                        {
                            // first time - so directly set the value
                            float32x4_t temp = vdupq_n_f32(0.0f);

                            // setting the value of 0 to the destination
                            vst1q_f32(&O(i, k), temp);
                        }

                        float32x4_t o_vec = vld1q_f32(&O(i, k));

                        o_vec = vmlaq_f32(o_vec, a_vec, b_vec);
                        vst1q_f32(&O(i, k), o_vec);
                    }

                    if (o_shape[1] % 4 != 0)
                    {
                        for (; k < o_shape[1]; ++k)
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
                else
                {
                    for (size_t k = 0; k < o_shape[1]; ++k)
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
    };

    auto totalIters = o_shape[0];
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

TensorView matMul2D(TensorView &A, TensorView &B, ArenaAllocator &alloc)
{
    auto a_shape = A.shape();
    auto b_shape = B.shape();

    // first ensure that matrix multiplication is possible
    assert(a_shape[1] == b_shape[0] && "Incompatible shapes were provided");

    std::vector<size_t> out_shape{a_shape[0], b_shape[1]};
    size_t bytesRequired = sizeof(float) * (out_shape[0] * out_shape[1]);

    TensorView result(alloc.alloc(bytesRequired), out_shape, 0);

    matMul2D_out(A, B, result);

    return result;
}

TensorView add2D(TensorView &A, TensorView &B, ArenaAllocator &alloc)
{
    auto a_shape = A.shape();
    auto b_shape = B.shape();

    // ensure that matrix addition is possible
    assert(a_shape[0] == b_shape[0] && a_shape[1] == b_shape[1] && "Provided matrices are not compatible for add operation");

    size_t bytesRequired = sizeof(float) * (a_shape[0] * a_shape[1]);

    TensorView result(alloc.alloc(bytesRequired), a_shape, 0);

    for (size_t i = 0; i < a_shape[0]; ++i)
    {
        for (size_t j = 0; j < a_shape[1]; ++j)
        {
            result(i, j) = A(i, j) + B(i, j);
        }
    }

    return result;
}

TensorView add1DTo2D(TensorView &A, TensorView &b, ArenaAllocator &alloc)
{
    // output shape will be same as A
    auto a_shape = A.shape();
    auto b_shape = b.shape();

    assert(a_shape[1] == b_shape[1] && "The columns are not equal");

    size_t bytesRequired = sizeof(float) * a_shape[0] * a_shape[1];
    TensorView result(alloc.alloc(bytesRequired), a_shape, 0);

    for (size_t i = 0; i < a_shape[0]; ++i)
    {
        for (size_t j = 0; j < a_shape[1]; ++j)
        {
            result(i, j) = A(i, j) + b(0, j);
        }
    }
    return result;
}

TensorView gelu(TensorView &A, ArenaAllocator &alloc)
{
    auto a_shape = A.shape();

    const float prefix = sqrt(2 / std::numbers::pi_v<float>);

    auto gelu_op = [&prefix](float x)
    {
        return 0.5 * x * (1.0 + tanh(prefix * (x + (0.044715 * x * x * x))));
    };

    size_t bytesRequired = sizeof(float) * (a_shape[0] * a_shape[1]);

    TensorView result(alloc.alloc(bytesRequired), a_shape, 0);

    for (size_t i = 0; i < a_shape[0]; ++i)
    {
        for (size_t j = 0; j < a_shape[1]; ++j)
        {
            result(i, j) = gelu_op(A(i, j));
        }
    }

    return result;
}

TensorView softmax(TensorView &A, ArenaAllocator &alloc)
{

    // for softmax - the input can be a matrix
    auto a_shape = A.shape();

    // output will be a row vector
    size_t bytesRequired = sizeof(float) * a_shape[0] * a_shape[1];

    // output tensor
    TensorView result(alloc.alloc(bytesRequired), a_shape, 0);

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
            result(i, j) = std::exp(A(i, j) - maxVal);
            totalSum += result(i, j);
        }

        for (size_t j = 0; j < a_shape[1]; ++j)
        {
            result(i, j) /= totalSum;
        }
    }

    return result;
}

TensorView layerNorm(TensorView &X, TensorView &weight, TensorView &bias, ArenaAllocator &alloc)
{
    auto out_shape = X.shape();

    size_t bytesRequired = sizeof(float) * out_shape[0] * out_shape[1];

    // prepare the output.
    TensorView result(alloc.alloc(bytesRequired), out_shape, 0);

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
            result(i, j) = weight(0, j) * normalizedVal + bias(0, j);
        }
    }
    return result;
}

// This doesn't allocate memory but returns a new TensorView
TensorView transpose2D(TensorView &X)
{
    auto x_shape = X.shape();
    auto x_stride = X.stride();

    std::reverse(x_shape.begin(), x_shape.end());
    std::reverse(x_stride.begin(), x_stride.end());

    return TensorView(X.get(), x_shape, x_stride, 0);
}

//////////////// Inplace Variants ////////////////

void add2D_inplace(TensorView &A, TensorView &B)
{
    auto a_shape = A.shape();
    auto b_shape = B.shape();

    // ensure that matrix addition is possible
    assert(a_shape[0] == b_shape[0] && a_shape[1] == b_shape[1] && "Provided matrices are not compatible for add operation");
    for (size_t i = 0; i < a_shape[0]; ++i)
    {
        for (size_t j = 0; j < a_shape[1]; ++j)
        {
            A(i, j) += B(i, j);
        }
    }
}

void add1DTo2D_inplace(TensorView &A, TensorView &b)
{
    auto a_shape = A.shape();
    auto b_shape = b.shape();

    assert(a_shape[1] == b_shape[1] && "The columns are not equal");

    for (size_t i = 0; i < a_shape[0]; ++i)
    {
        for (size_t j = 0; j < a_shape[1]; ++j)
        {
            A(i, j) += b(0, j);
        }
    }
}

void gelu_inplace(TensorView &A)
{
    auto a_shape = A.shape();

    const float prefix = sqrt(2 / std::numbers::pi_v<float>);

    auto gelu_op = [&prefix](float x)
    {
        return 0.5 * x * (1.0 + tanh(prefix * (x + (0.044715 * x * x * x))));
    };

    for (size_t i = 0; i < a_shape[0]; ++i)
    {
        for (size_t j = 0; j < a_shape[1]; ++j)
        {
            A(i, j) = gelu_op(A(i, j));
        }
    }
}

void softmax_inplace(TensorView &A)
{
    // for softmax - the input can be a matrix
    auto a_shape = A.shape();

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
            A(i, j) = std::exp(A(i, j) - maxVal);
            totalSum += A(i, j);
        }

        for (size_t j = 0; j < a_shape[1]; ++j)
        {
            A(i, j) /= totalSum;
        }
    }
}

void eleMul2D_inplace(TensorView &X, float val)
{
    auto x_shape = X.shape();
    for (size_t i = 0; i < x_shape[0]; ++i)
    {
        for (size_t j = 0; j < x_shape[1]; ++j)
        {
            X(i, j) *= val;
        }
    }
    return;
}