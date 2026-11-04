#include "mathOps.hpp"

#include "tensorView.hpp"
#include "arenaAllocator.hpp"
#include "taskQueue.hpp"

#include <numbers>
#include <arm_neon.h>

//////////////// Accepts output Variants ////////////////

void matMul2D_out(const TensorView &A, const TensorView &B, TensorView &O)
{
    assert(O.get() != A.get() && O.get() != B.get() && "matMul2D_out: output buffer cannot alias input buffers");

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

// This doesn't allocate memory but returns a new TensorView
TensorView transpose2D(TensorView &X)
{
    auto x_shape = X.shape();
    auto x_stride = X.stride();

    std::reverse(x_shape.begin(), x_shape.end());
    std::reverse(x_stride.begin(), x_stride.end());

    return TensorView(X.get(), x_shape, x_stride, 0);
}
