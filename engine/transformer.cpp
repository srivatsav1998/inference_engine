#include "transformer.hpp"

#include "mathOps.hpp"
#include "KVCache.hpp"
#include "arenaAllocator.hpp"
#include "tensorView.hpp"

TensorView attentionScore(TensorView &Q, TensorView &K, ArenaAllocator &alloc)
{
    auto K_trans = transpose2D(K);
    auto out = matMul2D(Q, K_trans, alloc);

    // we should perform an element wise division with sqrt(colSize)
    // here colSize is 64 -> sqrt(64) = 8
    // Dividing with 8 is equal to multiplying with 0.125
    eleMul2D_inplace(out, 0.125);

    return out;
}

void causalMasking(TensorView &S, size_t cachePos)
{
    auto s_shape = S.shape();

    for (size_t i = 0; i < s_shape[0]; ++i)
    {
        for (size_t j = cachePos + i + 1; j < s_shape[1]; ++j)
        {
            S(i, j) = -1e9;
        }
    }
}

TensorView forwardAttention(TensorView &normX, TensorView &attnWeight, TensorView &attnBias, TensorView &attnProjWeight, TensorView &attnProjBias, ArenaAllocator &alloc, KVCache &cache, size_t layerIdx, size_t cachePos)
{
    auto qkv = matMul2D(normX, attnWeight, alloc);
    add1DTo2D_inplace(qkv, attnBias);

    auto qkv_shape = qkv.shape();

    auto N = qkv_shape[0];
    auto colSize = qkv_shape[1] / 3;

    auto Q = qkv.horSplit(0, colSize);
    auto K = qkv.horSplit(colSize, 2 * colSize);
    auto V = qkv.horSplit(2 * colSize, 3 * colSize);

    cache.appendK(layerIdx, cachePos, K);
    cache.appendV(layerIdx, cachePos, V);

    K = cache.getK(layerIdx, cachePos + K.shape()[0]);
    V = cache.getV(layerIdx, cachePos + V.shape()[0]);

    // split each of Q, K, and V matrices into 12 pieces for passing to 12 attention heads
    // each head will then compute attention score

    // Output matrix
    size_t bytesRequired = sizeof(float) * N * colSize;
    TensorView O(alloc.alloc(bytesRequired), {N, colSize}, 0);

    for (int i = 0; i < 12; ++i)
    {
        // compute Q, K, and V for this head
        auto headColSize = 64;
        auto Qh = Q.horSplit(i * headColSize, (i + 1) * headColSize);
        auto Kh = K.horSplit(i * headColSize, (i + 1) * headColSize);
        auto Vh = V.horSplit(i * headColSize, (i + 1) * headColSize);
        auto Oh = O.horSplit(i * headColSize, (i + 1) * headColSize);

        auto out = attentionScore(Qh, Kh, alloc);

        causalMasking(out, cachePos);

        softmax_inplace(out);
        matMul2D_out(out, Vh, Oh);
    }

    auto out = matMul2D(O, attnProjWeight, alloc);
    add1DTo2D_inplace(out, attnProjBias);

    return out;
}

TensorView forwardMLP(TensorView &attn, TensorView &fcWeight, TensorView &fcBias, TensorView &projWeight, TensorView &projBias, ArenaAllocator &alloc)
{
    auto out = matMul2D(attn, fcWeight, alloc);
    add1DTo2D_inplace(out, fcBias);

    gelu_inplace(out);

    out = matMul2D(out, projWeight, alloc);
    add1DTo2D_inplace(out, projBias);

    return out;
}

size_t extractNextTokenId(TensorView &X)
{

    // we only care about last row of X
    auto row = X.shape()[0] - 1;
    auto maxSoFar = X(row, 0);
    size_t maxIdx = 0;

    for (size_t j = 0; j < X.shape()[1]; ++j)
    {
        if (maxSoFar < X(row, j))
        {
            // this is current max
            maxIdx = j;
            maxSoFar = X(row, j);
        }
    }
    return maxIdx;
}
