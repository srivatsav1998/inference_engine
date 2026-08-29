#include "transformer.hpp"

#include "mathOps.hpp"
#include "KVCache.hpp"
#include "arenaAllocator.hpp"
#include "tensorView.hpp"

#include <queue>
#include <random>

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

void applyTempScaling_inplace(TensorView &X, float temperature)
{
    auto x_shape = X.shape();
    if (temperature != 0.0f)
    {
        for (size_t j = 0; j < x_shape[1]; ++j)
        {
            X(x_shape[0] - 1, j) /= temperature;
        }
    }
}

// returns a vector of pairs containing the top K indices and their corresponding values from the last row of X, sorted in descending order of value
std::vector<std::pair<float, size_t>> getTopKIndices(TensorView &X, size_t topK)
{
    // using a min-heap to keep exactly K elements - each entry has probability and the index
    std::priority_queue<std::pair<float, size_t>, std::vector<std::pair<float, size_t>>, std::greater<std::pair<float, size_t>>> topKHeap;
    auto x_shape = X.shape();
    for (size_t j = 0; j < x_shape[1]; ++j)
    {
        topKHeap.push({X(x_shape[0] - 1, j), j});
        if (topKHeap.size() > topK)
        {
            topKHeap.pop();
        }
    }
    std::vector<std::pair<float, size_t>> result;
    while (!topKHeap.empty())
    {
        result.push_back(topKHeap.top());
        topKHeap.pop();
    }

    std::sort(result.begin(), result.end(), std::greater<std::pair<float, size_t>>());
    return result;
}

std::vector<size_t> applyTopPFiltering(const std::vector<std::pair<float, size_t>> &topKVec, float topP)
{
    // consider only till we reach the top-p threshold
    float cumulativeProb = 0.0f;
    std::vector<size_t> keptIndices;
    for (const auto &entry : topKVec)
    {
        cumulativeProb += entry.first;
        keptIndices.push_back(entry.second);
        if (cumulativeProb >= topP)
        {
            break;
        }
    }
    return keptIndices;
}

size_t sampleTokenIdx(TensorView &X, float temperature, size_t topK, float topP)
{
    // first perform temperature scaling on the last row of X
    applyTempScaling_inplace(X, temperature);

    // convert the logits to probabilities
    // TODO: Add verSplit to TensorView to avoid applying softmax to all rows during the prefill phase
    softmax_inplace(X);

    // apply top-k filtering
    auto topKVec = getTopKIndices(X, topK);

    // apply top-p filtering
    auto keptIndices = applyTopPFiltering(topKVec, topP);

    // the new max might not be 1.0f after filtering
    float sumProb = 0.0f;
    for (const auto &idx : keptIndices)
    {
        sumProb += X(X.shape()[0] - 1, idx);
    }

    // perform sampling from the kept indices
    std::random_device rd;
    std::mt19937 gen(rd());

    std::uniform_real_distribution<float> dis(0.0f, sumProb);

    float tar = dis(gen);

    float cumulativeProb = 0.0f;
    for (size_t i = 0; i < keptIndices.size(); ++i)
    {
        cumulativeProb += X(X.shape()[0] - 1, keptIndices[i]);
        if (tar <= cumulativeProb)
        {
            return keptIndices[i];
        }
    }

    return keptIndices.back();
}