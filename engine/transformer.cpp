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

    std::vector<size_t> output_shape{Q.shape()[0], K_trans.shape()[1]};
    size_t bytesRequired = sizeof(float) * output_shape[0] * output_shape[1];
    TensorView out(alloc.alloc(bytesRequired), output_shape, 0);
    matMul2D_out(Q, K_trans, out);

    // we should perform an element wise division with sqrt(colSize)
    // here colSize is 64 -> sqrt(64) = 8
    // Dividing with 8 is equal to multiplying with 0.125
    eleMul2D_out(out, 0.125f, out);

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
    std::vector<size_t> out_shape{normX.shape()[0], attnWeight.shape()[1]};
    size_t bytesRequired = sizeof(float) * out_shape[0] * out_shape[1];
    TensorView qkv(alloc.alloc(bytesRequired), out_shape, 0);
    matMul2D_out(normX, attnWeight, qkv);

    // in-place bias add
    add1DTo2D_out(qkv, attnBias, qkv);

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
    bytesRequired = sizeof(float) * N * colSize;
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

        // in-place softmax
        softmax_out(out, out);

        matMul2D_out(out, Vh, Oh);
    }

    out_shape[0] = O.shape()[0];
    out_shape[1] = attnProjWeight.shape()[1];
    bytesRequired = sizeof(float) * out_shape[0] * out_shape[1];
    TensorView outAttended(alloc.alloc(bytesRequired), out_shape, 0);

    matMul2D_out(O, attnProjWeight, outAttended);

    // in-place projection bias add
    add1DTo2D_out(outAttended, attnProjBias, outAttended);

    return outAttended;
}

TensorView forwardMLP(TensorView &attn, TensorView &fcWeight, TensorView &fcBias, TensorView &projWeight, TensorView &projBias, ArenaAllocator &alloc)
{
    std::vector<size_t> fc_shape{attn.shape()[0], fcWeight.shape()[1]};
    size_t fcBytes = sizeof(float) * fc_shape[0] * fc_shape[1];
    TensorView out(alloc.alloc(fcBytes), fc_shape, 0);
    matMul2D_out(attn, fcWeight, out);

    // in-place bias add
    add1DTo2D_out(out, fcBias, out);

    // in-place GELU
    gelu_out(out, out);

    std::vector<size_t> proj_shape{out.shape()[0], projWeight.shape()[1]};
    size_t projBytes = sizeof(float) * proj_shape[0] * proj_shape[1];
    TensorView outProj(alloc.alloc(projBytes), proj_shape, 0);
    matMul2D_out(out, projWeight, outProj);

    // in-place projection bias add
    add1DTo2D_out(outProj, projBias, outProj);

    return outProj;
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
    softmax_out(X, X);

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