
#include "KVCache.hpp"
#include "tensorView.hpp"

KVCache::KVCache(size_t max_seq_len, size_t num_layers, size_t embed_size) : maxSeqLen_(max_seq_len), numLayers_(num_layers), embedSize_(embed_size)
{
    // reserving the required space
    data_.resize(numLayers_ * 2 * maxSeqLen_ * embedSize_);
    strides_ = {2 * maxSeqLen_ * embedSize_, maxSeqLen_ * embedSize_, embedSize_, 1};
}

void KVCache::appendK(size_t layerIdx, size_t currSeqLen, TensorView &kNew)
{

    size_t ptrIdx = strides_[0] * layerIdx + strides_[1] * 0 /* K */ + strides_[2] * currSeqLen;

    auto k_shape = kNew.shape();
    for (size_t i = 0; i < k_shape[0]; ++i)
    {
        for (size_t j = 0; j < k_shape[1]; ++j)
        {
            data_[ptrIdx++] = kNew(i, j);
        }
    }
}

void KVCache::appendV(size_t layerIdx, size_t currSeqLen, TensorView &vNew)
{

    size_t ptrIdx = strides_[0] * layerIdx + strides_[1] * 1 /* V */ + strides_[2] * currSeqLen;

    auto v_shape = vNew.shape();
    for (size_t i = 0; i < v_shape[0]; ++i)
    {
        for (size_t j = 0; j < v_shape[1]; ++j)
        {
            data_[ptrIdx++] = vNew(i, j);
        }
    }
}

TensorView KVCache::getK(size_t layerIdx, size_t totalSeqLen)
{
    std::vector<size_t> out_shape = {totalSeqLen, embedSize_};

    size_t ptrIdx = strides_[0] * layerIdx + strides_[1] * 0 /* K */;

    TensorView out(&data_[ptrIdx], out_shape, 0);
    return out;
}

TensorView KVCache::getV(size_t layerIdx, size_t totalSeqLen)
{
    std::vector<size_t> out_shape = {totalSeqLen, embedSize_};

    size_t ptrIdx = strides_[0] * layerIdx + strides_[1] * 1 /* V */;

    TensorView out(&data_[ptrIdx], out_shape, 0);
    return out;
}
