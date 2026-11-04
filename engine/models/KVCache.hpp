#pragma once

class TensorView;

#include <vector>

class KVCache
{
private:
    std::vector<float> data_;
    size_t maxSeqLen_;
    size_t numLayers_;
    size_t embedSize_;
    std::vector<size_t> strides_;

public:
    KVCache(size_t max_seq_len, size_t num_layers, size_t embed_size);

    void appendK(size_t layerIdx, size_t currSeqLen, TensorView &kNew);

    void appendV(size_t layerIdx, size_t currSeqLen, TensorView &vNew);

    TensorView getK(size_t layerIdx, size_t totalSeqLen);

    TensorView getV(size_t layerIdx, size_t totalSeqLen);
};