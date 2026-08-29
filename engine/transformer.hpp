
#pragma once

#include <cstddef>

class TensorView;
class ArenaAllocator;
class KVCache;

TensorView attentionScore(TensorView &Q, TensorView &K, ArenaAllocator &alloc);

void causalMasking(TensorView &S, size_t cachePos);

TensorView forwardAttention(TensorView &normX, TensorView &attnWeight, TensorView &attnBias, TensorView &attnProjWeight, TensorView &attnProjBias, ArenaAllocator &alloc, KVCache &cache, size_t layerIdx, size_t cachePos);

TensorView forwardMLP(TensorView &attn, TensorView &fcWeight, TensorView &fcBias, TensorView &projWeight, TensorView &projBias, ArenaAllocator &alloc);

size_t extractNextTokenId(TensorView &X);

size_t sampleTokenIdx(TensorView &X, float temperature = 1.2f, size_t topK = 10, float topP = 0.9f);
