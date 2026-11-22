#pragma once

#include <cstddef>
#include <cstdint>

enum class OpType : uint8_t
{
    Embedding,  // Token + Positional embedding lookup: O = WTE[tok] + WPE[pos]
    MatMul,     // Matrix multiplication: O = A * B
    Add,        // Element-wise matrix addition: O = A + B
    BiasAdd,    // 1D row broadcasting addition: O = A + b
    GELU,       // Gaussian Error Linear Unit activation: O = GELU(A)
    Softmax,    // Row-wise exponential normalization: O = Softmax(A)
    LayerNorm,  // Layer normalization: O = LayerNorm(X, weight, bias)
    EleMul,     // Scalar scaling: O = X * val
    CausalMask, // Upper-triangular autoregressive attention mask: S(i, j) = -1e9
};

// ============================================================================
// Baked Invariant Attribute Schemas
// ============================================================================

struct EmbeddingAttrs
{
    bool prefillPhase = false;
};

struct LayerNormAttrs
{
    float eps = 1e-5f;
};

struct EleMulAttrs
{
    float val = 1.0f;
};

struct SoftmaxAttrs
{
    int axis = -1;
};

struct CausalMaskAttrs
{
    size_t cachePos = 0;
};

struct MatMulAttrs
{
    bool transposeA = false;
    bool transposeB = false;
};

// Union of baked operator attributes (POD, fixed-size, zero heap allocation)
union OpAttrs
{
    EmbeddingAttrs embed;
    LayerNormAttrs layerNorm;
    EleMulAttrs eleMul;
    SoftmaxAttrs softmax;
    CausalMaskAttrs causalMask;
    MatMulAttrs matMul;

    OpAttrs() : layerNorm{} {}
};

// Helper for debugging, logging, and DAG visualization
inline const char *opTypeToString(OpType type)
{
    switch (type)
    {
    case OpType::Embedding:
        return "Embedding";
    case OpType::MatMul:
        return "MatMul";
    case OpType::Add:
        return "Add";
    case OpType::BiasAdd:
        return "BiasAdd";
    case OpType::GELU:
        return "GELU";
    case OpType::Softmax:
        return "Softmax";
    case OpType::LayerNorm:
        return "LayerNorm";
    case OpType::EleMul:
        return "EleMul";
    case OpType::CausalMask:
        return "CausalMask";
    }
}
