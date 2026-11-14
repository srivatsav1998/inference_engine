#pragma once

#include "tensorView.hpp"

#define EMBEDDING_SIZE 768

struct TransformerBlockWeights
{
    // Layer Normalization
    TensorView ln_1_weight;
    TensorView ln_1_bias;

    // Attention QKV & Projection
    TensorView attn_c_attn_weight;
    TensorView attn_c_attn_bias;
    TensorView attn_c_proj_weight;
    TensorView attn_c_proj_bias;

    // Layer Normalization
    TensorView ln_2_weight;
    TensorView ln_2_bias;

    // Feed forward neural network
    TensorView mlp_c_fc_weight;
    TensorView mlp_c_fc_bias;
    TensorView mlp_c_proj_weight;
    TensorView mlp_c_proj_bias;
};

struct GPT2Model
{
    // Token and positional embeddings
    TensorView wte_weight;
    TensorView wpe_weight;

    // Transformer Layers
    std::vector<TransformerBlockWeights> layers;

    // Final Layer Normalization
    TensorView ln_f_weight;
    TensorView ln_f_bias;
};