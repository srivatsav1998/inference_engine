#include "engine.hpp"
#include "arenaAllocator.hpp"
#include "KVCache.hpp"
#include "mathOps.hpp"
#include "engineHelpers.hpp"
#include "GPT2Model.hpp"
#include "transformer.hpp"

TensorView embedTokens(std::vector<size_t> tokenIds, TensorView &wpe, TensorView &wte, int absStartPos, ArenaAllocator &alloc)
{
    // output shape
    std::vector<size_t> res_shape{tokenIds.size(), EMBEDDING_SIZE};

    size_t bytesRequired = res_shape[0] * res_shape[1] * sizeof(float);

    TensorView result(alloc.alloc(bytesRequired), res_shape, 0);

    for (size_t i = 0; i < res_shape[0]; ++i)
    {
        auto token = tokenIds[i];
        for (size_t j = 0; j < res_shape[1]; ++j)
        {
            result(i, j) = wte(token, j) + wpe(i + absStartPos, j);
        }
    }

    return result;
}

Engine::Engine(size_t arenaSize, const char *modelWeightsPath, const char *modelConfigPath) : loader_(modelWeightsPath), arena_(arenaSize * 1e6)
{
    model_ = buildModel(loader_, modelConfigPath);
}

void Engine::infer(std::vector<size_t> &prompt, unsigned int *response, unsigned int maxTokens)
{
    // copy contents of prompt to response
    auto respIdx = 0;
    for (const auto &wordT : prompt)
    {
        response[respIdx++] = wordT;
    }

    KVCache cache(maxTokens, model_.layers.size(), EMBEDDING_SIZE);
    size_t currSize = prompt.size() - 1;
    size_t cachePos = 0;

    // Loop runs exactly enough times to reach maxTokens total length
    for (size_t token = currSize; token < maxTokens - 1; token++)
    {
        // token indicates the id of the last token in prompt
        TensorView X;

        if (token == currSize)
        {
            // prefill phase
            X = embedTokens(prompt, model_.wpe_weight, model_.wte_weight, 0, arena_);
        }
        else
        {
            // decoder phase
            // passing in token as absStartPos
            X = embedTokens({prompt[token]}, model_.wpe_weight, model_.wte_weight, token, arena_);
        }

        // attention part
        for (size_t layerIdx = 0; layerIdx < model_.layers.size(); ++layerIdx)
        {
            auto layer = model_.layers[layerIdx];

            auto normX = layerNorm(X, layer.ln_1_weight, layer.ln_1_bias, arena_);
            auto attn_out = forwardAttention(normX, layer.attn_c_attn_weight, layer.attn_c_attn_bias, layer.attn_c_proj_weight, layer.attn_c_proj_bias, arena_, cache, layerIdx, cachePos);

            // residual connection
            add2D_inplace(X, attn_out);

            normX = layerNorm(X, layer.ln_2_weight, layer.ln_2_bias, arena_);
            auto mlp_out = forwardMLP(normX, layer.mlp_c_fc_weight, layer.mlp_c_fc_bias, layer.mlp_c_proj_weight, layer.mlp_c_proj_bias, arena_);

            // residual connection
            add2D_inplace(X, mlp_out);
        }

        // final layerNormalization
        auto out = layerNorm(X, model_.ln_f_weight, model_.ln_f_bias, arena_);

        // un-embedding projection
        auto wte_trans = transpose2D(model_.wte_weight);
        out = matMul2D(out, wte_trans, arena_);

        // extracting next word
        auto tokenId = extractNextTokenId(out);
        response[respIdx++] = tokenId;
        prompt.push_back(tokenId);

        // resetting memory
        arena_.reset();

        // updating the cache position
        cachePos += X.shape()[0];
    }
}