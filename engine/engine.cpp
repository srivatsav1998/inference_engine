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

void Engine::infer(std::vector<size_t> &prompt, unsigned int *response, unsigned int maxTokens, float temperature, size_t topK, float topP)
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

        size_t bytesRequired = sizeof(float) * X.shape()[0] * X.shape()[1];
        TensorView normX(arena_.alloc(bytesRequired), X.shape(), 0);

        // attention part
        for (size_t layerIdx = 0; layerIdx < model_.layers.size(); ++layerIdx)
        {
            auto layer = model_.layers[layerIdx];

            layerNorm_out(X, layer.ln_1_weight, layer.ln_1_bias, normX);

            auto attn_out = forwardAttention(normX, layer.attn_c_attn_weight, layer.attn_c_attn_bias, layer.attn_c_proj_weight, layer.attn_c_proj_bias, arena_, cache, layerIdx, cachePos);

            // in-place residual connection
            add2D_out(X, attn_out, X);

            layerNorm_out(X, layer.ln_2_weight, layer.ln_2_bias, normX);
            auto mlp_out = forwardMLP(normX, layer.mlp_c_fc_weight, layer.mlp_c_fc_bias, layer.mlp_c_proj_weight, layer.mlp_c_proj_bias, arena_);

            // in-place residual connection
            add2D_out(X, mlp_out, X);
        }

        // final layerNormalization (reusing normX buffer)
        layerNorm_out(X, model_.ln_f_weight, model_.ln_f_bias, normX);

        // un-embedding projection
        auto wte_trans = transpose2D(model_.wte_weight);

        std::vector<size_t> out_shape{normX.shape()[0], wte_trans.shape()[1]};
        bytesRequired = sizeof(float) * (out_shape[0] * out_shape[1]);

        TensorView result(arena_.alloc(bytesRequired), out_shape, 0);
        matMul2D_out(normX, wte_trans, result);

        // extracting next word
        size_t tokenId;

        float diff = std::abs(temperature - 0.0f);
        if (diff < 1e-6f)
        {
            tokenId = extractNextTokenId(result);
        }
        else
        {
            tokenId = sampleTokenIdx(result, temperature, topK, topP);
        }

        response[respIdx++] = tokenId;
        prompt.push_back(tokenId);

        // resetting memory
        arena_.reset();

        // updating the cache position
        cachePos += X.shape()[0];
    }
}