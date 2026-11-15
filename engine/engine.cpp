#include "engine.hpp"
#include "arenaAllocator.hpp"
#include "KVCache.hpp"
#include "mathOps.hpp"
#include "engineHelpers.hpp"
#include "GPT2Model.hpp"
#include "transformer.hpp"

#include <algorithm>
#include <cassert>
#include <stdexcept>
#include <string>

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

Engine::Engine(const char *modelWeightsPath, const char *modelConfigPath) : loader_(modelWeightsPath)
{
    model_ = buildModel(loader_, modelConfigPath);
}

size_t Engine::stepArenaBytes(size_t rows, size_t contextLen) const
{
    const auto &layer = model_.layers.front();
    const size_t D = EMBEDDING_SIZE;
    const size_t qkvCols = layer.attn_c_attn_weight.shape()[1];
    const size_t attnProjCols = layer.attn_c_proj_weight.shape()[1];
    const size_t fcCols = layer.mlp_c_fc_weight.shape()[1];
    const size_t mlpProjCols = layer.mlp_c_proj_weight.shape()[1];
    const size_t vocab = model_.wte_weight.shape()[0];

    auto block = [rows](size_t cols)
    { return ArenaAllocator::alignedSize(sizeof(float) * rows * cols); };

    // Per transformer layer. Nothing is freed until the end of the step, so layers add up.
    const size_t perLayer = block(qkvCols)                                  // qkv
                            + block(D)                                      // attention output O
                            + NUM_ATTENTION_HEADS * block(contextLen)       // scores, one per head
                            + block(attnProjCols)                           // attention projection
                            + block(fcCols)                                 // MLP hidden
                            + block(mlpProjCols);                           // MLP projection

    return block(D)                               // X (embeddings / residual stream)
           + block(D)                             // normX
           + model_.layers.size() * perLayer
           + block(vocab);                        // logits
}

void Engine::validateRequest(const std::vector<size_t> &prompt, unsigned int maxTokens) const
{
    const size_t contextLimit = model_.wpe_weight.shape()[0]; // 1024 positions for GPT-2
    const size_t vocab = model_.wte_weight.shape()[0];

    if (prompt.empty())
        throw std::invalid_argument("prompt must contain at least one token");
    if (maxTokens <= prompt.size())
        throw std::invalid_argument("max_tokens (" + std::to_string(maxTokens) + ") must be greater than the prompt length (" + std::to_string(prompt.size()) + ")");
    if (maxTokens > contextLimit)
        throw std::invalid_argument("max_tokens (" + std::to_string(maxTokens) + ") exceeds the model context of " + std::to_string(contextLimit));
    for (size_t token : prompt)
        if (token >= vocab)
            throw std::invalid_argument("token id " + std::to_string(token) + " is outside the vocabulary (" + std::to_string(vocab) + ")");
}

size_t Engine::infer(std::vector<size_t> &prompt, unsigned int *response, unsigned int maxTokens, float temperature, size_t topK, float topP)
{
    validateRequest(prompt, maxTokens);

    // Size the arena for this request before touching `response`. The peak step is either the
    // prefill (all prompt rows attending to each other) or the last decode step (one row
    // attending to the whole context). Capacity only grows, so repeated requests reuse it.
    const size_t promptLen = prompt.size();
    arena_.reserve(std::max(stepArenaBytes(promptLen, promptLen), stepArenaBytes(1, maxTokens - 1)));

    // copy contents of prompt to response
    size_t respIdx = 0;
    for (const auto &wordT : prompt)
    {
        response[respIdx++] = wordT;
    }

    KVCache cache(maxTokens, model_.layers.size(), EMBEDDING_SIZE);
    size_t idxOfLastToken = prompt.size() - 1;
    size_t cachePos = 0;

    // Loop runs exactly enough times to reach maxTokens total length
    for (size_t tokenIdx = idxOfLastToken; tokenIdx < maxTokens - 1; tokenIdx++)
    {
        // token indicates the id of the last token in prompt
        TensorView X;

        if (tokenIdx == idxOfLastToken)
        {
            // prefill phase
            X = embedTokens(prompt, model_.wpe_weight, model_.wte_weight, 0, arena_);
        }
        else
        {
            // decoder phase
            // passing in token as absStartPos
            X = embedTokens({prompt[tokenIdx]}, model_.wpe_weight, model_.wte_weight, tokenIdx, arena_);
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

        // un-embedding projection: normX [N, 768] * wte_weight^T [768, 50257] -> [N, 50257]
        std::vector<size_t> out_shape{normX.shape()[0], model_.wte_weight.shape()[0]};
        bytesRequired = sizeof(float) * (out_shape[0] * out_shape[1]);

        TensorView result(arena_.alloc(bytesRequired), out_shape, 0);
        matMul2D_out(normX, model_.wte_weight, result, /*transposeA=*/false, /*transposeB=*/true);

        // extracting next word
        size_t token;

        float diff = std::abs(temperature - 0.0f);
        if (diff < 1e-6f)
        {
            token = extractNextTokenId(result);
        }
        else
        {
            token = sampleTokenIdx(result, temperature, topK, topP);
        }

        response[respIdx++] = token;
        prompt.push_back(token);

        // The arena plan must match what this step really allocated (see stepArenaBytes).
        assert(arena_.bytesAllocated() == stepArenaBytes(X.shape()[0], cachePos + X.shape()[0]) && "stepArenaBytes is out of sync with infer's arena allocations");

        // resetting memory
        arena_.reset();

        // updating the cache position
        cachePos += X.shape()[0];

        if (token == END_OF_TEXT_TOKEN)
        {
            // End of text generation.
            break;
        }
    }

    return respIdx;
}
