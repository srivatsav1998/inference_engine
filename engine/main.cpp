
#include "modelWeightsLoader.hpp"
#include "tensorView.hpp"
#include "GPT2Model.hpp"
#include "arenaAllocator.hpp"
#include "mathOps.hpp"
#include "transformer.hpp"

#include <nlohmann/json.hpp>
#include <iostream>
#include <fstream>

constexpr const char *MODEL_WEIGHTS_FILE_PATH = "/Users/srivatsavg/Desktop/workspace/projects/inference_engine/model_convertor/model_weights.bin";
constexpr const char *MODEL_CONFIG_FILE_PATH = "/Users/srivatsavg/Desktop/workspace/projects/inference_engine/model_convertor/model_offsets.json";

using json = nlohmann::json;

json loadModelConfig(const char *filePath)
{
    std::ifstream configFile(filePath);
    if (!configFile.is_open())
    {
        std::cerr << "Failed to open config.json" << std::endl;
        throw std::runtime_error("Failed to open config.json");
    }

    return json::parse(configFile);
}

GPT2Model buildModel(const ModelWeightsLoader &modelWeights, const char *configFilePath)
{
    GPT2Model model;

    auto modelConfig = loadModelConfig(configFilePath);

    // Token and positional embeddings;
    json wte_data = modelConfig["wte.weight"];
    json wpe_data = modelConfig["wpe.weight"];
    model.wte_weight = TensorView(*modelWeights, wte_data["shape"], wte_data["byte_offset"]);
    model.wpe_weight = TensorView(*modelWeights, wpe_data["shape"], wpe_data["byte_offset"]);

    // Transformer layers
    model.layers.resize(12);
    for (int i = 0; i < 12; i++)
    {
        // Attention LayerNorm
        json ln_1_weight = modelConfig["h." + std::to_string(i) + ".ln_1.weight"];
        json ln_1_bias = modelConfig["h." + std::to_string(i) + ".ln_1.bias"];
        model.layers[i].ln_1_weight = TensorView(*modelWeights, {1, ln_1_weight["shape"][0]}, ln_1_weight["byte_offset"]);
        model.layers[i].ln_1_bias = TensorView(*modelWeights, {1, ln_1_bias["shape"][0]}, ln_1_bias["byte_offset"]);

        // Attention QKV & Projection
        json attn_c_attn_weight = modelConfig["h." + std::to_string(i) + ".attn.c_attn.weight"];
        json attn_c_attn_bias = modelConfig["h." + std::to_string(i) + ".attn.c_attn.bias"];
        json attn_c_proj_weight = modelConfig["h." + std::to_string(i) + ".attn.c_proj.weight"];
        json attn_c_proj_bias = modelConfig["h." + std::to_string(i) + ".attn.c_proj.bias"];
        model.layers[i].attn_c_attn_weight = TensorView(*modelWeights, attn_c_attn_weight["shape"], attn_c_attn_weight["byte_offset"]);
        model.layers[i].attn_c_attn_bias = TensorView(*modelWeights, {1, attn_c_attn_bias["shape"][0]}, attn_c_attn_bias["byte_offset"]);
        model.layers[i].attn_c_proj_weight = TensorView(*modelWeights, attn_c_proj_weight["shape"], attn_c_proj_weight["byte_offset"]);
        model.layers[i].attn_c_proj_bias = TensorView(*modelWeights, {1, attn_c_proj_bias["shape"][0]}, attn_c_proj_bias["byte_offset"]);

        // LayerNorm
        json ln_2_weight = modelConfig["h." + std::to_string(i) + ".ln_2.weight"];
        json ln_2_bias = modelConfig["h." + std::to_string(i) + ".ln_2.bias"];
        model.layers[i].ln_2_weight = TensorView(*modelWeights, {1, ln_2_weight["shape"][0]}, ln_2_weight["byte_offset"]);
        model.layers[i].ln_2_bias = TensorView(*modelWeights, {1, ln_2_bias["shape"][0]}, ln_2_bias["byte_offset"]);

        // Feed forward neural network
        json mlp_c_fc_weight = modelConfig["h." + std::to_string(i) + ".mlp.c_fc.weight"];
        json mlp_c_fc_bias = modelConfig["h." + std::to_string(i) + ".mlp.c_fc.bias"];
        json mlp_c_proj_weight = modelConfig["h." + std::to_string(i) + ".mlp.c_proj.weight"];
        json mlp_c_proj_bias = modelConfig["h." + std::to_string(i) + ".mlp.c_proj.bias"];
        model.layers[i].mlp_c_fc_weight = TensorView(*modelWeights, mlp_c_fc_weight["shape"], mlp_c_fc_weight["byte_offset"]);
        model.layers[i].mlp_c_fc_bias = TensorView(*modelWeights, {1, mlp_c_fc_bias["shape"][0]}, mlp_c_fc_bias["byte_offset"]);
        model.layers[i].mlp_c_proj_weight = TensorView(*modelWeights, mlp_c_proj_weight["shape"], mlp_c_proj_weight["byte_offset"]);
        model.layers[i].mlp_c_proj_bias = TensorView(*modelWeights, {1, mlp_c_proj_bias["shape"][0]}, mlp_c_proj_bias["byte_offset"]);
    }

    // Final Layer Norm
    json ln_f_data = modelConfig["ln_f.weight"];
    json ln_f_bias_data = modelConfig["ln_f.bias"];
    model.ln_f_weight = TensorView(*modelWeights, {1, ln_f_data["shape"][0]}, ln_f_data["byte_offset"]);
    model.ln_f_bias = TensorView(*modelWeights, {1, ln_f_bias_data["shape"][0]}, ln_f_bias_data["byte_offset"]);

    return model;
}

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

void infer(std::vector<size_t> &prompt, GPT2Model &model, ArenaAllocator &alloc)
{
    auto max_new_tokens = 5;
    for (int tokens = 0; tokens < max_new_tokens; tokens++)
    {
        TensorView X = embedTokens(prompt, model.wpe_weight, model.wte_weight, 0, alloc);

        // attention part
        for (int i = 0; i < 12; ++i)
        {
            auto normX = layerNorm(X, model.layers[i].ln_1_weight, model.layers[i].ln_1_bias, alloc);
            auto attn_out = forwardAttention(normX, model.layers[i].attn_c_attn_weight, model.layers[i].attn_c_attn_bias, model.layers[i].attn_c_proj_weight, model.layers[i].attn_c_proj_bias, alloc);

            // residual connection
            add2D_inplace(X, attn_out);

            normX = layerNorm(X, model.layers[i].ln_2_weight, model.layers[i].ln_2_bias, alloc);
            auto mlp_out = forwardMLP(normX, model.layers[i].mlp_c_fc_weight, model.layers[i].mlp_c_fc_bias, model.layers[i].mlp_c_proj_weight, model.layers[i].mlp_c_proj_bias, alloc);

            // residual connection
            add2D_inplace(X, mlp_out);
        }

        // final layerNormalization
        auto out = layerNorm(X, model.ln_f_weight, model.ln_f_bias, alloc);

        // un-embedding projection
        auto wte_trans = transpose2D(model.wte_weight);
        out = matMul2D(out, wte_trans, alloc);

        // extracting next word
        auto tokenId = extractNextTokenId(out);
        prompt.push_back(tokenId);

        // resetting memory
        alloc.reset();
    }
}

int main()
{
    // Maps the model weights as memory for program to directly read
    ModelWeightsLoader modelWeights(MODEL_WEIGHTS_FILE_PATH);
    ArenaAllocator arena(9000000); // 9 MB of memory

    GPT2Model model;
    try
    {
        model = buildModel(modelWeights, MODEL_CONFIG_FILE_PATH);
    }
    catch (const std::exception &e)
    {
        std::cout << e.what() << std::endl;
        return 1;
    }

    // 1. Create a dummy sequence of 4 token IDs.
    // In GPT-2, token 15496 is "This", 318 is " is", 257 is " a", and 1332 is " test".
    std::vector<size_t> dummy_prompt = {15496, 318, 257, 1332};

    infer(dummy_prompt, model, arena);

    for (auto token : dummy_prompt)
    {
        std::cout << token << " ";
    }

    std::cout << std::endl;

    return 0;
}
