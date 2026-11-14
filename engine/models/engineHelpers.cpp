
#include "engineHelpers.hpp"
#include "GPT2Model.hpp"
#include "modelWeightsLoader.hpp"

#include <fstream>
#include <iostream>

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
