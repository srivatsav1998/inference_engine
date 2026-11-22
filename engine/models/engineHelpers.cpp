
#include "engineHelpers.hpp"
#include "GPT2Model.hpp"
#include "modelWeightMap.hpp"

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

GPT2Model loadWeights(const ModelWeightMap &modelWeights, const json &modelConfig)
{
    GPT2Model model;

    // Token and positional embeddings;
    json wte_data = modelConfig["wte.weight"];
    json wpe_data = modelConfig["wpe.weight"];
    model.wte_weight = TensorView(*modelWeights, wte_data["shape"], wte_data["byte_offset"]);
    model.wpe_weight = TensorView(*modelWeights, wpe_data["shape"], wpe_data["byte_offset"]);

    // Transformer layers
    model.layers.resize(NUM_ATTENTION_HEADS);
    for (int i = 0; i < NUM_ATTENTION_HEADS; i++)
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

OperationNode addEmbeddingLayer(TensorView &input_t, TensorView &wte_t, TensorView &wpe_t, std::shared_ptr<ValueNode> &out, bool prefillPhase)
{
    auto input = std::make_shared<ValueNode>(input_t);
    auto wte = std::make_shared<ValueNode>(wte_t);
    auto wpe = std::make_shared<ValueNode>(wpe_t);

    Values op_inputs{input, wte, wpe};
    Values op_outs{out};

    OpAttrs attrs;
    attrs.embed.prefillPhase = prefillPhase;

    return OperationNode(op_inputs, op_inputs, OpType::Embedding, attrs);
}

std::vector<OperationNode> buildModelGraph(std::vector<float> &prompt, const GPT2Model &modelWeights, bool prefillPhase)
{
    /////////////////  EMBEDDDING LAYER /////////////////
    // tensors
    auto wte_t = std::make_shared<TensorView>(modelWeights.wte_weight);
    auto wpe_t = std::make_shared<TensorView>(modelWeights.wpe_weight);
    auto input_t = std::make_shared<TensorView>(prompt.data(), std::vector<size_t>{1, prompt.size()}, 0);

    auto embedding

        // ValueNodes
        auto input = std::make_shared<ValueNode>(input_t);
    auto wte = std::make_shared<ValueNode>(wte_t);
    auto wpe = std::make_shared<ValueNode>(wpe_t);

    Values op_inputs{input, wte, wpe};

    TensorShape embeddings_out_shape{prompt.size(), EMBEDDING_SIZE};
    auto bytesRequired = sizeof(float) * embeddings_out_shape[0] * embeddings_out_shape[1];

    auto embeddings_out = std::make_shared<ValueNode>(bytesRequired, embeddings_out_shape);
    Values op_outs{embeddings_out};
    OpAttrs attrs;
    attrs.embed.prefillPhase = prefillPhase;
    OperationNode embedding(op_inputs, op_outs, OpType::Embedding, attrs);
}
