#pragma once

#include <nlohmann/json.hpp>
#include "OperationNode.hpp"

struct GPT2Model;
class ModelWeightMap;

using json = nlohmann::json;
json loadModelConfig(const char *filePath);

GPT2Model loadWeights(const ModelWeightMap &modelWeights, const json &modelConfig);

// In theory, build model should accept an additional argument - model name or model id (unique identifier)
// Based on the modelId provided, build model will use its own model architecture information
// along with the mapped weights and provided config to build a DAG and return vector of starting Operations
std::vector<OperationNode> buildModelGraph(std::vector<float> &prompt, const GPT2Model &modelWeights, bool prefillPhase);