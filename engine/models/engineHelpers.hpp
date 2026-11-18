#pragma once

#include <nlohmann/json.hpp>

struct GPT2Model;
class ModelWeightMap;

using json = nlohmann::json;
json loadModelConfig(const char *filePath);

GPT2Model buildModel(const ModelWeightMap &modelWeights, const json &modelConfig);