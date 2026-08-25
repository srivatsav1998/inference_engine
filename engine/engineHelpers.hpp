#pragma once

#include <nlohmann/json.hpp>

struct GPT2Model;
class ModelWeightsLoader;

using json = nlohmann::json;
json loadModelConfig(const char *filePath);

GPT2Model buildModel(const ModelWeightsLoader &modelWeights, const char *configFilePath);