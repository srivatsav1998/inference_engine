#pragma once

#include "modelWeightsLoader.hpp"
#include "arenaAllocator.hpp"
#include "GPT2Model.hpp"

#include <vector>

class Engine
{
public:
    Engine(size_t arenaSize, const char *modelWeightsPath, const char *modelConfigPath);

    void infer(std::vector<size_t> &prompt, unsigned int *reponse, unsigned int maxTokens, float temperature, size_t topK, float topP);

private:
    ModelWeightsLoader loader_;
    ArenaAllocator arena_;
    GPT2Model model_;
};