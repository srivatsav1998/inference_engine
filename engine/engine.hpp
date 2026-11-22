#pragma once

#include "modelWeightMap.hpp"
#include "arenaAllocator.hpp"
#include "GPT2Model.hpp"

#include <vector>

class Engine
{
public:
    Engine(const char *modelWeightsPath, const char *modelConfigPath);

    // Writes the prompt followed by generated tokens into `response` (capacity: maxTokens) and
    // returns how many entries were written (<= maxTokens; fewer if <|endoftext|> is generated).
    // Throws std::invalid_argument for an out-of-range request and std::bad_alloc if the
    // activation arena cannot be grown; nothing is written to `response` in either case.
    size_t infer(std::vector<float> &prompt, unsigned int *response, unsigned int maxTokens, float temperature, size_t topK, float topP);

private:
    // Exact arena bytes one forward step uses: `rows` new tokens attending to `contextLen`
    // positions (cached + new). Mirrors every arena allocation in infer / forwardAttention /
    // forwardMLP; a debug-build assert in infer() fails if the two ever drift apart.
    [[nodiscard]] size_t stepArenaBytes(size_t rows, size_t contextLen) const;

    void validateRequest(const std::vector<float> &prompt, unsigned int maxTokens) const;

    ModelWeightMap loader_;
    json modelConfig_;
    ArenaAllocator arena_;
    GPT2Model weights_;
};
