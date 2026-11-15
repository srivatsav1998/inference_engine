
#include "engineAPI.hpp"
#include "engine.hpp"

#include <iostream>
#include <stdexcept>

#include <cstdlib>
#include <string>

constexpr const char *MODEL_WEIGHTS_FILE_NAME = "model_weights.bin";
constexpr const char *MODEL_CONFIG_FILE_NAME = "model_offsets.json";

void *init_engine()
{
    // engine pointer
    Engine *enginePtr = nullptr;

    // get the engine library bin directory
    const char *engDir = std::getenv("ENGINE_BIN_DIR");
    std::string binDir = engDir ? std::string(engDir) : ".";

    std::string weightsPath = binDir + "/" + MODEL_WEIGHTS_FILE_NAME;
    std::string configPath = binDir + "/" + MODEL_CONFIG_FILE_NAME;

    size_t arenaSizeInMB = 100;

    try
    {
        enginePtr = new Engine(arenaSizeInMB, weightsPath.c_str(), configPath.c_str());
    }
    catch (std::exception &ex)
    {
        std::cerr << "Failed to instantiate an engine due to " << ex.what() << std::endl;
    }

    return enginePtr;
}

void generate(void *enginePtr, unsigned int *prompt, unsigned int prompt_len, unsigned int *output, unsigned int max_tokens, float temperature, unsigned int topK, float topP)
{
    Engine *engine = static_cast<Engine *>(enginePtr);

    std::vector<size_t> prompt_;
    // check the max_tokens limit
    if (prompt_len >= max_tokens)
    {
        // reached the limit so only copy first max_tokens number of tokens to prompt
        prompt_ = std::vector<size_t>(prompt, prompt + prompt_len);
    }
    else
    {
        prompt_ = std::vector<size_t>(prompt, prompt + prompt_len);
    }

    engine->infer(prompt_, output, max_tokens, temperature, static_cast<size_t>(topK), topP);
}