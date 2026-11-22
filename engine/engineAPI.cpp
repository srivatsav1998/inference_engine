
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

    try
    {
        enginePtr = new Engine(weightsPath.c_str(), configPath.c_str());
    }
    catch (std::exception &ex)
    {
        std::cerr << "Failed to instantiate an engine due to " << ex.what() << std::endl;
    }

    return enginePtr;
}

int generate(void *enginePtr, unsigned int *prompt, unsigned int prompt_len, unsigned int *output, unsigned int max_tokens, float temperature, unsigned int topK, float topP)
{
    if (enginePtr == nullptr || prompt == nullptr || output == nullptr)
    {
        std::cerr << "generate: engine, prompt and output must be non-null" << std::endl;
        return -1;
    }

    Engine *engine = static_cast<Engine *>(enginePtr);
    std::vector<float> prompt_(prompt, prompt + prompt_len);

    // Exceptions must not cross the C boundary into the caller (ctypes would abort the process).
    try
    {
        return static_cast<int>(engine->infer(prompt_, output, max_tokens, temperature, static_cast<size_t>(topK), topP));
    }
    catch (const std::exception &ex)
    {
        std::cerr << "generate: " << ex.what() << std::endl;
        return -1;
    }
}
