
#include "engineAPI.hpp"
#include "engine.hpp"

#include <iostream>
#include <stdexcept>

constexpr const char *MODEL_WEIGHTS_FILE_PATH = "/Users/srivatsavg/Desktop/workspace/projects/inference_engine/model_convertor/model_weights.bin";
constexpr const char *MODEL_CONFIG_FILE_PATH = "/Users/srivatsavg/Desktop/workspace/projects/inference_engine/model_convertor/model_offsets.json";

void *init_engine()
{
    // initializing the engine with default parameters
    Engine *engine;

    try
    {
        engine = new Engine(90, MODEL_WEIGHTS_FILE_PATH, MODEL_CONFIG_FILE_PATH);
    }
    catch (std::exception &ex)
    {
        std::cerr << "Failed to instantiate an engine due to " << ex.what() << std::endl;
    }

    return engine;
}

void generate(void *enginePtr, unsigned int *prompt, unsigned int prompt_len, unsigned int *output, unsigned int max_tokens, float temperature, unsigned int topK, float topP)
{
    Engine *engine = static_cast<Engine *>(enginePtr);

    std::vector<size_t> prompt_(prompt, prompt + prompt_len);

    engine->infer(prompt_, output, max_tokens, temperature, static_cast<size_t>(topK), topP);
}