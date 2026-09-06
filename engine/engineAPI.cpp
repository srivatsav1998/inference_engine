
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
    // initializing the engine with default parameters
    Engine *engine;

    const char* env_p = std::getenv("ENGINE_BIN_DIR");
    std::string bin_dir = env_p ? std::string(env_p) : ".";
    
    std::string weights_path = bin_dir + "/" + MODEL_WEIGHTS_FILE_NAME;
    std::string config_path = bin_dir + "/" + MODEL_CONFIG_FILE_NAME;

    try
    {
        engine = new Engine(90, weights_path.c_str(), config_path.c_str());
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