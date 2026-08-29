
#pragma once

extern "C" void *init_engine();

extern "C" void generate(void *enginePtr, unsigned int *prompt, unsigned int prompt_len, unsigned int *output, unsigned int max_tokens, float temperature, unsigned int topK, float topP);