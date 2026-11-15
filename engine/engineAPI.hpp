#pragma once

// Returns an engine handle, or NULL if the weights could not be loaded (the reason is printed to stderr).
extern "C" void *init_engine();

// Writes the prompt followed by generated tokens into `output`, which must have room for
// `max_tokens` entries. Returns the number of entries written: max_tokens, or fewer if
// <|endoftext|> was generated. Returns -1 (writing nothing; reason on stderr) if the request is
// invalid: null pointers, empty prompt, max_tokens <= prompt_len, max_tokens above the model
// context (1024), a token id outside the vocabulary, or not enough memory for activations.
// Note: topK = 0 is not yet supported when temperature > 0 (it crashes in sampling).
extern "C" int generate(void *enginePtr, unsigned int *prompt, unsigned int prompt_len, unsigned int *output, unsigned int max_tokens, float temperature, unsigned int topK, float topP);
