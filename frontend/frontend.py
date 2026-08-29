
import ctypes
import os
import tiktoken

### Setup
bin_path = os.path.abspath("/Users/srivatsavg/Desktop/workspace/projects/inference_engine/engine/bin/")
lib_path = bin_path + os.path.sep + "libengine.dylib"

engine = ctypes.CDLL(lib_path)

### Engine functions setup
engine.init_engine.argtypes = []
engine.init_engine.restype = ctypes.c_void_p

engine.generate.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32), ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint32), ctypes.c_uint32, ctypes.c_float, ctypes.c_uint32, ctypes.c_float]
engine.generate.restype = None

### Engine initialization
engine_ptr = engine.init_engine();

### Text generation phase
max_tokens = 24
prompt = input("Enter the prompt (total sequence length is 24): ")

### Tokenize the given prompt
gpt2_tokenizer = tiktoken.get_encoding("gpt2")
prompt_ids = gpt2_tokenizer.encode(prompt)

c_prompt = (ctypes.c_uint32 * len(prompt_ids))(*prompt_ids)

c_out = (ctypes.c_uint32 * (max_tokens))()

### Generate the text
engine.generate(engine_ptr, c_prompt, len(prompt_ids), c_out, max_tokens, ctypes.c_float(0.8), ctypes.c_uint32(40), ctypes.c_float(0.95))
out_list = list(c_out)

print(f'Generated {len(out_list)} words')

out_text = gpt2_tokenizer.decode(out_list)
print(out_text)