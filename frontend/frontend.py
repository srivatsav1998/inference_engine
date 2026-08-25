
import ctypes
import os
import tiktoken

gpt2_tokenizer = tiktoken.get_encoding("gpt2")

bin_path = os.path.abspath("/Users/srivatsavg/Desktop/workspace/projects/inference_engine/engine/bin/")
lib_path = bin_path + os.path.sep + "libengine.dylib"

engine = ctypes.CDLL(lib_path)

engine.init_engine.argtypes = []
engine.init_engine.restype = ctypes.c_void_p

engine_ptr = engine.init_engine();

engine.generate.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32), ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint32), ctypes.c_uint32]
engine.generate.restype = None

prompt = "In stock trading, an options long call is"
prompt_ids = gpt2_tokenizer.encode(prompt)

c_prompt = (ctypes.c_uint32 * len(prompt_ids))(*prompt_ids)

max_tokens = 20
c_out = (ctypes.c_uint32 * (max_tokens))()

engine.generate(engine_ptr, c_prompt, len(prompt_ids), c_out, max_tokens)
out_list = list(c_out)

print(f'generated {len(out_list)} words')

out_text = gpt2_tokenizer.decode(out_list)
print(out_text)