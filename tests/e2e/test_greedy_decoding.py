import ctypes
import os
import sys
import tiktoken

def main():
    # Dynamically resolve bin directory relative to repo root
    current_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
    bin_path = os.path.join(repo_root, "bin")
    
    # Support platform dynamic library extension
    lib_ext = ".dylib" if sys.platform == "darwin" else ".so"
    lib_path = os.path.join(bin_path, f"libengine{lib_ext}")

    if not os.path.exists(lib_path):
        print(f"Error: Shared library not found at {lib_path}")
        print("Please build the engine first using 'make'.")
        sys.exit(1)

    # Pass the bin directory to C++ runtime to find model weights
    os.environ["ENGINE_BIN_DIR"] = bin_path

    print(f"Loading engine from: {lib_path}")
    engine = ctypes.CDLL(lib_path)

    ### Engine functions setup
    engine.init_engine.argtypes = []
    engine.init_engine.restype = ctypes.c_void_p

    engine.generate.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_uint32,
        ctypes.c_float,
        ctypes.c_uint32,
        ctypes.c_float
    ]
    engine.generate.restype = None

    print("Initializing engine...")
    engine_ptr = engine.init_engine()
    if not engine_ptr:
        print("Error: Engine initialization failed.")
        sys.exit(1)

    max_tokens = 9
    prompt = "A quick brown fox"
    expected_output = "A quick brown fox, with a long tail"

    gpt2_tokenizer = tiktoken.get_encoding("gpt2")
    prompt_ids = gpt2_tokenizer.encode(prompt)

    c_prompt = (ctypes.c_uint32 * len(prompt_ids))(*prompt_ids)
    c_out = (ctypes.c_uint32 * max_tokens)()

    print(f"Running greedy generation (prompt: '{prompt}')...")
    engine.generate(
        engine_ptr,
        c_prompt,
        len(prompt_ids),
        c_out,
        max_tokens,
        ctypes.c_float(0.0),  # temperature = 0 for greedy decoding
        ctypes.c_uint32(40),
        ctypes.c_float(0.95)
    )

    out_list = list(c_out)
    out_text = gpt2_tokenizer.decode(out_list)
    print(f"Generated text: '{out_text}'")

    if out_text == expected_output:
        print("E2E Greedy Decoding Test: PASSED")
        sys.exit(0)
    else:
        print(f"E2E Greedy Decoding Test: FAILED (Expected: '{expected_output}', Got: '{out_text}')")
        sys.exit(1)

if __name__ == "__main__":
    main()
