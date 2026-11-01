import ctypes
import os
import sys

def main():
    # Dynamically resolve paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
    bin_path = os.path.join(repo_root, "bin")
    synthetic_dir = os.path.join(bin_path, "synthetic")

    lib_ext = ".dylib" if sys.platform == "darwin" else ".so"
    lib_path = os.path.join(bin_path, f"libengine{lib_ext}")

    if not os.path.exists(lib_path):
        print(f"Error: Shared library not found at {lib_path}")
        print("Please compile first via 'make'.")
        sys.exit(1)

    # Ensure synthetic weights exist
    weights_bin = os.path.join(synthetic_dir, "model_weights.bin")
    offsets_json = os.path.join(synthetic_dir, "model_offsets.json")
    if not (os.path.exists(weights_bin) and os.path.exists(offsets_json)):
        print("Synthetic weights not found; generating on-the-fly...")
        sys.path.insert(0, os.path.join(repo_root, "scripts"))
        from generate_synthetic_weights import generate_synthetic_weights
        generate_synthetic_weights(synthetic_dir, seed=42)

    os.environ["ENGINE_BIN_DIR"] = synthetic_dir

    print(f"Loading engine from: {lib_path}")
    engine = ctypes.CDLL(lib_path)

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

    print("Initializing engine with synthetic weights...")
    engine_ptr = engine.init_engine()
    if not engine_ptr:
        print("Error: Engine initialization failed.")
        sys.exit(1)

    # Test prompt token IDs
    prompt_ids = [100, 200, 300]
    max_tokens = 6
    expected_output = [100, 200, 300, 11892, 11892, 11892]

    c_prompt = (ctypes.c_uint32 * len(prompt_ids))(*prompt_ids)
    c_out = (ctypes.c_uint32 * max_tokens)()

    print(f"Running deterministic greedy generation with prompt {prompt_ids}...")
    engine.generate(
        engine_ptr,
        c_prompt,
        len(prompt_ids),
        c_out,
        max_tokens,
        ctypes.c_float(0.0),  # Greedy
        ctypes.c_uint32(40),
        ctypes.c_float(0.95)
    )

    out_list = list(c_out)
    print(f"Generated token IDs: {out_list}")

    if out_list == expected_output:
        print("Synthetic E2E Regression Test: PASSED")
        sys.exit(0)
    else:
        print(f"Synthetic E2E Regression Test: FAILED (Expected: {expected_output}, Got: {out_list})")
        sys.exit(1)

if __name__ == "__main__":
    main()
