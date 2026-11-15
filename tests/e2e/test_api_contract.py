"""C API contract test for generate(), on synthetic weights (no downloads; runs in CI).

Checks that:
  - invalid requests return -1 and write nothing into the caller's buffer;
  - valid requests return the number of entries written and never write past max_tokens;
  - prompts longer than the old fixed 90 MB arena allowed (~192 tokens) now work, because the
    engine sizes its activation arena from the request;
  - the arena is reused correctly across a long request followed by a short one.
"""

import ctypes
import os
import sys

CANARY = 0xDEADBEEF


def load_engine():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    bin_path = os.path.join(repo_root, "bin")
    synthetic_dir = os.path.join(bin_path, "synthetic")
    lib_ext = ".dylib" if sys.platform == "darwin" else ".so"
    lib_path = os.path.join(bin_path, f"libengine{lib_ext}")
    if not os.path.exists(lib_path):
        sys.exit(f"Error: shared library not found at {lib_path}; run 'make' first.")

    if not os.path.exists(os.path.join(synthetic_dir, "model_weights.bin")):
        sys.path.insert(0, os.path.join(repo_root, "scripts"))
        from generate_synthetic_weights import generate_synthetic_weights
        generate_synthetic_weights(synthetic_dir, seed=42)

    os.environ["ENGINE_BIN_DIR"] = synthetic_dir
    lib = ctypes.CDLL(lib_path)
    lib.init_engine.argtypes = []
    lib.init_engine.restype = ctypes.c_void_p
    lib.generate.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32), ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_uint32), ctypes.c_uint32,
        ctypes.c_float, ctypes.c_uint32, ctypes.c_float,
    ]
    lib.generate.restype = ctypes.c_int
    handle = lib.init_engine()
    if not handle:
        sys.exit("Error: init_engine returned NULL")
    return lib, handle


def call(lib, handle, prompt, max_tokens, buffer_len=None, prompt_len=None):
    """Runs generate with a canary-filled output buffer larger than max_tokens."""
    buffer_len = buffer_len if buffer_len is not None else max(max_tokens, 1) + 16
    out = (ctypes.c_uint32 * buffer_len)(*([CANARY] * buffer_len))
    c_prompt = (ctypes.c_uint32 * max(len(prompt), 1))(*prompt)
    n = lib.generate(handle, c_prompt, len(prompt) if prompt_len is None else prompt_len,
                     out, max_tokens, 0.0, 40, 0.95)
    return n, list(out)


failures = []


def check(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        failures.append(msg)


def main():
    lib, handle = load_engine()

    print("Invalid requests return -1 and leave the output buffer untouched:")
    invalid = {
        "prompt longer than max_tokens (30 > 24)": (list(range(100, 130)), 24),
        "prompt length equal to max_tokens": ([100, 200, 300], 3),
        "max_tokens = 0": ([100, 200, 300], 0),
        "max_tokens above the 1024-token context": ([100, 200, 300], 1025),
        "token id outside the vocabulary (50257)": ([100, 50257, 300], 8),
        "empty prompt": ([], 8),
    }
    for name, (prompt, max_tokens) in invalid.items():
        n, out = call(lib, handle, prompt, max_tokens)
        check(n == -1 and all(v == CANARY for v in out), name)

    n = lib.generate(None, None, 0, None, 8, 0.0, 40, 0.95)
    check(n == -1, "null engine / prompt / output pointers")

    print("Valid requests return the count and never write past max_tokens:")
    prompt, max_tokens = [100, 200, 300], 6
    n, out = call(lib, handle, prompt, max_tokens)
    check(n == max_tokens, f"returns {max_tokens} for a full generation (got {n})")
    check(out[:3] == prompt, "output starts with the prompt")
    check(all(v == CANARY for v in out[max_tokens:]), "nothing written past max_tokens")

    print("Arena is sized from the request:")
    long_prompt = [(i * 37) % 50000 + 10 for i in range(200)]  # past the old ~192-token crash
    n, out = call(lib, handle, long_prompt, 204)
    check(n == 204, f"200-token prompt completes (got {n})")
    check(all(v == CANARY for v in out[204:]), "nothing written past max_tokens on the long request")

    n, out = call(lib, handle, prompt, max_tokens)
    check(n == max_tokens and out[:max_tokens] == call(lib, handle, prompt, max_tokens)[1][:max_tokens],
          "a short request after a long one gives the same, deterministic result")

    if failures:
        print(f"API Contract Test: FAILED ({len(failures)} check(s))")
        sys.exit(1)
    print("API Contract Test: PASSED")


if __name__ == "__main__":
    main()
