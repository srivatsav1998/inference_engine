"""Golden greedy-decoding regression test, through the C API only.

Because it uses nothing but init_engine() / generate(), it stays valid across internal refactors.
Expected tokens were produced by an independent reference (HuggingFace GPT-2) on the same
weights; see scripts/record_goldens.py.

    python tests/e2e/test_golden_regression.py           # synthetic weights (CI)
    python tests/e2e/test_golden_regression.py --full    # real GPT-2 weights in bin/

Covers: many prompt lengths (prefill), long decode (KV cache growth), the 1024-token context
limit, early stop at <|endoftext|>, the sampling path in its deterministic settings, and state
leaking between requests or engine instances.

The only part that depends on the on-disk weight format is write_eos_variant().
"""

import argparse
import ctypes
import json
import os
import shutil
import sys
import tempfile

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
GOLDEN_DIR = os.path.join(REPO_ROOT, "tests", "e2e", "golden")
EOS = 50256

failures = []


def check(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        failures.append(msg)


def load_library():
    ext = ".dylib" if sys.platform == "darwin" else ".so"
    path = os.path.join(REPO_ROOT, "bin", f"libengine{ext}")
    if not os.path.exists(path):
        sys.exit(f"Error: shared library not found at {path}; run 'make' first.")
    lib = ctypes.CDLL(path)
    lib.init_engine.argtypes = []
    lib.init_engine.restype = ctypes.c_void_p
    lib.generate.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32), ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_uint32), ctypes.c_uint32,
        ctypes.c_float, ctypes.c_uint32, ctypes.c_float,
    ]
    lib.generate.restype = ctypes.c_int
    return lib


def init_engine(lib, weights_dir):
    os.environ["ENGINE_BIN_DIR"] = weights_dir
    handle = lib.init_engine()
    if not handle:
        sys.exit(f"Error: init_engine failed for {weights_dir}")
    return handle


def generate(lib, handle, case, temperature=0.0, top_k=40, top_p=0.95):
    prompt, max_tokens = case["prompt"], case["max_tokens"]
    c_prompt = (ctypes.c_uint32 * len(prompt))(*prompt)
    out = (ctypes.c_uint32 * max_tokens)()
    n = lib.generate(handle, c_prompt, len(prompt), out, max_tokens, temperature, top_k, top_p)
    return list(out)[:n] if n >= 0 else None


def describe_mismatch(case, got):
    expected = case["expected"]
    if got is None:
        return "generate() returned -1"
    for i, (e, g) in enumerate(zip(expected, got)):
        if e != g:
            return f"first difference at position {i} (prompt length {len(case['prompt'])}): expected {e}, got {g}"
    return f"expected {len(expected)} tokens, got {len(got)}"


def check_case(lib, handle, case, label=None, **sampling):
    got = generate(lib, handle, case, **sampling)
    label = label or f"{case['name']} ({len(case['prompt'])} prompt, {len(case['expected'])} total)"
    ok = got == case["expected"]
    check(ok, label if ok else f"{label}: {describe_mismatch(case, got)}")


def write_eos_variant(src_dir, dst_dir, alias_token):
    """Copies the weights, setting the <|endoftext|> embedding row to 1.01x that of alias_token."""
    with open(os.path.join(src_dir, "model_offsets.json")) as f:
        wte = json.load(f)["wte.weight"]
    for name in ("model_offsets.json", "model_weights.bin"):
        shutil.copyfile(os.path.join(src_dir, name), os.path.join(dst_dir, name))
    weights = np.memmap(os.path.join(dst_dir, "model_weights.bin"), dtype=np.float32, mode="r+")
    start, (vocab, dim) = wte["byte_offset"] // 4, wte["shape"]
    table = weights[start:start + vocab * dim].reshape(vocab, dim)
    table[EOS] = table[alias_token] * np.float32(1.01)
    weights.flush()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="use the real GPT-2 weights in bin/")
    args = parser.parse_args()

    if args.full:
        weights_dir, golden = os.path.join(REPO_ROOT, "bin"), "real.json"
    else:
        weights_dir, golden = os.path.join(REPO_ROOT, "bin", "synthetic_golden"), "synthetic.json"
        if not os.path.exists(os.path.join(weights_dir, "model_weights.bin")):
            sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))
            from generate_synthetic_weights import generate_synthetic_weights
            generate_synthetic_weights(weights_dir, seed=42, varied_affine=True)

    with open(os.path.join(GOLDEN_DIR, golden)) as f:
        cases = json.load(f)
    by_name = {c["name"]: c for c in cases}

    lib = load_library()
    handle = init_engine(lib, weights_dir)

    print("Greedy decoding matches the reference:")
    for case in cases:
        if "eos_alias_token" not in case:
            check_case(lib, handle, case)

    short, long = (by_name["short"], by_name["long_decode"]) if args.full else (by_name["prefill_5"], by_name["prefill_128"])

    print("Sampling path, in settings that must reduce to greedy:")
    check_case(lib, handle, short, "temperature=1.0, top_k=1", temperature=1.0, top_k=1, top_p=1.0)
    check_case(lib, handle, short, "temperature=0.7, top_p~0", temperature=0.7, top_k=40, top_p=1e-9)

    print("No state leaks between requests or engine instances:")
    check_case(lib, handle, long, "long request")
    check_case(lib, handle, short, "short request after a long one")
    check_case(lib, handle, long, "long request again")
    second = init_engine(lib, weights_dir)
    check_case(lib, second, short, "second engine instance")
    check_case(lib, handle, short, "first engine still correct")

    eos_cases = [c for c in cases if "eos_alias_token" in c]
    if eos_cases:
        print("Generation stops at <|endoftext|>:")
    for case in eos_cases:
        with tempfile.TemporaryDirectory() as tmp:
            write_eos_variant(weights_dir, tmp, case["eos_alias_token"])
            check_case(lib, init_engine(lib, tmp), case)

    if failures:
        print(f"Golden Regression Test: FAILED ({len(failures)} check(s))")
        sys.exit(1)
    print("Golden Regression Test: PASSED")


if __name__ == "__main__":
    main()
