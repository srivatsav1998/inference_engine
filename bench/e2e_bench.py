"""End-to-end generation benchmark for one backend (engine or PyTorch), emitting JSON on stdout.

Each backend runs in its own process (run_bench.py spawns them) so that peak RSS and CPU
utilization are attributable to that backend alone.

Metrics per prompt length L (greedy decoding, identical token IDs for both backends):
  - ttft_s           : time to first token = prefill of L tokens + 1 sampled token
  - prefill_tok_s    : L / ttft_s (includes one LM-head + sampling; approximate)
  - decode_ms_per_tok: (t(L -> L+1+D) - t(L -> L+1)) / D   for the engine (black-box C API),
                       measured directly per step for PyTorch
  - decode_tok_s     : 1000 / decode_ms_per_tok
  - cpu_util_cores   : (user+sys CPU time) / wall time during the full generation (avg busy cores)
  - output_tokens    : generated token IDs (used for the greedy-agreement quality check)
Process-level:
  - first_call_s     : first generate() call after init (includes first-touch page faults on
                       the mmap'd weights — a "cold" run)
  - peak_rss_mb      : max resident set size of the process
"""

import argparse
import ctypes
import json
import os
import resource
import statistics
import sys
import time

PASSAGE = (
    "The history of computing is a story of abstraction. Early programmers wrote machine code "
    "by hand, flipping switches and punching cards, until assemblers and compilers let them "
    "describe intent rather than mechanism. Each new layer traded a little efficiency for a "
    "great deal of expressiveness, and each generation of hardware made that trade cheaper. "
    "Today, neural networks are compiled into graphs of tensor operations, scheduled across "
    "cores and accelerators, and executed by runtimes that must balance memory, latency and "
    "throughput. Understanding those runtimes means understanding both the mathematics of the "
    "model and the machine that evaluates it."
)


def prompt_tokens(length):
    import tiktoken

    ids = tiktoken.get_encoding("gpt2").encode(PASSAGE)
    out = []
    while len(out) < length:
        out.extend(ids)
    return out[:length]


def rusage_cpu():
    r = resource.getrusage(resource.RUSAGE_SELF)
    return r.ru_utime + r.ru_stime


def peak_rss_mb():
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes, Linux reports kilobytes
    return rss / (1024 * 1024) if sys.platform == "darwin" else rss / 1024


# ---------------------------------------------------------------------------
# Engine backend (black-box via the C API)
# ---------------------------------------------------------------------------
def run_engine(args):
    os.environ["ENGINE_BIN_DIR"] = os.path.abspath(args.weights_dir)
    lib = ctypes.CDLL(os.path.abspath(args.lib))
    lib.init_engine.argtypes = []
    lib.init_engine.restype = ctypes.c_void_p
    lib.generate.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32), ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_uint32), ctypes.c_uint32,
        ctypes.c_float, ctypes.c_uint32, ctypes.c_float,
    ]
    lib.generate.restype = ctypes.c_int

    t0 = time.perf_counter()
    handle = lib.init_engine()
    init_s = time.perf_counter() - t0
    if not handle:
        raise RuntimeError("init_engine failed")

    def generate(prompt, total_len):
        c_prompt = (ctypes.c_uint32 * len(prompt))(*prompt)
        c_out = (ctypes.c_uint32 * total_len)()
        cpu0, t0 = rusage_cpu(), time.perf_counter()
        n = lib.generate(handle, c_prompt, len(prompt), c_out, total_len, 0.0, 0, 0.0)
        wall = time.perf_counter() - t0
        if n != total_len:
            # An error (-1) or an early <|endoftext|> would make the per-token timing meaningless.
            raise RuntimeError(f"generate wrote {n} of {total_len} tokens")
        return wall, rusage_cpu() - cpu0, list(c_out)

    first_prompt = prompt_tokens(args.prompt_lens[0])
    first_call_s, _, _ = generate(first_prompt, len(first_prompt) + 1)

    results = []
    for L in args.prompt_lens:
        prompt = prompt_tokens(L)
        D = args.decode_tokens
        ttfts, decodes, utils, out_tokens = [], [], [], None
        for _ in range(args.repeats):
            t_prefill, _, _ = generate(prompt, L + 1)
            t_full, cpu_full, out = generate(prompt, L + 1 + D)
            ttfts.append(t_prefill)
            decodes.append((t_full - t_prefill) / D)
            utils.append(cpu_full / t_full)
            out_tokens = out[L:]
        results.append(summarize(L, D, ttfts, decodes, utils, out_tokens))

    return {
        "backend": "engine",
        "init_s": init_s,
        "first_call_s": first_call_s,
        "peak_rss_mb": peak_rss_mb(),
        "results": results,
    }


# ---------------------------------------------------------------------------
# PyTorch / HuggingFace reference backend
# ---------------------------------------------------------------------------
def run_torch(args):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")  # use the local HF cache; never hit the network
    import torch
    from transformers import GPT2LMHeadModel

    t0 = time.perf_counter()
    model = GPT2LMHeadModel.from_pretrained("gpt2", dtype=torch.float32).eval()
    init_s = time.perf_counter() - t0

    @torch.inference_mode()
    def generate(prompt, decode_steps):
        ids = torch.tensor([prompt], dtype=torch.long)
        cpu0, t0 = rusage_cpu(), time.perf_counter()
        out = model(input_ids=ids, use_cache=True)
        tok = out.logits[0, -1].argmax().item()
        ttft = time.perf_counter() - t0
        generated, past = [tok], out.past_key_values
        t1 = time.perf_counter()
        for _ in range(decode_steps):
            out = model(input_ids=torch.tensor([[tok]]), past_key_values=past, use_cache=True)
            past = out.past_key_values
            tok = out.logits[0, -1].argmax().item()
            generated.append(tok)
        decode_total = time.perf_counter() - t1
        wall = time.perf_counter() - t0
        return ttft, decode_total, wall, rusage_cpu() - cpu0, generated

    first_prompt = prompt_tokens(args.prompt_lens[0])
    t0 = time.perf_counter()
    generate(first_prompt, 0)
    first_call_s = time.perf_counter() - t0

    results = []
    for L in args.prompt_lens:
        prompt = prompt_tokens(L)
        D = args.decode_tokens
        ttfts, decodes, utils, out_tokens = [], [], [], None
        for _ in range(args.repeats):
            ttft, decode_total, wall, cpu, out = generate(prompt, D)
            ttfts.append(ttft)
            decodes.append(decode_total / D)
            utils.append(cpu / wall)
            out_tokens = out
        results.append(summarize(L, D, ttfts, decodes, utils, out_tokens))

    return {
        "backend": "torch",
        "torch_version": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "init_s": init_s,
        "first_call_s": first_call_s,
        "peak_rss_mb": peak_rss_mb(),
        "results": results,
    }


def summarize(L, D, ttfts, decodes, utils, out_tokens):
    ttft = statistics.median(ttfts)
    dec = statistics.median(decodes)
    return {
        "prompt_len": L,
        "decode_tokens": D,
        "ttft_s": ttft,
        "prefill_tok_s": L / ttft,
        "decode_ms_per_tok": dec * 1e3,
        "decode_tok_s": 1.0 / dec,
        "cpu_util_cores": statistics.median(utils),
        "output_tokens": out_tokens,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["engine", "torch"], required=True)
    ap.add_argument("--lib", default="bin/release/libengine.dylib")
    ap.add_argument("--weights-dir", default="bin")
    ap.add_argument("--prompt-lens", type=int, nargs="+", default=[8, 32, 128])
    ap.add_argument("--decode-tokens", type=int, default=64)
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()

    for L in args.prompt_lens:
        if L + 1 + args.decode_tokens > 1024:
            ap.error(f"prompt_len {L} + decode_tokens exceeds GPT-2's 1024-position context")

    result = run_engine(args) if args.backend == "engine" else run_torch(args)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
