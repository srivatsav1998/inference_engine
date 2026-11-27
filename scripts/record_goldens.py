"""Records the golden outputs used by tests/e2e/test_golden_regression.py.

Expected tokens come from HuggingFace's GPT-2 (PyTorch) loaded with the SAME weight files the
engine reads, never from the engine itself. Only needed when the cases or the weights change;
running the test does not need torch or transformers.

    python scripts/record_goldens.py            # synthetic weights (bin/synthetic_golden)
    python scripts/record_goldens.py --full     # real GPT-2 weights (bin/, `make weights`)

Each case also records the smallest top-1 minus top-2 logit gap seen during generation. A tiny
gap means ordinary float rounding could flip the argmax, so such cases are rejected.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from transformers import GPT2Config, GPT2LMHeadModel

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
GOLDEN_DIR = os.path.join(REPO_ROOT, "tests", "e2e", "golden")

EOS = 50256
MIN_MARGIN = 2e-4

# Real-weight prompts, as text so they can be reviewed here; the JSON stores token ids.
REAL_PROMPTS = {
    "single_token": ("The", 24),
    "short": ("A quick brown fox", 24),
    "factual": ("The capital of France is", 32),
    "code": ("def fibonacci(n):\n    if n < 2:\n        return", 32),
    "narrative": ("Once upon a time, in a small village by the sea, there lived", 48),
    "list": ("Monday, Tuesday, Wednesday,", 16),
    "long_decode": ("The history of computing is", 160),
    "paragraph": (
        "Inference engines compile a model into a graph of operators, bind each operator to a "
        "kernel suited to the hardware, plan the memory for every intermediate tensor, and then "
        "execute the graph once per generated token. The first step is to",
        32,
    ),
}


def load_reference(weights_dir):
    with open(os.path.join(weights_dir, "model_offsets.json")) as f:
        schema = json.load(f)
    raw = np.memmap(os.path.join(weights_dir, "model_weights.bin"), dtype=np.float32, mode="r")

    state = {}
    for name, meta in schema.items():
        start = meta["byte_offset"] // 4
        count = int(np.prod(meta["shape"]))
        state["transformer." + name] = torch.from_numpy(np.array(raw[start:start + count]).reshape(meta["shape"]))
    state["lm_head.weight"] = state["transformer.wte.weight"]  # tied

    model = GPT2LMHeadModel(GPT2Config()).eval()
    model.load_state_dict(state, strict=True)
    return model


@torch.no_grad()
def greedy(model, prompt, max_tokens):
    """Mirrors Engine::infer: returns (prompt + generated, smallest top-2 gap); stops after EOS."""
    out = list(prompt)
    feed = torch.tensor([prompt])
    past = None
    min_margin = float("inf")
    while len(out) < max_tokens:
        step = model(feed, past_key_values=past, use_cache=True)
        past = step.past_key_values
        top = torch.topk(step.logits[0, -1], 2)
        min_margin = min(min_margin, (top.values[0] - top.values[1]).item())
        token = top.indices[0].item()
        out.append(token)
        if token == EOS:
            break
        feed = torch.tensor([[token]])
    return out, min_margin


def record(model, name, prompt, max_tokens):
    out, margin = greedy(model, prompt, max_tokens)
    print(f"  {name:<22} prompt={len(prompt):<5} output={len(out):<5} min_margin={margin:.4f}")
    if margin < MIN_MARGIN:
        sys.exit(f"Case '{name}' has a near-tie (margin {margin:.2e}); change its prompt or seed.")
    return {"name": name, "prompt": prompt, "max_tokens": max_tokens, "expected": out}


def synthetic_cases(model):
    rng = np.random.default_rng(1234)

    def prompt(length):
        return [int(t) for t in rng.integers(0, 50000, length)]

    cases = []
    # Prompt lengths around the sizes that kernels, threading and block unrolls care about.
    for length in (1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 128):
        cases.append(record(model, f"prefill_{length}", prompt(length), length + 12))
    cases.append(record(model, "prefill_257", prompt(257), 257 + 8))
    cases.append(record(model, "single_new_token", prompt(6), 7))
    cases.append(record(model, "long_decode", prompt(4), 4 + 160))
    cases.append(record(model, "context_limit", prompt(1012), 1024))  # ends at the limit; last position fed is 1022

    # EOS handling: random weights never predict <|endoftext|> on their own, so give its
    # embedding row 1.01x the row of a token that does get predicted. EOS then wins instead.
    base = cases[[c["name"] for c in cases].index("prefill_8")]
    alias = base["expected"][len(base["prompt"]) + 3]
    wte = model.transformer.wte.weight
    saved = wte[EOS].clone()
    with torch.no_grad():
        wte[EOS] = wte[alias] * 1.01
    eos = record(model, "stops_at_eos", base["prompt"], base["max_tokens"])
    with torch.no_grad():
        wte[EOS] = saved
    if eos["expected"][-1] != EOS or len(eos["expected"]) >= eos["max_tokens"]:
        sys.exit("The EOS case did not stop early; pick another alias token.")
    eos["eos_alias_token"] = alias
    cases.append(eos)
    return cases


def real_cases(model):
    import tiktoken

    enc = tiktoken.get_encoding("gpt2")
    cases = []
    for name, (text, new_tokens) in REAL_PROMPTS.items():
        ids = enc.encode(text)
        cases.append(record(model, name, ids, len(ids) + new_tokens))
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--full", action="store_true", help="record against the real GPT-2 weights in bin/")
    args = parser.parse_args()

    if args.full:
        weights_dir, out_name = os.path.join(REPO_ROOT, "bin"), "real.json"
    else:
        weights_dir, out_name = os.path.join(REPO_ROOT, "bin", "synthetic_golden"), "synthetic.json"
        if not os.path.exists(os.path.join(weights_dir, "model_weights.bin")):
            from generate_synthetic_weights import generate_synthetic_weights
            generate_synthetic_weights(weights_dir, seed=42, varied_affine=True)

    model = load_reference(weights_dir)
    print(f"Recording reference outputs from {weights_dir}")
    cases = real_cases(model) if args.full else synthetic_cases(model)

    os.makedirs(GOLDEN_DIR, exist_ok=True)
    out_path = os.path.join(GOLDEN_DIR, out_name)
    with open(out_path, "w") as f:
        f.write("[\n" + ",\n".join(json.dumps(c, separators=(",", ":")) for c in cases) + "\n]\n")
    print(f"Wrote {len(cases)} cases to {out_path}")


if __name__ == "__main__":
    main()
