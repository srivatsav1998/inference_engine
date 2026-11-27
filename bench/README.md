# Benchmarks

Measurement harness for the engine. Its job is to tell you whether an optimization strategy (caching, batching, speculative decoding, quantization, kernel selection) actually helped, and by how much. Hardware ceilings are included only as context.

```bash
make bench        # full suite: kernels + bandwidth + engine E2E + PyTorch baseline (~3 min)
make bench_quick  # 2 prompt lengths, 1 repeat, no PyTorch (~30 s)
```

Both targets build the engine with `BUILD=release` (`-O3 -DNDEBUG`, so asserts are off) into `bin/release/`. The default debug build in `bin/` keeps asserts on for tests. Real GPT-2 weights must be present in `bin/` (`make weights`).

## What is measured

| Component | File | What it reports |
| --- | --- | --- |
| Memory bandwidth | `kernel_bench.cpp` | STREAM-style NEON read over 1 GiB with 1, P-core-count, and all threads (threads are not pinned; where the P-core count is unavailable, half the threads is used). This is the ceiling for weight-streaming decode. |
| MatMul kernels | `kernel_bench.cpp` | `matMul2D_out` vs Accelerate `cblas_sgemm` (which uses AMX) at every GPT-2 projection shape, for M = 1 (decode), 32 and 128 (prefill). Reports ms, GFLOP/s, weight GB/s and max abs diff. |
| Pointwise kernels | `kernel_bench.cpp` | LayerNorm, GELU, Softmax, Add at decode and prefill shapes (µs, GB/s). |
| End-to-end | `e2e_bench.py` | Engine (through the C API) and PyTorch/HF GPT-2 on identical token IDs. Reports TTFT, prefill tok/s, decode ms/tok, avg busy cores, first (cold) call time, peak RSS, and the generated tokens. |
| Orchestration | `run_bench.py` | Runs everything and records hardware (`sysctl`: P/E cores, caches, ISA features), commit and build. Derives the decode roofline and greedy agreement with PyTorch, then writes `results/<timestamp>-<commit>.json` and prints a markdown summary. |

## Methodology notes

- **Medians**: each kernel case runs warmups, then repeats until at least 0.5 s and 7 runs have elapsed, and reports the median. E2E numbers are medians of 3 repeats.
- **DRAM-realistic matmul**: weight matrices are rotated across copies totalling ≥256 MB, so every timed call streams `B` from DRAM as real decode does. Without this, small matrices stay in L2/SLC and the GB/s figures are inflated.
- **Engine decode timing** is black-box: `(t(L→L+1+D) − t(L→L+1)) / D`, because the C API doesn't expose per-step timing. TTFT includes one LM-head projection and sampling.
- **Roofline**: decode streams every fp32 weight once per token (≈498 MB). The floor is that size divided by the measured peak read bandwidth.
- **Isolation**: each E2E backend runs in its own process, so peak RSS and CPU utilization belong to that backend only. PyTorch runs offline (`HF_HUB_OFFLINE=1`) from the local HF cache.
- **Prompt lengths stay at 8 / 32 / 128** so results remain comparable with earlier runs. (Longer prompts used to overflow the engine's fixed 90 MB arena; the arena is now sized per request.)

## Not yet measured (needs engine support)

- **Perplexity / logit-level accuracy.** The C API returns only token IDs. A logits-returning entry point would allow perplexity on a fixed corpus, which matters once quantization lands.
- **Batched throughput** (tokens/s across concurrent sequences) and **speculative-decoding acceptance rate.** Both wait on the corresponding engine features.
