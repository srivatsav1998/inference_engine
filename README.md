# C++ Inference Engine (GPT-2 124M)

## Goal

A from-scratch C++20 inference engine for GPT-2 124M, built to learn **how inference engines work**:

- how a model is **compiled** into an executable graph, and how that compile step uses hardware information to **bind each operator to the best-performing kernel**
- the optimization strategies production engines rely on: **KV caching**, **batching**, **speculative decoding** and **quantization**

GPT-2 124M is small enough to iterate on quickly, yet structurally a full decoder-only transformer.

**What the engine is today:**
- **No ML-framework dependencies.** No BLAS, PyTorch or tensor library in the engine. The only third-party dependency is the header-only `nlohmann-json`, used to parse weight metadata.
- **Weights are memory-mapped**, not copied. They are fp32 tensors exported from HuggingFace into one raw binary plus a JSON offset table (`scripts/convert_weights.py`).
- **ARM64 only.** The MatMul kernel uses NEON intrinsics.
- **Interface:** a C API (`libengine.dylib`) driven by a small Python frontend (`ctypes` + `tiktoken`).

---

## Architecture

| Module | Contents |
| --- | --- |
| `engine/core/` | `TensorView` (non-owning view with shape and strides; bounds checks are debug-only asserts), `ArenaAllocator` (bump allocator for activations, reset every generation step), `TaskQueue` (singleton thread pool built on a mutex and condition variables), `ModelWeightsLoader` (mmap) |
| `engine/ops/` | `mathOps`: kernels that write into a caller-provided output buffer (`*_out`), with documented rules on when the output may alias an input. MatMul has two NEON fast paths and a scalar fallback, and is multithreaded across output rows; the other kernels are scalar. `opType.hpp` / `operator.hpp`: the operator vocabulary and a polymorphic `Operator` hierarchy, groundwork for the graph runtime. |
| `engine/models/` | GPT-2 weight structs, attention/MLP blocks, `KVCache`, sampling (greedy, temperature, top-k, top-p) |
| `engine/engine.cpp`, `engineAPI.cpp` | Hand-written prefill + autoregressive decode loop; C-linkage API |
| `frontend/`, `scripts/` | Python frontend; weight export, synthetic weight generation and verification |
| `tests/` | doctest kernel unit tests; synthetic and real-weight greedy-decoding E2E regression tests |
| `bench/` | Benchmark suite: kernels, memory bandwidth, E2E engine vs PyTorch ([details](./bench/README.md)) |

---

## Quickstart

```bash
make weights          # download & export HuggingFace GPT-2 weights into bin/ (needs scripts/requirements.txt)

make check            # debug build + C++ unit tests (2,179 assertions) + synthetic E2E test (~2 s)
make test             # C++ unit tests only
make test_synthetic   # synthetic-weight greedy regression (no downloads; what CI runs)
make test_full        # real GPT-2 greedy decoding regression (needs bin/ weights)

make bench            # full benchmark suite (release build, ~3 min, needs bin/ weights)
make bench_quick      # quick benchmark (~30 s, no PyTorch baseline)
```

`BUILD=debug` (the default) keeps asserts on and builds into `bin/`. `BUILD=release` adds `-DNDEBUG` and builds into `bin/release/`. Benchmarks always use the release build.

---

## Benchmarks

Apple M2 (4P + 4E cores, 16 GB), release build. Methodology is in [`bench/README.md`](./bench/README.md). Each row links to a committed results file.

| Version | Results | Decode ms/tok | Decode tok/s | TTFT @128 (ms) | Prefill tok/s @128 |
| --- | --- | ---: | ---: | ---: | ---: |
| Baseline (`f435414`) | [json](./bench/results/20261114-173813-f435414.json) | 101.1 | 9.9 | 2839 | 45 |
| MatMul pointer hoisting (`a468d5b`) | [json](./bench/results/20261114-234524-a468d5b.json) | **19.5** | **51.2** | **737** | **174** |
| PyTorch 2.15 (HF, CPU) reference | same json | 17.4 | 57.4 | 84 | 1516 |

Current state (`a468d5b`) at every prompt length:

| Backend | Prompt | TTFT (ms) | Prefill tok/s | Decode ms/tok | Decode tok/s | Avg busy cores |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Engine | 8 | 52 | 153 | 19.5 | 51.2 | 1.2 |
| Engine | 32 | 178 | 180 | 19.8 | 50.6 | 1.5 |
| Engine | 128 | 737 | 174 | 20.2 | 49.4 | 2.6 |
| PyTorch 2.15 (HF, CPU) | 8 | 26 | 311 | 17.4 | 57.4 | 1.0 |
| PyTorch 2.15 (HF, CPU) | 32 | 41 | 771 | 18.3 | 54.6 | 1.0 |
| PyTorch 2.15 (HF, CPU) | 128 | 84 | 1516 | 18.3 | 54.7 | 1.1 |

- **Decode floor:** each generated token streams all ≈498 MB of fp32 weights. At the measured 65.5 GB/s read bandwidth, the floor is ≈7.6 ms/token (≈132 tok/s). The engine reaches 39% of it, up from 7% at baseline; PyTorch reaches about 42%.
- **Correctness:** the engine's greedy output matches PyTorch for all 65 generated tokens at every prompt length.
- **Memory:** peak RSS is 644 MB for the engine and 900 MB for PyTorch.
- **Kernel gap vs Accelerate (`cblas_sgemm`):**
  - The decode projection MatMuls (M = 1) are at parity: they are bandwidth-bound at ≈35–37 GB/s of weight streaming, the same as Accelerate.
  - The decode LM head (`wte^T`, M = 1) is 3.1× slower and is now ≈43% of each decode step (see Known Issues).
  - At prefill shapes (M = 32–128) the projections are 3–9× slower, and the LM head is 9–20× slower.
- **These are measurements, not targets.** The harness exists to measure the effect of each roadmap item below.

Earlier pre-harness numbers (cold-start wall-clock over a few tokens, plus peak activation footprint) are kept in [PROTOTYPE_ARCHIVE.md](./PROTOTYPE_ARCHIVE.md#historical-measurements).

---

## Known Issues

1. **Decode MatMul runs on one thread.** `matMul2D_out` splits work across output rows and skips the thread pool when there are ≤ 4 rows. Decode has M = 1, so every decode MatMul is single-threaded (≈1 busy core above). The projection MatMuls are already bandwidth-bound on one core, so this mostly matters for the LM head.
2. **The transposed-B (dot-product) MatMul path is latency-bound.** It accumulates each output into a single NEON register, so every FMA waits on the previous one. The decode LM head (1×768×50257) takes ≈8.3 ms, against ≈2.7 ms for Accelerate. At M > 1 it also re-streams all 154 MB of `wte` for every row (the matrix doesn't fit in cache), so its cost grows linearly with M.
3. **The work split ignores P/E cores.** Rows are split into equal chunks across all 8 cores and the call waits for every chunk, so a chunk scheduled on a slower efficiency core delays the whole call.
4. **Prefill computes logits for every prompt row**, but only the last row is used to pick the next token. At 128 tokens, the LM-head MatMul alone takes ≈260 ms of the ≈737 ms TTFT.
5. **The activation arena is a fixed 90 MB.** Prefill runs out of memory at about 192 prompt tokens: `ArenaAllocator::alloc` returns `nullptr` and the process crashes.
6. **`TaskQueue` tracks completion with one global counter.** Two concurrent callers would each wait for the other's tasks as well, so the pool assumes a single caller at a time.
7. **`LayerNormOp` stores `eps` and `SoftmaxOp` stores `axis`, but neither is passed to its kernel.**
8. **`init_engine` returns an uninitialized pointer** if `Engine` construction throws.
9. **Model dimensions are hardcoded:** `EMBEDDING_SIZE 768`, 12 layers.
10. **Pointwise kernels reach only ≈1–4 GB/s.** They are scalar loops through `TensorView::operator()`. That costs well under 1 ms per decode step today, but it will show up as the MatMuls get faster.
11. **The release build warns about unused variables** in `matMul2D_out` (`M`, `K_B`), because they are only used inside asserts.

---

## Roadmap

### Completed
- **Prototype** (tensor views, mmap weights, NEON MatMul, full GPT-2, KV cache, sampling, Python frontend). See [PROTOTYPE_ARCHIVE.md](./PROTOTYPE_ARCHIVE.md).
- **Decoupling & concurrency:** split declarations from implementations; persistent thread pool (`TaskQueue`) instead of spawning threads per operation.
- **Correctness & CI:** doctest kernel unit tests; synthetic E2E regression test in GitHub Actions (macOS ARM64); real-weight greedy regression test for local runs.
- **Engine foundation:** kernels write into caller-provided output buffers with documented aliasing rules; module split into `core/`, `ops/` and `models/`; `OpType` vocabulary and `Operator` hierarchy.
- **Measurement:** release build mode and benchmark suite (kernels, bandwidth, E2E vs PyTorch, roofline, greedy agreement).

### Phase 2: Graph Compiler & Runtime *(next)*
- [ ] Graph data structures (`graph/`): nodes, values, dependency tracking, topological ordering.
- [ ] Hardcoded GPT-2 graph, validated against `make check` / `make test_full` and benchmarked against the hand-written loop.
- [ ] **Hardware-aware kernel binding.** A compile step that inspects the hardware (core topology, cache sizes, ISA features such as DotProd / I8MM / BF16) and the operator shapes, then binds each graph node to a kernel implementation.
- [ ] Lifetime-aware static memory planner for activations.
- [ ] External graph format and parser so arbitrary models can run.

### Phase 3: KV Caching Strategies
- [ ] Persistent KV cache across `generate` calls (multi-turn sessions).
- [ ] Prefix caching: reuse KV for shared prompt prefixes.
- [ ] Paged / block-based KV cache.

### Phase 4: Batching
- [ ] Batched decode across multiple sequences.
- [ ] Continuous (iteration-level) batching: sequences join and leave the batch between steps.
- [ ] Benchmark: aggregate tokens/s vs per-request latency as batch size grows.

### Phase 5: Speculative Decoding
- [ ] Draft-model speculation with `distilgpt2` (shares GPT-2's tokenizer; needs dynamic layer count) as the draft for GPT-2 124M.
- [ ] Draft-free variants (e.g. prompt-lookup / n-gram drafting).
- [ ] Benchmark: acceptance rate and end-to-end speedup.

### Phase 6: Quantization
- [ ] Logits-returning API + perplexity evaluation on a fixed corpus (accuracy baseline before any quantization).
- [ ] Int8 weight-only quantization.
- [ ] Int4 group-wise quantization.
- [ ] Quantized kernels as additional implementations for the kernel-binding step.
- [ ] KV-cache quantization.

### Cross-cutting prerequisites
- [ ] Dynamic model dimensions (no hardcoded 768 / 12 layers). Needed for `distilgpt2` and other models.
- [ ] Native C++ `safetensors` loader (zero-copy, no Python conversion step).
- [ ] Work through the Known Issues above.

### Later / stretch
- [ ] Llama-family support: RoPE, RMSNorm, SwiGLU, Grouped-Query Attention.
- [ ] C++ BPE tokenizer, removing the `tiktoken` dependency from the frontend.
- [ ] GPU backend (Metal) as another kernel-binding target.

---

## Platform Requirements
- **Architecture:** ARM64 / AArch64 only (Apple Silicon or Linux AArch64). The kernels use `<arm_neon.h>` and will not compile on x86_64 without porting.
- **OS:** macOS or Linux (AArch64). Install `nlohmann-json` via Homebrew (`brew install nlohmann-json`) or apt (`nlohmann-json3-dev`).
- **Compiler:** C++20 (Apple Clang 15+ or GCC 11+).
- **Python:** 3.9+ with `scripts/requirements.txt` (PyTorch, Transformers, tiktoken, NumPy). PyTorch/Transformers are needed only for weight export and the benchmark baseline.
- **Benchmarks:** the Accelerate comparison is macOS-only and is skipped elsewhere.

## Design Notes
- **NEON instead of AVX2:** the development machine is an Apple M2.
- **`std::thread` instead of OpenMP:** kept deliberately low-level for learning and to avoid dependencies.
- **Bounds checking:** `TensorView` computes offsets from shape and strides and asserts index bounds in debug builds. Release builds (`-DNDEBUG`) do no bounds checking.
