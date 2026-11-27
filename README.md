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
| `tests/` | doctest kernel and arena unit tests (including a MatMul sweep over shapes, transposes and strided layouts against a double-precision reference); synthetic and real-weight greedy-decoding E2E regression tests; golden regression test against HuggingFace outputs (`scripts/record_goldens.py` records them); C API contract test (input validation, buffer bounds) |
| `bench/` | Benchmark suite: kernels, memory bandwidth, E2E engine vs PyTorch ([details](./bench/README.md)) |

---

## Quickstart

```bash
make weights          # download & export HuggingFace GPT-2 weights into bin/ (needs scripts/requirements.txt)

make check            # debug build + C++ unit tests (2,340 assertions) + synthetic E2E, C API contract and golden regression tests (~21 s)
make test             # C++ unit tests only
make test_synthetic   # synthetic-weight greedy, C API contract and golden regression tests (no downloads; what CI runs)
make test_full        # real GPT-2 greedy and golden regression tests (needs bin/ weights)

make bench            # full benchmark suite (release build, ~3 min, needs bin/ weights)
make bench_quick      # quick benchmark (~30 s, no PyTorch baseline)
```

`BUILD=debug` (the default) keeps asserts on and builds into `bin/`. `BUILD=release` adds `-DNDEBUG` and builds into `bin/release/`. Benchmarks always use the release build.

---

## Benchmarks

Release builds, measured on two machines. Methodology is in [`bench/README.md`](./bench/README.md). Each results link is a committed JSON file.

### Apple M2 (4P + 4E cores, 16 GB): optimization history

| Version | Results | Decode ms/tok | Decode tok/s | TTFT @128 (ms) | Prefill tok/s @128 |
| --- | --- | ---: | ---: | ---: | ---: |
| Baseline (`f435414`) | [json](./bench/results/20261114-173813-f435414.json) | 101.1 | 9.9 | 2839 | 45 |
| MatMul pointer hoisting (`a468d5b`) | [json](./bench/results/20261114-234524-a468d5b.json) | 19.5 | 51.2 | 737 | 174 |
| Dot-product path: 4 accumulators (`b60def8`) | [json](./bench/results/20261115-004124-b60def8.json) | 14.1 | 71.0 | 667 | 192 |
| Last-row LM head in prefill (`7845889`) | [json](./bench/results/20261115-163411-e57bc44.json) | **14.0** | **71.5** | **450** | **284** |
| PyTorch 2.15 (HF, CPU) reference | latest json | 17.7 | 56.6 | 78 | 1647 |

Current state (`7845889`) at every prompt length:

| Backend | Prompt | TTFT (ms) | Prefill tok/s | Decode ms/tok | Decode tok/s | Avg busy cores |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Engine | 8 | 41 | 194 | 14.0 | 71.5 | 1.1 |
| Engine | 32 | 115 | 278 | 14.3 | 70.2 | 1.4 |
| Engine | 128 | 450 | 284 | 15.0 | 66.6 | 2.1 |
| PyTorch 2.15 (HF, CPU) | 8 | 28 | 290 | 17.7 | 56.6 | 1.0 |
| PyTorch 2.15 (HF, CPU) | 32 | 41 | 789 | 17.6 | 56.9 | 1.0 |
| PyTorch 2.15 (HF, CPU) | 128 | 78 | 1647 | 17.9 | 55.7 | 1.1 |

- **Decode floor:** each generated token streams all ≈498 MB of fp32 weights. At the measured 65.5 GB/s read bandwidth, the floor is ≈7.6 ms/token (≈132 tok/s). The engine reaches 54% of it, up from 7% at baseline, and now decodes ≈21% faster than PyTorch (about 43%).
- **Correctness:** the engine's greedy output matches PyTorch for all 65 generated tokens at every prompt length.
- **Memory:** peak RSS is 624 MB for the engine and 900 MB for PyTorch.
- **Kernel gap vs Accelerate (`cblas_sgemm`):**
  - All decode MatMuls (M = 1) are at or near parity: the projections are bandwidth-bound at ≈35–37 GB/s of weight streaming, and the LM head (`wte^T`) runs at ≈3.0 ms against Accelerate's ≈2.7 ms.
  - At prefill shapes (M = 32–128) the projections are 3–9× slower, and the LM head is 5–15× slower (see Known Issues).
- **These are measurements, not targets.** The harness exists to measure the effect of each roadmap item below.

### Mac Studio, Apple M1 Max (8P + 2E cores, 32 GB)

Engine code identical to `7845889` (run at `88d6fc0`), same OS, compiler and PyTorch version as the latest M2 run. [Results JSON](./bench/results/20261127-171939-88d6fc0.json).

| Backend | Prompt | TTFT (ms) | Prefill tok/s | Decode ms/tok | Decode tok/s | Avg busy cores |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Engine | 8 | 43 | 184 | 14.7 | 68.2 | 1.1 |
| Engine | 32 | 95 | 338 | 14.9 | 67.1 | 1.3 |
| Engine | 128 | 403 | 318 | 15.7 | 63.9 | 2.0 |
| PyTorch 2.15 (HF, CPU) | 8 | 25 | 316 | 26.4 | 37.9 | 1.1 |
| PyTorch 2.15 (HF, CPU) | 32 | 37 | 868 | 26.9 | 37.1 | 1.1 |
| PyTorch 2.15 (HF, CPU) | 128 | 50 | 2549 | 27.2 | 36.7 | 1.1 |

Side by side:

| Metric | M2 | M1 Max |
| --- | ---: | ---: |
| Read bandwidth, 1 thread (GB/s) | 62.3 | 61.1 |
| Read bandwidth, P-core-count threads (GB/s) | 65.5 (4) | 118.9 (8) |
| Read bandwidth, best measured (GB/s) | 65.5 | 124.1 |
| Decode floor (ms/token) | 7.6 | 4.0 |
| Engine decode, prompt 8 (ms/token) | 14.0 | 14.7 |
| Engine TTFT @128 (ms) | 450 | 403 |
| LM head kernel, M = 128 (ms): engine / Accelerate | 196 / 13.0 | 91 / 7.1 |

- **Decode does not benefit from the extra bandwidth.** Decode MatMuls run on one thread, and single-thread read bandwidth is about the same on both chips. The engine reaches 27% of the M1 Max floor, against 54% on the M2.
- **Against PyTorch:** the engine generates ≈80% more tokens/s in decode, mostly because PyTorch decodes slower here than on the M2 (26.4 vs 17.7 ms/token; cause not investigated), while PyTorch's TTFT is lower at every prompt length (50 ms against 403 ms at 128 tokens).
- **Correctness:** the engine's greedy output matches PyTorch for all 65 generated tokens at every prompt length.
- **Memory:** peak RSS is 626 MB for the engine and 911 MB for PyTorch.
- **Kernel gap vs Accelerate:** at M = 1 the engine ranges from 1.5× slower to 1.4× faster. At M = 32 the projections and LM head are 2.4–3.7× slower, and at M = 128 they are 9–13× slower.

Earlier pre-harness numbers (cold-start wall-clock over a few tokens, plus peak activation footprint) are kept in [PROTOTYPE_ARCHIVE.md](./PROTOTYPE_ARCHIVE.md#historical-measurements).

---

## Known Issues

1. **Decode MatMul runs on one thread.** `matMul2D_out` splits work across output rows and skips the thread pool when there are ≤ 4 rows. Decode has M = 1, so every decode MatMul is single-threaded (≈1 busy core above). How much this costs depends on the chip: on the M2 one thread already reads at close to the best measured bandwidth (62 of 66 GB/s), but on the M1 Max it reads at less than half of it (61 of 124 GB/s).
2. **The transposed-B (dot-product) MatMul path re-streams `B` for every row.** At M > 1 it reads all of `B` once per row of `A`, which is costly whenever `B` doesn't fit in cache: the LM head kernel benchmark (154 MB `wte`) takes ≈196 ms at M = 128 on the M2, against ≈13 ms for Accelerate. The engine no longer hits this at M > 1 with a large `B`: the LM head now runs at M = 1, and the only other transposed-B call is the attention `Q·Kᵀ` score MatMul, whose `B` (`K`, at most 1024 × 64 floats per head) fits in cache.
3. **The work split ignores P/E cores.** Rows are split into equal chunks across all cores and the call waits for every chunk, so a chunk scheduled on a slower efficiency core delays the whole call.
4. **Activation memory grows quickly with prompt length.** The arena is sized exactly per request (`Engine::stepArenaBytes`), but nothing is freed between layers within a step. A 1023-token prompt needs ≈1.0 GB of arena (≈986 MB), of which ≈61% is attention scores (one `rows × contextLen` block per head per layer); logits are one row and negligible. A decode step needs ≈1.2 MB.
5. **`TaskQueue` tracks completion with one global counter.** Two concurrent callers would each wait for the other's tasks as well, so the pool assumes a single caller at a time.
6. **`LayerNormOp` stores `eps` and `SoftmaxOp` stores `axis`, but neither is passed to its kernel.**
7. **Model dimensions are hardcoded:** `EMBEDDING_SIZE 768`, 12 layers.
8. **Pointwise kernels reach only ≈1–4 GB/s.** They are scalar loops through `TensorView::operator()`. That costs well under 1 ms per decode step today, but it will show up as the MatMuls get faster.
9. **The release build warns about unused variables** in `matMul2D_out` (`M`, `K_B`), because they are only used inside asserts.
10. **`matMul2D_out` gives wrong results for a column-strided output** (`O.stride()[1] != 1`) in the non-transposed fast path. That path checks only `B`'s column stride, but stores 4 adjacent floats into `O`. No caller in the engine passes such an output today; the unit-test sweep excludes that layout until the kernel handles it.

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
