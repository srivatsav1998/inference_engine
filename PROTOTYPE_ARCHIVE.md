# Prototype Phase (Completed)

This document tracks the completed milestones from the initial 0-to-1 build of the C++ Inference Engine (GPT-2 124M architecture). These phases successfully established a zero-dependency, memory-safe foundation with basic CPU optimizations.

## Phase 1: Tensor Architecture & Memory Safety
- [x] Extract HuggingFace GPT-2 weights to raw binary (`float32`).
- [x] Create metadata JSON (shapes and byte offsets).
- [x] Implement POSIX `mmap` wrapper (`MMAPHelperSP`).
- [x] Implement zero-copy `TensorView` class (with multi-dimensional strides and variadic indexing).
- [x] Parse JSON metadata into C++ and instantiate `TensorView`s for all weights.
- [x] Build an Activation Memory Arena (bump allocator for dynamic runtime buffers).

## Phase 2: Core Math Operations & Micro-kernels
- [x] Implement mathematical primitives (Element-wise Add, Multiply, Scalar ops).
- [x] Implement `LayerNorm` (standard layer normalization with epsilon).
- [x] Implement `GELU` (Gaussian Error Linear Unit) activation function.
- [x] Implement `Softmax` for attention scores.
- [x] Implement Naive Matrix Multiplication (`MatMul`).
- [x] Optimize `MatMul` with ARM NEON SIMD (Vectorization).
- [x] Optimize `MatMul` with C++ `std::thread` pool (Multi-threading).

## Phase 3: Transformer Block Components
- [x] Implement Token Embedding (`wte`) & Positional Embedding (`wpe`) Lookup.
- [x] Implement Multi-Head Self-Attention (QKV splitting, Attention mask).
- [x] Implement KV-Cache (Key-Value caching for autoregressive generation).
- [x] Implement MLP / Feed-Forward Network.
- [x] Assemble single Transformer Block (Pre-LayerNorm architecture).

## Phase 4: Full Engine Assembly & Token Generation
- [x] Chain 12 Transformer Blocks together.
- [x] Implement final `LayerNorm` and language modeling head (projection to vocab size).
- [x] Implement generation loop (Autoregressive).

## Phase 5: Initial Interfaces
- [x] Build a Python Frontend / API binding (via `ctypes`) to handle Tokenization and UI.

## Historical Measurements

These numbers were recorded by hand before the benchmark suite (`bench/`) existed. They are kept for history but are **not comparable** to current results:
- They are single cold-start wall-clock runs over a handful of tokens, mixing model load, prefill and decode.
- They used the default build, with asserts on.

See the README's Benchmarks section for current, reproducible numbers.

### Execution Latency (cold start, wall clock)
| Engine Version | Optimization Level | CPU Threads | Hardware | Tokens | Time (s) |
| --- | --- | --- | --- | --- | --- |
| v0.1 (Naive C++) | `-O3` | 1 | Apple Silicon M2 | 4 | ~4.37s |
| v0.2 (`std::thread`) | `-O3` | 8 | Apple Silicon M2 | 4 | ~1.26s |
| v0.3 (NEON SIMD) | `-O3` | 8 | Apple Silicon M2 | 4 | ~0.94s |
| v1.0 (Autoregressive Loop) | `-O3` | 8 | Apple Silicon M2 | 4 + 5 generated | ~7.52s |
| v1.1 (KV-Cache) | `-O3` | 8 | Apple Silicon M2 | 4 + 8 generated | ~6.37s |

### Peak Activation Memory Footprint (Arena Allocator)
Peak activation bytes per generation step (GPT-2 124M, prompt length N = 3, greedy decoding). The measuring instrumentation is not part of the repo.

| Execution Phase | Baseline (v1.1 Prototype) | Destination-Passing, Separate Buffers | Destination-Passing, In-Place Aliasing | Net vs. Baseline |
| :--- | :--- | :--- | :--- | :--- |
| **Prefill** (N = 3) | 1,943,536 B (~1.94 MB) | 3,517,648 B (~3.52 MB) | **1,734,352 B (~1.73 MB)** | **-10.8%** |
| **Decode step 1** (N = 1, pos 3) | 651,856 B (~651.9 KB) | 1,172,560 B (~1.17 MB) | **578,128 B (~0.58 MB)** | **-11.3%** |
| **Decode step 2** (N = 1, pos 4) | 654,160 B (~654.2 KB) | 1,179,472 B (~1.18 MB) | **580,432 B (~0.58 MB)** | **-11.3%** |

The `*_out` refactor removed all allocation from inside kernels. Reusing the `normX` buffer across layers and doing residual additions in place roughly halved the peak decode activation footprint compared with naive destination-passing.
