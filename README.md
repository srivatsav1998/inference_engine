# C++ AI Inference Engine (GPT-2 124M)

## Goal
Architect a zero-dependency C++ inference engine for a 124M parameter transformer (GPT-2), utilizing advanced C++ paradigms to isolate tensor operations and ensure strict memory safety. 

Key objectives:
- **Zero-Copy Memory**: Custom memory allocator using memory-mapped files (`mmap`) for tensor weights.
- **High Performance**: Optimized core matrix multiplication via multi-threading (`std::thread` pool) and ARM NEON SIMD instructions.
- **Zero Dependencies**: Pure modern C++ (C++20) without relying on large external tensor frameworks.

---

## Status Tracker

### Phase 1: Tensor Architecture & Memory Safety
- [x] Extract HuggingFace GPT-2 weights to raw binary (`float32`).
- [x] Create metadata JSON (shapes and byte offsets).
- [x] Implement POSIX `mmap` wrapper (`MMAPHelperSP`).
- [x] Implement zero-copy `TensorView` class (with multi-dimensional strides and variadic indexing).
- [x] Parse JSON metadata into C++ and instantiate `TensorView`s for all weights.
- [x] Build an Activation Memory Arena (bump allocator for dynamic runtime buffers).

### Phase 2: Core Math Operations & Micro-kernels
- [x] Implement mathematical primitives (Element-wise Add, Multiply, Scalar ops).
- [x] Implement `LayerNorm` (standard layer normalization with epsilon).
- [x] Implement `GELU` (Gaussian Error Linear Unit) activation function.
- [x] Implement `Softmax` for attention scores.
- [x] Implement Naive Matrix Multiplication (`MatMul`).
- [x] Optimize `MatMul` with ARM NEON SIMD (Vectorization).
- [x] Optimize `MatMul` with C++ `std::thread` pool (Multi-threading).

### Phase 3: Transformer Block Components
- [x] Implement Token Embedding (`wte`) & Positional Embedding (`wpe`) Lookup.
- [x] Implement Multi-Head Self-Attention (QKV splitting, Attention mask).
- [ ] Implement KV-Cache (Key-Value caching for autoregressive generation).
- [x] Implement MLP / Feed-Forward Network.
- [x] Assemble single Transformer Block (Pre-LayerNorm architecture).

### Phase 4: Full Engine Assembly & Token Generation
- [x] Chain 12 Transformer Blocks together.
- [x] Implement final `LayerNorm` and language modeling head (projection to vocab size).
- [x] Implement generation loop (Autoregressive).
- [ ] Implement Sampling strategies (Greedy, Temperature, Top-K/Top-P).
- [ ] (Optional) Integrate simple BPE Tokenizer or accept CLI token IDs.

### Phase 5: Production Polish & Optimization
- [ ] Refactor C++ concurrency to use a global Thread Pool (eliminates Thread Explosion).
- [x] Build a Python Frontend / API binding (via `pybind11` or `ctypes`) to handle Tokenization and UI.
- [ ] Architectural Refactor: Separate declarations (`.hpp`) from implementations (`.cpp`) for cleaner code.
- [ ] Expand Unit Testing suite to cover Transformer components and KV-Cache.
- [ ] Add Doxygen-style documentation across the codebase.
- [ ] (Advanced) Metal Backend: Offload `MatMul` to the M2 GPU using Apple Metal Performance Shaders (MPS).

### Phase 6: Modern LLM Architecture (Llama 3 / Gemma)
- [ ] Implement RoPE (Rotary Positional Embeddings) to replace static `wpe`.
- [ ] Implement RMSNorm to replace `LayerNorm`.
- [ ] Implement SwiGLU / SiLU activation to replace `GELU`.
- [ ] Implement Grouped-Query Attention (GQA) for KV-Cache memory efficiency.
- [ ] Build Int8 / Int4 Quantization micro-kernels to run 7B+ models in limited RAM.

---

## Benchmarks
| Engine Version | Optimization Level | CPU Threads | Hardware | Tokens | Time (s) |
| --- | --- | --- | --- | --- | --- |
| v0.1 (Naive C++) | `-O3` | 1 | Apple Silicon M2 | 4 | ~4.37s |
| v0.2 (`std::thread`) | `-O3` | 8 | Apple Silicon M2 | 4 | ~1.26s |
| v0.3 (NEON SIMD) | `-O3` | 8 | Apple Silicon M2 | 4 | ~0.94s |
| v1.0 (Autoregressive Loop) | `-O3` | 8 | Apple Silicon M2 | 4 + 5 generated | ~7.52s |
| v1.1 (KV-Cache) | `-O3` | 8 | Apple Silicon M2 | 4 + 8 generated | ~6.37s |


---

## Notes & Design Decisions
- **Apple Silicon M2**: Accelerating math using ARM NEON intrinsics instead of AVX2.
- **Threading**: Sticking to standard C++ `<thread>` instead of OpenMP for maximum learning and true zero-dependency architecture.
- **Memory Safety**: `TensorView` abstracts memory offsets using dynamically computed C-contiguous strides to prevent buffer overflows and pointer arithmetic bugs.
