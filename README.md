# C++ AI Inference Engine (GPT-2 124M)

## Goal
Architect a zero-dependency C++ inference engine for a 124M parameter transformer (GPT-2), utilizing advanced C++ paradigms to isolate tensor operations and ensure strict memory safety. 

Key objectives:
- **Zero-Copy Memory**: Custom memory allocator using memory-mapped files (`mmap`) for tensor weights.
- **High Performance**: Optimized core matrix multiplication via multi-threading (`std::thread` pool) and ARM NEON SIMD instructions.
- **Zero Dependencies**: Pure modern C++ (C++20) without relying on large external tensor frameworks.

---

## Status Tracker (Production & Scaling Phase)

*Note: The initial 0-to-1 prototype phases (Tensor memory, basic MatMul, GPT-2 assembly, KV-Cache) are completed. See [PROTOTYPE_ARCHIVE.md](./PROTOTYPE_ARCHIVE.md) for the historical completion list.*

### Phase 1: Architectural Decoupling & Concurrency
- [ ] Refactor architecture: Separate declarations (`.hpp`) from implementations (`.cpp`) for clean compilation boundaries.
- [ ] Implement a global lock-free Task Queue / Thread Pool (fixes `std::thread` explosion per operation).
- [ ] Abstract hardcoded GPT-2 logic into a generic `ModelLoader` class to support arbitrary graphs.

### Phase 2: Cryptographic Correctness & CI/CD
- [ ] Integrate lightweight C++ testing framework (Catch2 or GoogleTest).
- [ ] Write unit tests for individual math micro-kernels (SIMD MatMul vs. Naive MatMul).
- [ ] Set up GitHub Actions CI pipeline (auto-build on commit).
- [ ] Implement automated regression test: forward pass on a tiny model asserting PyTorch logit parity (1e-4 tolerance).

### Phase 3: Engine Generalization (Llama 3 / Gemma)
- [ ] Implement RoPE (Rotary Positional Embeddings) to replace static `wpe`.
- [ ] Implement RMSNorm to replace `LayerNorm`.
- [ ] Implement SwiGLU / SiLU activation.
- [ ] Implement Grouped-Query Attention (GQA) for modern KV-Cache memory efficiency.
- [ ] Implement robust Sampling strategies (Greedy, Temperature, Top-K/Top-P).
- [ ] Integrate simple BPE/SentencePiece Tokenizer in C++ to reduce Python frontend dependency.

### Phase 4: Quantization & Advanced Memory Efficiency
- [ ] Design Int8 / Int4 quantization formats for tensor weights.
- [ ] Write custom SIMD kernels to multiply quantized integers and dequantize on the fly.
- [ ] (Advanced) Offload MatMul to Apple Silicon M2 GPU using Metal Performance Shaders (MPS).

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
