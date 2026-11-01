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
- [x] Refactor architecture: Separate declarations (`.hpp`) from implementations (`.cpp`) for clean compilation boundaries.
- [x] Implement a global lock-free Task Queue / Thread Pool (fixes `std::thread` explosion per operation).

### Phase 2: The DAG Execution Runtime Refactor
- [x] Implement local E2E Regression Test (Greedy Decoding) on the current engine to establish a ground-truth baseline.
- [x] Standardize repository layout (`scripts/`, `tests/unit/`, `tests/e2e/`, clean portable `Makefile`).
- [ ] *Design and implement the generic ML Compiler architecture (Frontend IR -> Graph Parser -> C++ Execution Runtime).*
- [ ] *(Sub-tasks to be expanded and defined as the architectural design becomes concrete)*

### Phase 3: Cryptographic Correctness & CI/CD
- [x] Integrate lightweight C++ testing framework (`doctest.h`).
- [x] Write comprehensive unit tests for individual math micro-kernels (SIMD MatMul, transposed fallback, M=1 decoding, GELU, LayerNorm, Softmax stability, in-place variants).
- [x] Set up GitHub Actions CI pipeline with fast synthetic E2E regression test (<1s execution, zero network calls).
- [x] Retain full real GPT-2 124M greedy decoding test for manual local regression verification (`make test_full`).
- [ ] **(TODO: Medium Term)** Implement native C++ `safetensors` loader (zero-copy parsing directly into `TensorView` without Python conversion).
- [ ] **(TODO: Medium Term)** Dynamic architecture dimensions (replace `#define EMBEDDING_SIZE 768` and 12-layer hardcoding to support truly arbitrary models and tiny fixtures).

### Phase 4: Engine Generalization (Llama 3 / Gemma)
- [ ] Implement RoPE (Rotary Positional Embeddings) to replace static `wpe`.
- [ ] Implement RMSNorm to replace `LayerNorm`.
- [ ] Implement SwiGLU / SiLU activation.
- [ ] Implement Grouped-Query Attention (GQA) for modern KV-Cache memory efficiency.
- [ ] Implement robust Sampling strategies (Greedy, Temperature, Top-K/Top-P).
- [ ] Integrate simple BPE/SentencePiece Tokenizer in C++ to reduce Python frontend dependency.

### Phase 5: Quantization & Advanced Memory Efficiency
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

## Hardware & Platform Requirements
- **Architecture**: **ARM64 / AArch64 only** (Apple Silicon M1/M2/M3/M4 or Linux AArch64). The engine's core micro-kernels directly invoke ARM NEON SIMD intrinsics (`<arm_neon.h>`) for vectorized float operations. It will **not** compile or run on x86_64 (Intel/AMD) machines without porting kernels to AVX2 or using an emulation header like `neon2sse`.
- **Operating System**: macOS (Apple Silicon) or Linux (AArch64).
  - On macOS, `nlohmann-json` header path is resolved dynamically via Homebrew (`brew install nlohmann-json`).
  - On Linux (AArch64), `nlohmann-json3-dev` should be installed via system package manager (`apt install nlohmann-json3-dev`).
- **Compiler**: C++20-compliant compiler (Apple Clang 15+ or GCC 11+).
- **Python**: Python 3.9+ with packages in `scripts/requirements.txt` (PyTorch, Transformers, Tiktoken, NumPy).

---

## Notes & Design Decisions
- **Apple Silicon M2**: Accelerating math using ARM NEON intrinsics instead of AVX2.
- **Threading**: Sticking to standard C++ `<thread>` instead of OpenMP for maximum learning and true zero-dependency architecture.
- **Memory Safety**: `TensorView` abstracts memory offsets using dynamically computed C-contiguous strides to prevent buffer overflows and pointer arithmetic bugs.
