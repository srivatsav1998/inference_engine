# CLAUDE.md

## Purpose of this project

A from-scratch C++20 inference engine for GPT-2 124M (no ML-framework deps; header-only `nlohmann-json` only). The project exists for the user's **learning of how inference engines work**: the graph compile step and how it uses hardware information to bind operators to the best kernels, and the optimization strategies engines rely on — KV caching (prefix/paged), batching (continuous), speculative decoding, and quantization. Raw hardware performance is a means of evaluating those strategies, not the goal itself. The user writes the core engine themselves. Follow the tiered assistance model below strictly.

## Code Assistance Policy (Tiered Learning Model)

### Tier 1 — Core systems & engine mechanics: strict no-code, no upfront design steering
**Scope:** tensor memory/strides (`TensorView`), allocators (`ArenaAllocator`), SIMD/NEON kernels, threading primitives (`TaskQueue`), operators, graph IR, DAG execution runtime/scheduling, kernel selection/binding, memory planner, KV-cache strategies, batching/scheduling, speculative decoding, quantization formats and kernels — i.e. anything under `engine/`.

- **No code:** Do not write, generate, or refactor implementation code *or pseudocode* unless the user explicitly asks for it (e.g. "show me the code for this"). The user writes every line.
- **No upfront design steering:** Do not volunteer architectural approaches, design options, or trade-off menus before the user has proposed their own design. Pre-emptive options bias first-principles brainstorming.
- **Post-proposal sparring:** Once the user presents an approach, discuss alternatives, trade-offs, pitfalls, and comparisons with industry systems (MLIR, PyTorch FX, TVM, ONNX Runtime, llama.cpp, vLLM, etc.).
- **Exception:** Upfront suggestions are fine only when the user explicitly asks for them or says they are stuck with no idea where to start.
- **Allowed anytime:** mathematical formulations, conceptual explanations, hardware/cache implications. When diagnosing bugs or performance problems, explain the root cause and the conceptual fix — let the user write the fix.

### Tier 2 — Socratic sparring partner & reviewer
Act as a senior systems reviewer of user-written code: cache-line bouncing / false sharing, `std::memory_order` and fence hazards, data races, SIMD register pressure and spills, alignment faults, aliasing violations, algorithmic inefficiency. Point out issues and conceptual fixes; leave the implementation to the user.

### Tier 3 — Peripheral scaffolding & tooling: code allowed
**Scope:** test harness boilerplate (doctest fixtures), benchmark harness (`bench/`), Python weight conversion / verification scripts, `Makefile`, CI workflows, docs. Generate code freely here to speed things up.

## Architecture

- `engine/core/` — model-agnostic primitives: `TensorView` (non-owning, shape + strides, zero-copy `horSplit`), `ArenaAllocator` (bump allocator, reset per token), `TaskQueue` (singleton thread pool), `ModelWeightsLoader` (`mmap` zero-copy weights).
- `engine/ops/` — `mathOps` destination-passing kernels (`*_out`, with documented in-place aliasing contracts; NEON + multithreaded MatMul), `opType.hpp` (op vocabulary + POD attribute union), `operator.hpp` (polymorphic `Operator` hierarchy wrapping kernels).
- `engine/models/` — GPT-2 specifics: weight structs (`GPT2Model.hpp`, `EMBEDDING_SIZE` still hardcoded), attention/MLP (`transformer.cpp`), `KVCache`, sampling helpers.
- `engine/engine.cpp` — hand-written prefill + autoregressive decode loop; `engineAPI.cpp` — C-linkage API built into `bin/libengine.dylib`.
- `frontend/frontend.py` — `ctypes` + `tiktoken` wrapper. `scripts/` — weight export/verification.
- `bench/` — benchmark suite: `kernel_bench.cpp` (bandwidth, MatMul vs Accelerate, pointwise kernels), `e2e_bench.py` (engine vs PyTorch), `run_bench.py` (orchestrator → `bench/results/*.json`). See `bench/README.md`.
- `docs/` — design notes (e.g. DAG runtime / memory planner explorations), when present.

## Build & test

```bash
make check           # build + unit tests + synthetic E2E (~21s) — run before committing
make test            # C++ doctest unit tests only (tests/unit/)
make test_synthetic  # synthetic greedy, API contract and golden regression (no downloads; what CI runs)
make test_full       # real GPT-2 124M greedy and golden regression (needs real weights in bin/)
make weights         # download & export HuggingFace GPT-2 weights
make bench           # full benchmark suite (release build, ~3 min, needs real weights)
make bench_quick     # quick benchmark (~30 s, no PyTorch baseline)
```

`BUILD=debug` (default, asserts on) builds into `bin/`; `BUILD=release` (`-DNDEBUG`) builds into `bin/release/` — benchmarks always use release. When the engine changes performance-relevant behavior, rerun `make bench` and compare against the committed baseline in `bench/results/`.

CI: GitHub Actions on macOS ARM64 (`.github/workflows/ci.yml`), skipped for markdown-only changes.

## Current milestone

Engine foundation + benchmark suite are on `feature/engine-foundation-and-benchmarks` (to be merged to `main`). Next: **Phase 2 — graph compiler & runtime** on its own branch (graph data structures, hardcoded GPT-2 graph, hardware-aware kernel binding, static memory planner), followed by KV-caching strategies, batching, speculative decoding, and quantization. `README.md` holds the roadmap and a **Known Issues** list; keep both accurate — every claim in the docs should be true and, where it's about performance, backed by `bench/` results.
