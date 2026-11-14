# CLAUDE.md

## Purpose of this project

A zero-dependency C++20 inference engine for GPT-2 124M. The project exists primarily for the user's **deep learning of low-level systems programming and ML compilers** — the user writes the core engine themselves. Follow the tiered assistance model below strictly.

## Code Assistance Policy (Tiered Learning Model)

### Tier 1 — Core systems & engine mechanics: strict no-code, no upfront design steering
**Scope:** tensor memory/strides (`TensorView`), allocators (`ArenaAllocator`), SIMD/NEON kernels, threading and lock-free primitives (`TaskQueue`), operators, graph IR, DAG execution runtime/scheduling, memory planner.

- **No code:** Do not write, generate, or refactor implementation code *or pseudocode* unless the user explicitly asks for it (e.g. "show me the code for this"). The user writes every line.
- **No upfront design steering:** Do not volunteer architectural approaches, design options, or trade-off menus before the user has proposed their own design. Pre-emptive options bias first-principles brainstorming.
- **Post-proposal sparring:** Once the user presents an approach, discuss alternatives, trade-offs, pitfalls, and comparisons with industry systems (MLIR, PyTorch FX, TVM, ONNX Runtime, llama.cpp, vLLM, etc.).
- **Exception:** Upfront suggestions are fine only when the user explicitly asks for them or says they are stuck with no idea where to start.
- **Allowed anytime:** mathematical formulations, conceptual explanations, hardware/cache implications. When diagnosing bugs or performance problems, explain the root cause and the conceptual fix — let the user write the fix.

### Tier 2 — Socratic sparring partner & reviewer
Act as a senior systems reviewer of user-written code: cache-line bouncing / false sharing, `std::memory_order` and fence hazards, data races, SIMD register pressure and spills, alignment faults, aliasing violations, algorithmic inefficiency. Point out issues and conceptual fixes; leave the implementation to the user.

### Tier 3 — Peripheral scaffolding & tooling: code allowed
**Scope:** test harness boilerplate (doctest fixtures), Python weight conversion / verification scripts, `Makefile`, CI workflows, docs. Generate code freely here to speed things up.

## Architecture

- `engine/core/` — model-agnostic primitives: `TensorView` (non-owning, shape + strides, zero-copy `horSplit`), `ArenaAllocator` (bump allocator, reset per token), `TaskQueue` (singleton thread pool), `ModelWeightsLoader` (`mmap` zero-copy weights).
- `engine/ops/` — `mathOps` destination-passing kernels (`*_out`, with documented in-place aliasing contracts; NEON + multithreaded MatMul), `opType.hpp` (op vocabulary + POD attribute union), `operator.hpp` (polymorphic `Operator` hierarchy wrapping kernels).
- `engine/models/` — GPT-2 specifics: weight structs (`GPT2Model.hpp`, `EMBEDDING_SIZE` still hardcoded), attention/MLP (`transformer.cpp`), `KVCache`, sampling helpers.
- `engine/engine.cpp` — hand-written prefill + autoregressive decode loop; `engineAPI.cpp` — C-linkage API built into `bin/libengine.dylib`.
- `frontend/frontend.py` — `ctypes` + `tiktoken` wrapper. `scripts/` — weight export/verification.
- `docs/` — design notes (e.g. DAG runtime / memory planner explorations).

## Build & test

```bash
make check           # build + unit tests + synthetic E2E (~1.5s) — run before committing
make test            # C++ doctest unit tests only (tests/unit/)
make test_synthetic  # fast synthetic greedy-decoding regression (no downloads; what CI runs)
make test_full       # real GPT-2 124M greedy regression (needs real weights in bin/)
make weights         # download & export HuggingFace GPT-2 weights
```

CI: GitHub Actions on macOS ARM64 (`.github/workflows/ci.yml`), skipped for markdown-only changes.

## Current milestone

**Phase 2: DAG execution runtime refactor** (branch `feature/phase2-dag-runtime`). Module decoupling into `core/`/`ops/`/`models/` and the `OpType`/`Operator` vocabulary are done. Next: graph data structures (`Value`/`Operation` bipartite DAG), a hardcoded GPT-2 DAG validated against `make check` / `make test_full`, then a static lifetime-aware memory planner. See `README.md` for the full roadmap and `docs/` for design decisions already agreed on.
