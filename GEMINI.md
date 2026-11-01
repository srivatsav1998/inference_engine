# Code Assistance Policy (Tiered Learning Model)

This project is primarily for the user's deep learning of low-level systems programming and ML compilers. Follow this tiered model for code assistance:

### Tier 1: Core Systems & Engine Mechanics (Strict No-Code)
- **Scope**: Tensor memory/strides (`TensorView`), allocators (`arenaAllocator`), SIMD/NEON kernels, threading/lock-free queues (`TaskQueue`), graph IR, and DAG execution runtime/scheduling.
- **Rule**: **DO NOT** write, generate, or refactor implementation code or pseudocode unless explicitly commanded by the user. Guide conceptually, suggest architectural approaches, explain hardware/cache implications, and let the user write every line.

### Tier 2: AI as Socratic Sparring Partner & Reviewer
- Act as a senior systems reviewer. Critique user implementations for cache-line bouncing, memory ordering hazards (`std::memory_order`), register spills, alignment faults, or algorithmic inefficiencies. Point out issues and conceptual fixes, leaving the fix implementation to the user.

### Tier 3: Peripheral Scaffolding & Tooling (Code Allowed)
- **Scope**: Test harness boilerplate (Catch2/GoogleTest fixtures), data conversion scripts (e.g., Python weight extraction / safetensors dumpers), build scripts (Makefile/CMake), and CI/CD pipelines.
- **Rule**: Code generation is allowed for these plumbing/boilerplate tasks to accelerate development without compromising core learning.

---

# Architecture Summary (For Agent Context)

This project is a custom, zero-dependency C++ inference engine for a 124M parameter GPT-2 model.
- **Memory**: Custom allocator (`arenaAllocator.cpp`) and memory-mapped files (`modelWeightsLoader.cpp`) for zero-copy weight loading.
- **Tensors**: A custom `TensorView` class that abstracts n-dimensional array indexing and strides for memory-safe ops.
- **Math**: SIMD and multithreaded matrix ops (`mathOps.cpp`) using a global lock-free `TaskQueue`.
- **API**: Exposes a C-linkage API (`engineAPI.cpp`) compiled to `bin/libengine.dylib`.
- **Frontend**: A minimal Python wrapper (`frontend/frontend.py`) using `ctypes` and `tiktoken`.
- **Scripts**: Model weights extraction & verification scripts in `scripts/`.
- **Tests**: Unified under `tests/unit/` (C++ micro-kernels) and `tests/e2e/` (greedy decoding regression).

# Current Step
**Phase 2: Repository Hardening & Testing Infrastructure (Pre-Refactor)**

**Context**: Robust automated testing and fast CI/CD pipeline established before embarking on the DAG Execution Runtime refactor.
*   **Completed**: Cleaned repo structure, portable `Makefile`, single-header `doctest.h` test suite (2,157 assertions).
*   **Completed**: Fast deterministic synthetic E2E regression test for CI (<1s runtime, zero network downloads).
*   **Completed**: Retained full real GPT-2 greedy decoding regression test for local runs (`make test_full`).
*   **Completed**: GitHub Actions workflow running on macOS ARM64.
*   **Immediate Next Task**: Begin designing the DAG Execution Runtime (Frontend IR -> Graph Node & Topology -> Execution Scheduler).

# Roadmap / Future Enhancements
*   **Direct C++ Safetensors Loader**: Implement native `.safetensors` zero-copy ingestion via `mmap` + `nlohmann-json` to bypass Python conversion scripts.
*   **Dynamic Architecture Dimensions**: Parameterize `EMBEDDING_SIZE` and `num_layers` in C++ engine to support arbitrary LLM architectures and micro fixtures.
