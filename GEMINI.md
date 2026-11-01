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

**Context**: Establishing automated testing (micro-kernels, deterministic greedy decoding) and a CI/CD pipeline before designing the DAG Execution Runtime.
*   **Completed**: Cleaned repo structure: centralized `scripts/`, organized `tests/unit/` and `tests/e2e/`, portable `Makefile`, cleaned up root `.gitignore`, discarded stale experimental branches.
*   **Immediate Next Task**: Add GitHub Actions workflow for automated CI/CD and verify local test suite execution.
