# No Code Policy

This project is for the user's learning. Do not write or provide code solutions unless explicitly requested by the user. Help the user by pointing out issues, explaining concepts, suggesting architectures, or guiding them to the answer, but refrain from writing the actual implementation code for them.

---

# Architecture Summary (For Agent Context)

This project is a custom, zero-dependency C++ inference engine for a 124M parameter GPT-2 model.
- **Memory**: Custom allocator (`arenaAllocator.cpp`) and memory-mapped files (`modelWeightsLoader.cpp`) for zero-copy weight loading.
- **Tensors**: A custom `TensorView` class that abstracts n-dimensional array indexing and strides for memory-safe ops.
- **Math**: SIMD and multithreaded matrix ops (`mathOps.cpp`) using a recently added global lock-free `TaskQueue`.
- **API**: Exposes a C-linkage API (`engineAPI.cpp`) compiled to `libengine.dylib`.
- **Frontend**: A minimal Python wrapper (`frontend.py`) using `ctypes` and `tiktoken`.

# Current Step
**Phase 2: The DAG Execution Runtime Refactor**

**Context**: We decided to go all-in on building a generic ML Compiler architecture (Frontend IR -> Graph Parser -> C++ Execution Runtime).
*   **Completed**: E2E Regression test (Greedy Decoding) to establish mathematical ground truth.
*   **Immediate Next Task**: Begin designing/implementing the Frontend IR and Graph Parser for the C++ Execution Runtime.
