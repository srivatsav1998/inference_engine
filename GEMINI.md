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

# Next Step (Parked)
The user has paused the project and will resume later.
**Immediate Next Task**: Begin Phase 2: **The DAG Execution Runtime Refactor**. 
**Context**: We decided to go all-in on building a generic ML Compiler architecture (Frontend IR -> Graph Parser -> C++ Execution Runtime).
**First Action Item upon return**: Implement an E2E Regression test locally on the current hardcoded engine using Greedy Decoding (temperature = 0.0) so we have a mathematical ground truth to verify against before the C++ engine is gutted.
