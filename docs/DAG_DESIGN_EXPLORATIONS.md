# DAG Runtime & Memory Planning: Design Explorations

This document chronicles the first-principles brainstorming, architectural explorations, and trade-off analyses conducted for the Phase 2 DAG execution runtime and memory planner refactor in the C++ GPT-2 inference engine.

---

## 1. Graph Representation: Bipartite Dataflow / SSA Graph

### Initial Hypothesis
* Structure the computation graph as a bipartite directed acyclic graph (DAG) containing two distinct types of nodes:
  * **`Value` nodes**: Represent tensors (inputs, intermediate activations, constant weights/biases).
  * **`Operation` nodes**: Represent computational kernels/transformations.
* **Strict Interleaving**: Edges strictly alternate: `Value` $\rightarrow$ `Operation` $\rightarrow$ `Value`. No two `Operation` nodes connect directly, and no two `Value` nodes connect directly.
* `Value` nodes serve as descriptors (shape, byte size), and operations consume input values to yield output values when scheduled.

### Industry Comparisons & Analysis
* **MLIR SSA Form (`Operation` $\leftrightarrow$ `Value`)**: Closely mirrors MLIR's Static Single Assignment format, where operations produce and consume SSA values.
* **TensorFlow v1 (`tf.Operation` $\leftrightarrow$ `tf.Tensor`)**: Similar separation between execution operators and data handles.
* **Alternative Explored (Op-Centric DAG / PyTorch FX)**: Nodes are operations and edges are direct references to upstream operations (with output indices). While simpler to traverse, it obscures multi-consumer tracking.

### Key Architectural Strengths Agreed Upon
1. **Clean Fan-out Tracking**: In transformer architectures, tensors frequently have high fan-out (e.g. residual connection input read by LayerNorm and later read by the residual Add). Explicit `Value` nodes make consumer counts and reference tracking trivial.
2. **Substrate for Memory Planning**: Each `Value` node naturally defines a memory liveness interval:
   * **Birth**: Producing `Operation` finishes.
   * **Death**: Final consumer `Operation` finishes.
3. **Storage Partitioning**: Allows clear classification of underlying memory:
   * **Static Weights**: Backed by zero-copy `mmap`.
   * **Transient Activations**: Backed by the scratch arena.
   * **Persistent State**: KV-Cache buffers surviving across decoding steps.

*(Note: Concrete C++ class designs, ownership semantics, and exact scheduler mechanics remain open for future brainstorming.)*

---

## 2. Operator Granularity: Attention Splitting (`QKVSplitOp` vs. `SplitOp`)

### Initial Hypothesis
* In GPT-2, the projection weight (`c_attn.weight`, shape `[768, 2304]`) combines Query, Key, and Value projections.
* Because the columns are evenly divided ($2304 / 3 = 768$), propose a specialized `QKVSplitOp` that takes the fused matrix and yields three separate output `Value` nodes ($V_Q$, $V_K$, $V_V$).

### Explorations & Trade-offs
1. **Model Generality**:
   * While GPT-2 MHA uses an exact 1:1:1 split, multi-head attention immediately partitions each 768-column matrix across 12 attention heads ($12 \times 64 = 768$) using `horSplit`.
   * A generalized `SplitOp` (specifying axis and chunk count / slice sizes) handles both QKV projection splitting and head splitting without needing architecture-specific ops.
2. **Compute vs. Metadata Operation**:
   * `TensorView::horSplit` does not copy bytes or invoke compute kernels; it creates a sliced `TensorView` adjusting pointer offset and shape while preserving row strides (zero-copy view).
   * A `SplitOp` is therefore a metadata/view operation. The producing buffer must remain alive as long as *any* of its slice `Value` nodes are actively read by downstream operations.
3. **Graph Topology (1-to-Many)**:
   * Confirmed that 1 `Operation` node yielding 3 `Value` nodes integrates cleanly: once the split op executes, all three output `Value` nodes become ready simultaneously.

### Agreement Reached
* Adopt a general `SplitOp` concept rather than a hardcoded `QKVSplitOp`.

---

## 3. Memory Reclamation: Dynamic Runtime vs. Static Compile-Time

### Initial Hypothesis (Dynamic Allocator with FIFO Ring Buffer)
* The scheduler allocates memory for output `Value` nodes just before an `Operation` runs.
* When the operation completes, the scheduler decrements the consumer count of input `Value` nodes. When the count drops to zero, the scheduler triggers memory reclamation.
* Proposal for `ArenaAllocator`: Allocate from the tail and reclaim from the head (FIFO ring buffer), sorting allocations by death time with longest-living tensors placed on the right.

### Edge Cases & Pitfalls Identified
1. **The FIFO Assumption**:
   * A FIFO ring buffer assumes the *oldest* allocation dies *first*.
   * In a transformer block, the residual tensor is allocated first (at the head), followed by short-lived intermediate tensors. The intermediate tensors die immediately, but the residual must stay alive until the end of the block.
   * Because the oldest allocation is pinned at the head, the head cannot advance, breaking the ring buffer model.
2. **The "Holes" Dilemma at Runtime**:
   * Freeing short-lived tensors located between pinned tensors leaves intermediate unallocated holes.
   * Handling these holes dynamically at runtime requires free-lists, searching for best-fit blocks, splitting, and coalescing. This converts a fast $O(1)$ bump arena into a full-blown heap allocator (like `malloc`), reintroducing memory fragmentation and runtime latency.

### The Conceptual Breakthrough: Static Liveness & 2D Packing
* **The Determinism Property**: LLM graph topology and tensor byte sizes are completely known ahead of time.
* **The Non-Overlap Rule**:
  > Two tensors can safely share the exact same memory offset if and only if their active lifetimes `[birth, death]` do not overlap in time.
* **2D Box Packing Model**:
  * **X-axis (Time)**: Execution step interval `[birth, death]`.
  * **Y-axis (Memory)**: Byte offset and size.
  * Tensors from different transformer layers (e.g. Layer 0 residual vs. Layer 1 residual) never overlap in time and can immediately reuse the exact same byte offset (e.g. Offset 0).

### Agreement Reached
* Abandon dynamic runtime hole management in favor of a **static compile-time memory planner**.
* The planner computes `[birth, death]` intervals for all transient `Value` nodes and pre-assigns fixed byte offsets inside a single flat scratch arena. At runtime, memory binding is pure pointer arithmetic (`base_ptr + offset`) with zero dynamic allocations or frees.

---

## 4. Sequence Length Variation ($N$) & Memory Planning Strategies

### The Challenge: The Dynamic Shapes Dilemma
* During the **Prefill Phase**, the engine processes a prompt of $N$ tokens (e.g., $N=3$ or $N=128$), producing activations sized $[N, \text{dim}]$.
* During the **Autoregressive Decode Phase**, the engine processes strictly **$N = 1$ token** per step, keeping activations sized $[1, \text{dim}]$.
* Changing $N$ alters tensor byte sizes, shifting the dimensions of the rectangles on the memory timeline.

### Strategies Explored

#### Strategy A: Worst-Case Sizing (Max-Sequence Planning)
* Statically size all buffers for the maximum sequence length (e.g., $N_{\text{max}} = 1024$).
* Simple and robust, but over-allocates physical memory for small prompts.

#### Strategy B: Dedicated Prefill Plan vs. Decode Plan
* Separate the plan for prompt ingestion from the autoregressive generation loop.
* **Analysis on Arena Size**: 
  * Does Strategy B reduce the initial arena allocation if Prefill is sized for $N_{\text{max}}$? 
  * *Finding*: The maximum physical DRAM capacity reserved from the OS remains bounded by $\max(\text{Peak}_{\text{prefill}}, \text{Peak}_{\text{decode}})$.
  * However, because Prefill and Decode **never run simultaneously** (temporal mutual exclusion), Decode can reuse the exact same arena memory starting back at Byte 0.
* **The Hardware Cache Locality Win**:
  * For $N=1$, peak activation memory in GPT-2 is only **~580 KB**.
  * A dedicated Decode plan packs all working memory into this tight 580 KB footprint at the base of the arena.
  * 580 KB fits entirely within the CPU's on-chip L2/L3 cache (e.g. Apple Silicon M2 shared cache), meaning the autoregressive loop runs with near-100% cache residency, avoiding main DRAM roundtrips.

#### Strategy C: Runtime Shape-Specialized Plan Caching
* Maintain an LRU cache (`std::unordered_map<size_t, MemoryPlan>`) and dynamically compile a plan the first time a new sequence length $N$ is seen.
* **User Analysis & Validation**:
  * Highly applicable to persistent server/chat environments (e.g. vLLM) where $N$ spans thousands of variations and static enumeration is impossible.
  * For a self-contained local engine with $N \le 1024$, runtime caching adds unnecessary moving parts (hash maps, cache misses, invoking planner algorithms during `generate()`).
  * Furthermore, because 99.9% of generation time is spent in the decode loop ($N=1$), caching plans for every $N \in [1, 1024]$ provides no tangible benefit over a dedicated $N=1$ plan.

### Agreement Reached
* Dismiss Strategy C for our zero-dependency engine.
* Focus on an ultra-optimized static plan for Decode ($N=1$) that maximizes CPU cache locality, alongside a Prefill phase that reuses that same arena buffer space.

---

## 5. Summary of Current Consensus & Next Steps

| Topic | Consensus / Decision | Rationale |
| :--- | :--- | :--- |
| **Graph Model** | Bipartite Dataflow (`Value` $\leftrightarrow$ `Operation`) | Explicit fan-out tracking and natural substrate for tensor lifetime intervals. |
| **Split Op** | General `SplitOp` (axis & partitions) | Reusable across QKV split and attention head splitting; treats slicing as zero-copy views. |
| **Memory Allocation** | Static Compile-Time Memory Planner | Eliminates runtime allocator complexity/fragmentation; achieves zero runtime allocation overhead. |
| **Lifespan Reuse** | Non-overlapping Interval Packing | Tensors with disjoint `[birth, death]` intervals share the exact same arena byte offsets. |
| **Dynamic Shapes** | Dedicated Decode ($N=1$) Plan + Shared Prefill | Decode working set (~580 KB) remains 100% CPU cache resident; prefill reuses the same memory space. |

### Pending Items (To Be Brainstormed in Future Steps)
* Concrete C++ class representations and memory ownership for `ValueNode` and `OpNode`.
* Detailed topological sorting / scheduling mechanics.
* Exact algorithm for the 2D interval memory packing.
* Mechanics of integrating the static memory plan into the existing `ArenaAllocator`.
