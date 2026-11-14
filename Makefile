CXX      := g++
CXXFLAGS := -std=c++20 -Wall -Wextra -g -O3 -MMD -MP -fPIC

# Portable nlohmann-json header discovery
BREW_JSON_INC := $(shell brew --prefix nlohmann-json 2>/dev/null)/include
ifneq ($(wildcard $(BREW_JSON_INC)),)
    CXXFLAGS += -I$(BREW_JSON_INC)
endif

# Expose engine headers
CXXFLAGS += -Iengine -Iengine/core -Iengine/ops -Iengine/models

BIN_DIR  := bin

# Build mode: `debug` (default, asserts ON — used by tests) or `release` (-DNDEBUG — used by benchmarks).
# Release artifacts live in separate directories so both builds can coexist.
BUILD ?= debug
ifeq ($(BUILD),release)
    CXXFLAGS += -DNDEBUG
    LIB_DIR  := $(BIN_DIR)/release
    OBJ_DIR  := $(BIN_DIR)/obj-release
else ifeq ($(BUILD),debug)
    LIB_DIR  := $(BIN_DIR)
    OBJ_DIR  := $(BIN_DIR)/obj
else
    $(error BUILD must be 'debug' or 'release', got '$(BUILD)')
endif

# Shared library extension detection
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
    LIB_EXT := dylib
else
    LIB_EXT := so
endif
TARGET := $(LIB_DIR)/libengine.$(LIB_EXT)

SRCS     := $(wildcard engine/*.cpp) $(wildcard engine/*/*.cpp)
OBJS     := $(patsubst engine/%.cpp, $(OBJ_DIR)/engine/%.o, $(SRCS))

# Test source & binary
TEST_UNIT_SRC := tests/unit/test_mathOps.cpp
TEST_UNIT_BIN := $(BIN_DIR)/test_mathOps
TEST_OBJS     := $(OBJ_DIR)/engine/core/arenaAllocator.o \
                 $(OBJ_DIR)/engine/core/tensorView.o \
                 $(OBJ_DIR)/engine/ops/mathOps.o \
                 $(OBJ_DIR)/engine/core/taskQueue.o

# Python executable (prefer .venv if available)
PYTHON := $(shell if [ -f .venv/bin/python3 ]; then echo .venv/bin/python3; else echo python3; fi)

# Default target: build the engine library
all: $(TARGET)

$(TARGET): $(OBJS)
	@mkdir -p $(LIB_DIR)
	$(CXX) $(CXXFLAGS) -shared $^ -o $@

$(OBJ_DIR)/engine/%.o: engine/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Build and run C++ unit tests (doctest)
test: $(TARGET) $(TEST_OBJS)
	@mkdir -p $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $(TEST_UNIT_SRC) $(TEST_OBJS) -o $(TEST_UNIT_BIN)
	@echo "Running C++ Unit Tests..."
	@./$(TEST_UNIT_BIN)

# Fast E2E deterministic synthetic regression test (Zero download, < 1 second)
test_synthetic: $(TARGET)
	@echo "Running Synthetic E2E Regression Test..."
	@$(PYTHON) tests/e2e/test_synthetic_regression.py

# Full E2E greedy decoding regression test (Requires real HuggingFace weights)
test_full: $(TARGET)
	@echo "Running Full Real GPT-2 E2E Greedy Decoding Regression Test..."
	@$(PYTHON) tests/e2e/test_greedy_decoding.py

# Default E2E points to fast synthetic test
test_e2e: test_synthetic

# Run default fast test suite (C++ micro-kernels + Synthetic E2E)
check: test test_synthetic

# ----------------------------------------------------------------------------
# Benchmarks (always built with BUILD=release; see bench/README.md)
# ----------------------------------------------------------------------------
KERNEL_BENCH_SRC := bench/kernel_bench.cpp
KERNEL_BENCH_BIN := $(BIN_DIR)/release/kernel_bench
KERNEL_BENCH_OBJS := $(BIN_DIR)/obj-release/engine/core/tensorView.o \
                     $(BIN_DIR)/obj-release/engine/ops/mathOps.o \
                     $(BIN_DIR)/obj-release/engine/core/taskQueue.o
ifeq ($(UNAME_S),Darwin)
    KERNEL_BENCH_LIBS := -framework Accelerate
    KERNEL_BENCH_DEFS := -DHAVE_ACCELERATE -DACCELERATE_NEW_LAPACK
endif

# Build the release engine library + kernel microbenchmark binary
bench_build:
	@$(MAKE) --no-print-directory BUILD=release all
	@$(MAKE) --no-print-directory BUILD=release $(KERNEL_BENCH_BIN)

$(KERNEL_BENCH_BIN): $(KERNEL_BENCH_SRC) $(KERNEL_BENCH_OBJS)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -DNDEBUG $(KERNEL_BENCH_DEFS) $(KERNEL_BENCH_SRC) $(KERNEL_BENCH_OBJS) $(KERNEL_BENCH_LIBS) -o $@

# Full benchmark suite: kernels + bandwidth + engine E2E + PyTorch baseline (needs real weights)
bench: bench_build
	@$(PYTHON) bench/run_bench.py

# Quick run: fewer prompt lengths / repeats, skips PyTorch baseline
bench_quick: bench_build
	@$(PYTHON) bench/run_bench.py --quick

# Generate synthetic weights on demand
synthetic_weights:
	@mkdir -p $(BIN_DIR)/synthetic
	@$(PYTHON) scripts/generate_synthetic_weights.py --out-dir $(BIN_DIR)/synthetic

# Generate full real weights from HuggingFace
weights:
	@mkdir -p $(BIN_DIR)
	@echo "Exporting HuggingFace weights to $(BIN_DIR)..."
	@$(PYTHON) scripts/convert_weights.py --out-dir $(BIN_DIR)

# Clean build artifacts (keeps model weights intact)
clean:
	rm -rf $(BIN_DIR)/obj $(BIN_DIR)/obj-release $(BIN_DIR)/release $(BIN_DIR)/libengine.$(LIB_EXT) $(TEST_UNIT_BIN) tests/unit/test_mathOps

.PHONY: all test test_synthetic test_full test_e2e check bench bench_build bench_quick synthetic_weights weights clean

-include $(OBJS:.o=.d)
