CXX      := g++
CXXFLAGS := -std=c++20 -Wall -Wextra -g -O3 -MMD -MP -fPIC

# Portable nlohmann-json header discovery
BREW_JSON_INC := $(shell brew --prefix nlohmann-json 2>/dev/null)/include
ifneq ($(wildcard $(BREW_JSON_INC)),)
    CXXFLAGS += -I$(BREW_JSON_INC)
endif

# Expose engine headers
CXXFLAGS += -Iengine

BIN_DIR  := bin
OBJ_DIR  := $(BIN_DIR)/obj

# Shared library extension detection
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
    TARGET := $(BIN_DIR)/libengine.dylib
else
    TARGET := $(BIN_DIR)/libengine.so
endif

SRCS     := $(wildcard engine/*.cpp)
OBJS     := $(patsubst engine/%.cpp, $(OBJ_DIR)/engine/%.o, $(SRCS))

# Test source & binary
TEST_UNIT_SRC := tests/unit/test_mathOps.cpp
TEST_UNIT_BIN := $(BIN_DIR)/test_mathOps
TEST_OBJS     := $(OBJ_DIR)/engine/arenaAllocator.o \
                 $(OBJ_DIR)/engine/tensorView.o \
                 $(OBJ_DIR)/engine/mathOps.o \
                 $(OBJ_DIR)/engine/taskQueue.o

# Python executable (prefer .venv if available)
PYTHON := $(shell if [ -f .venv/bin/python3 ]; then echo .venv/bin/python3; else echo python3; fi)

# Default target: build the engine library
all: $(TARGET)

$(TARGET): $(OBJS)
	@mkdir -p $(BIN_DIR)
	$(CXX) $(CXXFLAGS) -shared $^ -o $@

$(OBJ_DIR)/engine/%.o: engine/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Build and run C++ unit tests
test: $(TARGET) $(TEST_OBJS)
	@mkdir -p $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $(TEST_UNIT_SRC) $(TEST_OBJS) -o $(TEST_UNIT_BIN)
	@echo "Running C++ Unit Tests..."
	@./$(TEST_UNIT_BIN)

# Run Python E2E deterministic greedy decoding test
test_e2e: $(TARGET)
	@echo "Running E2E Greedy Decoding Regression Test..."
	@$(PYTHON) tests/e2e/test_greedy_decoding.py

# Run all tests
check: test test_e2e

# Generate weights using Python script
weights:
	@mkdir -p $(BIN_DIR)
	@echo "Exporting weights to $(BIN_DIR)..."
	@$(PYTHON) scripts/convert_weights.py --out-dir $(BIN_DIR)

# Clean build artifacts (keeps model weights intact)
clean:
	rm -rf $(OBJ_DIR) $(TARGET) $(TEST_UNIT_BIN) tests/unit/test_mathOps

.PHONY: all test test_e2e check weights clean

-include $(OBJS:.o=.d)
