CXX      := g++
CXXFLAGS := -std=c++20 -Wall -Wextra -g -O3 -MMD -MP -fPIC -I/opt/homebrew/Cellar/nlohmann-json/3.12.0/include

BIN_DIR  := bin
OBJ_DIR  := $(BIN_DIR)/obj

TARGET   := $(BIN_DIR)/libengine.dylib
SRCS     := $(wildcard *.cpp)
OBJS     := $(patsubst %.cpp, $(OBJ_DIR)/%.o, $(SRCS))
ENGINE_OBJS := $(filter-out $(OBJ_DIR)/main.o, $(OBJS))

# Default target
all: $(TARGET)

# Link the final binary
$(TARGET): $(OBJS)
	@mkdir -p $(BIN_DIR)
	$(CXX) $(CXXFLAGS) -shared $^ -o $@

# Compile object files
$(OBJ_DIR)/%.o: %.cpp
	@mkdir -p $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean all generated files
clean:
	rm -rf $(BIN_DIR)

# Test target
test: $(ENGINE_OBJS)
	@mkdir -p $(BIN_DIR)
	$(CXX) $(CXXFLAGS) tests/test_mathOps.cpp $(ENGINE_OBJS) -o $(BIN_DIR)/test_mathOps
	./$(BIN_DIR)/test_mathOps

# Prevent make from confusing these rules with file names
.PHONY: all clean test

-include $(OBJS:.o=.d)