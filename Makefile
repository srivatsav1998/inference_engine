CXX      := g++
CXXFLAGS := -std=c++20 -Wall -Wextra -g -O3 -MMD -MP -fPIC -I/opt/homebrew/Cellar/nlohmann-json/3.12.0/include

# Expose headers from all modules
CXXFLAGS += -Icore -Igraph -Imodels -Iengine

BIN_DIR  := bin
OBJ_DIR  := $(BIN_DIR)/obj

TARGET   := $(BIN_DIR)/libengine.dylib

# Directories to search for source files. We keep engine here for now while you transition.
MODULES  := core graph models engine
SRCS     := $(foreach mod, $(MODULES), $(wildcard $(mod)/*.cpp))
OBJS     := $(patsubst %.cpp, $(OBJ_DIR)/%.o, $(SRCS))

# Default target
all: $(TARGET)

# Link the final binary
$(TARGET): $(OBJS)
	@mkdir -p $(BIN_DIR)
	$(CXX) $(CXXFLAGS) -shared $^ -o $@

# Compile object files
$(OBJ_DIR)/%.o: %.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean build artifacts but keep weights
clean:
	rm -rf $(OBJ_DIR) $(TARGET)

# Optional force flag, use: make generate_weights FORCE_GENERATE=1
FORCE_GENERATE ?= 0

# Generate model weights and copy to bin directory
generate_weights:
	@mkdir -p $(BIN_DIR)
	@if [ "$(FORCE_GENERATE)" = "1" ] || [ ! -f "model_convertor/model_weights.bin" ] || [ ! -f "model_convertor/model_offsets.json" ]; then \
		echo "Generating model weights..."; \
		cd model_convertor && . bin/activate && python3 model_convertor.py; \
	else \
		echo "Found cached weights in model_convertor/, skipping Python script..."; \
	fi
	@cp model_convertor/model_weights.bin $(BIN_DIR)/ 2>/dev/null || true
	@cp model_convertor/model_offsets.json $(BIN_DIR)/ 2>/dev/null || true
	@echo "Weights and config successfully copied to $(BIN_DIR)/"

.PHONY: all clean generate_weights

-include $(OBJS:.o=.d)
