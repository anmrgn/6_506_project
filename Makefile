CXX = xcrun /opt/opencilk/bin/clang++
CPPFLAGS = -Wall -std=c++17 -fopencilk -O3
SRC_DIR = src
BIN_DIR = bin

all: $(BIN_DIR)/.dirstamp $(BIN_DIR)/main

$(BIN_DIR)/main: $(SRC_DIR)/distribution.hpp $(SRC_DIR)/lloyd.hpp $(SRC_DIR)/main.cpp
	$(CXX) $(CPPFLAGS) $(SRC_DIR)/main.cpp -o $@

.PHONY: clean
clean:
	rm -rf $(BIN_DIR)

$(BIN_DIR)/.dirstamp:
	mkdir -p $(BIN_DIR)
	touch $(BIN_DIR)/.dirstamp