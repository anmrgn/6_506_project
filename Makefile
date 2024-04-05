CXX = /opt/homebrew/bin/g++-13
CPPFLAGS = -Wall
SRC_DIR = src
BIN_DIR = bin

all: $(BIN_DIR)/.dirstamp $(BIN_DIR)/main

$(BIN_DIR)/main: $(SRC_DIR)/main.cpp
	$(CXX) $(CPPFLAGS) $^ -o $@

.PHONY: clean
clean:
	rm -rf $(BIN_DIR)

$(BIN_DIR)/.dirstamp:
	mkdir -p $(BIN_DIR)
	touch $(BIN_DIR)/.dirstamp