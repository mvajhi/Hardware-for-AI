# Makefile for compiling any given model C file with the main tester.

# Compiler and flags
CC = gcc
CFLAGS = -Wall -Wextra -std=c11 -Og -lm -D_POSIX_C_SOURCE=199309L -pg

# The main C file for testing
MAIN_SRC = main_compare.c

# --- Verbosity Control ---
# To build silently, run: make QUIET=1 <target>
# Example: make QUIET=1 test_q8
# The '@' prefix suppresses the command from being echoed.
ifeq ($(QUIET),1)
    Q = @
else
    Q =
endif

# Default target
all:
	@echo "Usage: make runner MODEL_C_FILE=<your_model.c>"
	@echo "e.g., make runner MODEL_C_FILE=q6.c"

# Generic rule to build a runner for a specific model file
# Usage: make runner MODEL_C_FILE=<your_model_file.c>
runner: $(MAIN_SRC)
	@if [ -z "$(MODEL_C_FILE)" ]; then \
		echo "Error: MODEL_C_FILE is not set."; \
		echo "Usage: make runner MODEL_C_FILE=<your_model_file.c>"; \
		exit 1; \
	fi
	$(Q)echo "Compiling model: $(MODEL_C_FILE)..."
	$(Q)$(CC) $(CFLAGS) -o runner $(MAIN_SRC) -DMODEL_HEADER='"$(MODEL_C_FILE)"' -DMODEL_NAME='"$(MODEL_C_FILE)"'
	$(Q)echo "Created executable: 'runner'"

# Specific targets for convenience
test_q8:
	@# Recursively call make, passing the QUIET variable if it is set
	@make QUIET=$(QUIET) runner MODEL_C_FILE=q8_new3.c

test_q6:
	@# Recursively call make, passing the QUIET variable if it is set
	@make QUIET=$(QUIET) runner MODEL_C_FILE=q6.c

# Rule to clean up generated files
clean:
	$(Q)echo "Cleaning up..."
	$(Q)rm -f runner
