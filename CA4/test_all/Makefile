CC = gcc
CFLAGS = -Wall -Wextra -std=c11 -O2
TARGET = test_model
SOURCES = main.c

# Math library might be needed
LIBS = -lm

all: $(TARGET)

$(TARGET): $(SOURCES)
	$(CC) $(CFLAGS) -o $(TARGET) $(SOURCES) $(LIBS)

clean:
	rm -f $(TARGET)

run: $(TARGET)
	./$(TARGET)

.PHONY: all clean run
