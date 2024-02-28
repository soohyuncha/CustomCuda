# CUDA Compiler
NVCC = nvcc

# Source files
SOURCES = linear_kernel.cu

# Object files
OBJECTS = $(SOURCES:.cu=.o)

# Target
TARGET = linear

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(NVCC) -o $@ $(OBJECTS)
	rm -f $(OBJECTS)

%.o: %.cu
	$(NVCC) -c $< -o $@

clean:
	rm -f *.o
	rm -f $(TARGET)
