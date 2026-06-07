NVCC ?= nvcc
ARCH ?= sm_70
CXXSTD ?= c++17

TARGET := stream_fma_v6_v2_m4_c128_cta512
SRC := stream_fma_v6_v2_m4_c128_cta512.cu

NVCCFLAGS ?= -O3 -std=$(CXXSTD) -arch=$(ARCH) -lineinfo -cudart shared
RUN_ARGS := --n 524288 --iters 128 --memory-iters 4 --warmup 0 --repeats 1

.PHONY: all run clean

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCCFLAGS) -o $@ $<

run: $(TARGET)
	./$(TARGET) $(RUN_ARGS)

clean:
	rm -f $(TARGET)
