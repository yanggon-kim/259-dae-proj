NVCC ?= nvcc
ARCH ?= sm_70
CXXSTD ?= c++17

TARGET := stream_fma_v6_v2_m4_c128_cta512
SRC := stream_fma_v6_v2_m4_c128_cta512.cu

NVCCFLAGS ?= -O3 -std=$(CXXSTD) -arch=$(ARCH) -cudart shared
RUN_ARGS := --n 524288 --iters 128 --memory-iters 4 --warmup 0 --repeats 1
V2_RUN_ARGS := $(RUN_ARGS) --variant 2
V3_RUN_ARGS := $(RUN_ARGS) --variant 3

.PHONY: all run run-v2 run-v3 ptx clean

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCCFLAGS) -o $@ $<

run: run-v2

run-v2: $(TARGET)
	./$(TARGET) $(V2_RUN_ARGS)

run-v3: $(TARGET)
	./$(TARGET) $(V3_RUN_ARGS)

ptx: $(SRC)
	$(NVCC) $(NVCCFLAGS) -ptx -o $(TARGET).ptx $<

clean:
	rm -f $(TARGET) $(TARGET).ptx
