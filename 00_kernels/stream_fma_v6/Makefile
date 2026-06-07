NVCC ?= /usr/local/cuda-13.2/bin/nvcc
CUOBJDUMP ?= /usr/local/cuda-13.2/bin/cuobjdump
ARCH ?= sm_120
TARGET := stream_fma_v6
INSTR_DIR := 00_instructions

NVCCFLAGS ?= -O3 -std=c++17 -arch=$(ARCH) -lineinfo

.PHONY: all run instructions clean clean-instructions

all: $(TARGET)

$(TARGET): stream_fma_v6.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

run: $(TARGET)
	./$(TARGET) --n 1032192 --iters 16 --memory-iters 4 --warmup 2 --repeats 5

$(INSTR_DIR):
	mkdir -p $(INSTR_DIR)

instructions: $(TARGET) $(INSTR_DIR)
	$(NVCC) $(NVCCFLAGS) -ptx -o $(INSTR_DIR)/stream_fma_v6.ptx stream_fma_v6.cu
	$(CUOBJDUMP) --dump-sass $(TARGET) > $(INSTR_DIR)/stream_fma_v6.sass
	awk 'BEGIN{emit=0} /^\.visible \.entry stream_fma_v6_m4_kernel\(/ {emit=1} emit {print} emit && /^}/ {emit=0}' $(INSTR_DIR)/stream_fma_v6.ptx > $(INSTR_DIR)/stream_fma_v6_m4_kernel.ptx
	awk 'BEGIN{emit=0} /Function : stream_fma_v6_m4_kernel/ {emit=1} emit && /Function :/ && $$0 !~ /stream_fma_v6_m4_kernel/ {exit} emit {print}' $(INSTR_DIR)/stream_fma_v6.sass > $(INSTR_DIR)/stream_fma_v6_m4_kernel.sass
	awk '/^[[:space:]]*\/\*[0-9a-fA-F]+/ {line=$$0; sub(/[[:space:]]*\/\* 0x[0-9a-fA-F]+ \*\/[[:space:]]*$$/, "", line); print line}' $(INSTR_DIR)/stream_fma_v6_m4_kernel.sass > $(INSTR_DIR)/stream_fma_v6_m4_kernel.instructions.sass

clean:
	rm -f $(TARGET)

clean-instructions:
	rm -f $(INSTR_DIR)/stream_fma_v6.ptx \
	      $(INSTR_DIR)/stream_fma_v6.sass \
	      $(INSTR_DIR)/stream_fma_v6_m4_kernel.ptx \
	      $(INSTR_DIR)/stream_fma_v6_m4_kernel.sass \
	      $(INSTR_DIR)/stream_fma_v6_m4_kernel.instructions.sass
