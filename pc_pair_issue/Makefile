NVCC ?= /usr/local/cuda-11.7/bin/nvcc
ARCH ?= sm_70
TARGET := pc_pair_issue
TARGET_V2 := pc_pair_issue_v2
TARGET_V3 := pc_pair_issue_v3
TARGET_V4 := pc_pair_issue_v4
TARGET_V5 := pc_pair_issue_v5
TARGET_V6 := pc_pair_issue_v6
TARGET_V7 := pc_pair_issue_v7
TARGET_V8 := pc_pair_issue_v8
TARGET_V9 := pc_pair_issue_v9
TARGET_V10 := pc_pair_issue_v10

NVCCFLAGS ?= -O3 -std=c++14 -arch=$(ARCH) -cudart shared

.PHONY: all run-baseline run-split run-v2-baseline run-v2-split run-v3-baseline run-v3-split run-v4-baseline run-v4-split run-v5-baseline run-v5-split run-v6-baseline run-v6-split-h200 run-v6-split-dual run-v7-baseline run-v7-split-h200 run-v7-stream-dual run-v8-baseline run-v8-split-h200 run-v8-stream-dual run-v9-baseline run-v9-split-h200 run-v9-stream-dual run-v10-baseline run-v10-split-h200 run-v10-stream-dual run-v10-queue-dual ptx ptx-v2 ptx-v3 ptx-v4 ptx-v5 ptx-v6 ptx-v7 ptx-v8 ptx-v9 ptx-v10 clean

all: $(TARGET) $(TARGET_V2) $(TARGET_V3) $(TARGET_V4) $(TARGET_V5) $(TARGET_V6) $(TARGET_V7) $(TARGET_V8) $(TARGET_V9) $(TARGET_V10)

$(TARGET): pc_pair_issue.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

$(TARGET_V2): pc_pair_issue_v2.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

$(TARGET_V3): pc_pair_issue_v3.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

$(TARGET_V4): pc_pair_issue_v4.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

$(TARGET_V5): pc_pair_issue_v5.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

$(TARGET_V6): pc_pair_issue_v6.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

$(TARGET_V7): pc_pair_issue_v7.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

$(TARGET_V8): pc_pair_issue_v8.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

$(TARGET_V9): pc_pair_issue_v9.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

$(TARGET_V10): pc_pair_issue_v10.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

run-baseline: $(TARGET)
	./$(TARGET) --mode baseline --n 67584 --iters 8 --warmup 0 --repeats 1

run-split: $(TARGET)
	./$(TARGET) --mode split --n 67584 --iters 8 --warmup 0 --repeats 1

run-v2-baseline: $(TARGET_V2)
	./$(TARGET_V2) --mode baseline --n 67584 --iters 8 --warmup 0 --repeats 1

run-v2-split: $(TARGET_V2)
	./$(TARGET_V2) --mode split --n 67584 --iters 8 --warmup 0 --repeats 1

run-v3-baseline: $(TARGET_V3)
	./$(TARGET_V3) --mode baseline --n 67584 --iters 8 --warmup 0 --repeats 1

run-v3-split: $(TARGET_V3)
	./$(TARGET_V3) --mode split --n 67584 --iters 8 --warmup 0 --repeats 1

run-v4-baseline: $(TARGET_V4)
	./$(TARGET_V4) --mode baseline --n 67584 --iters 8 --warmup 0 --repeats 1

run-v4-split: $(TARGET_V4)
	./$(TARGET_V4) --mode split --n 67584 --iters 8 --warmup 0 --repeats 1

run-v5-baseline: $(TARGET_V5)
	./$(TARGET_V5) --mode baseline --n 67584 --iters 8 --warmup 0 --repeats 1

run-v5-split: $(TARGET_V5)
	./$(TARGET_V5) --mode split --n 67584 --iters 8 --warmup 0 --repeats 1

run-v6-baseline: $(TARGET_V6)
	./$(TARGET_V6) --mode baseline --n 67584 --iters 8 --warmup 0 --repeats 1

run-v6-split-h200: $(TARGET_V6)
	./$(TARGET_V6) --mode split_h200 --n 67584 --iters 8 --warmup 0 --repeats 1

run-v6-split-dual: $(TARGET_V6)
	./$(TARGET_V6) --mode split_dual --n 67584 --iters 8 --warmup 0 --repeats 1

run-v7-baseline: $(TARGET_V7)
	./$(TARGET_V7) --mode baseline --n 4096 --iters 8 --tiles 64 --warmup 0 --repeats 1

run-v7-split-h200: $(TARGET_V7)
	./$(TARGET_V7) --mode split_h200 --n 4096 --iters 8 --tiles 64 --warmup 0 --repeats 1

run-v7-stream-dual: $(TARGET_V7)
	./$(TARGET_V7) --mode stream_dual --n 4096 --iters 8 --tiles 64 --warmup 0 --repeats 1

run-v8-baseline: $(TARGET_V8)
	./$(TARGET_V8) --mode baseline --n 4096 --iters 8 --tiles 32 --warmup 0 --repeats 1

run-v8-split-h200: $(TARGET_V8)
	./$(TARGET_V8) --mode split_h200 --n 4096 --iters 8 --tiles 32 --warmup 0 --repeats 1

run-v8-stream-dual: $(TARGET_V8)
	./$(TARGET_V8) --mode stream_dual --n 4096 --iters 8 --tiles 32 --warmup 0 --repeats 1

run-v9-baseline: $(TARGET_V9)
	./$(TARGET_V9) --mode baseline --n 4096 --iters 8 --tiles 32 --warmup 0 --repeats 1

run-v9-split-h200: $(TARGET_V9)
	./$(TARGET_V9) --mode split_h200 --n 4096 --iters 8 --tiles 32 --warmup 0 --repeats 1

run-v9-stream-dual: $(TARGET_V9)
	./$(TARGET_V9) --mode stream_dual --n 4096 --iters 8 --tiles 32 --warmup 0 --repeats 1

run-v10-baseline: $(TARGET_V10)
	./$(TARGET_V10) --mode baseline --n 4096 --iters 8 --tiles 32 --warmup 0 --repeats 1

run-v10-split-h200: $(TARGET_V10)
	./$(TARGET_V10) --mode split_h200 --n 4096 --iters 8 --tiles 32 --warmup 0 --repeats 1

run-v10-stream-dual: $(TARGET_V10)
	./$(TARGET_V10) --mode stream_dual --n 4096 --iters 8 --tiles 32 --warmup 0 --repeats 1

run-v10-queue-dual: $(TARGET_V10)
	./$(TARGET_V10) --mode queue_dual --n 4096 --iters 8 --tiles 32 --warmup 0 --repeats 1

ptx: pc_pair_issue.cu
	$(NVCC) $(NVCCFLAGS) -ptx -o pc_pair_issue.ptx $<

ptx-v2: pc_pair_issue_v2.cu
	$(NVCC) $(NVCCFLAGS) -ptx -o pc_pair_issue_v2.ptx $<

ptx-v3: pc_pair_issue_v3.cu
	$(NVCC) $(NVCCFLAGS) -ptx -o pc_pair_issue_v3.ptx $<

ptx-v4: pc_pair_issue_v4.cu
	$(NVCC) $(NVCCFLAGS) -ptx -o pc_pair_issue_v4.ptx $<

ptx-v5: pc_pair_issue_v5.cu
	$(NVCC) $(NVCCFLAGS) -ptx -o pc_pair_issue_v5.ptx $<

ptx-v6: pc_pair_issue_v6.cu
	$(NVCC) $(NVCCFLAGS) -ptx -o pc_pair_issue_v6.ptx $<

ptx-v7: pc_pair_issue_v7.cu
	$(NVCC) $(NVCCFLAGS) -ptx -o pc_pair_issue_v7.ptx $<

ptx-v8: pc_pair_issue_v8.cu
	$(NVCC) $(NVCCFLAGS) -ptx -o pc_pair_issue_v8.ptx $<

ptx-v9: pc_pair_issue_v9.cu
	$(NVCC) $(NVCCFLAGS) -ptx -o pc_pair_issue_v9.ptx $<

ptx-v10: pc_pair_issue_v10.cu
	$(NVCC) $(NVCCFLAGS) -ptx -o pc_pair_issue_v10.ptx $<

clean:
	rm -f $(TARGET) $(TARGET_V2) $(TARGET_V3) $(TARGET_V4) $(TARGET_V5) $(TARGET_V6) $(TARGET_V7) $(TARGET_V8) $(TARGET_V9) $(TARGET_V10) pc_pair_issue.ptx pc_pair_issue_v2.ptx pc_pair_issue_v3.ptx pc_pair_issue_v4.ptx pc_pair_issue_v5.ptx pc_pair_issue_v6.ptx pc_pair_issue_v7.ptx pc_pair_issue_v8.ptx pc_pair_issue_v9.ptx pc_pair_issue_v10.ptx
