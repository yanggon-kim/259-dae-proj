# Variant 3 Design: Faithful Register-File Queue

## Status

This is a revised design-review draft. It replaces the earlier kernel-only RFQ
emulation plan. No Variant 3 implementation should start until the user
approves this simulator-side RFQ design.

## Goal

Variant 3 should implement the WASP-style register-file queue (RFQ) in
GPGPU-Sim and use it with a warp-specialized streaming-FMA workload. It must
measure the benefit and cost of a real RFQ against Variant 2. It must not add
producer/consumer co-issue; cross-warp co-issue remains Variant 4.

## Design Intention and Hypothesis

The RFQ is a named FIFO channel between a producer warp stage and a consumer
warp stage. Producer RFQ loads reserve queue entries, normal global memory
responses mark those entries ready, and consumer RFQ reads consume ready head
entries in FIFO order.

This differs from the earlier emulation plan:

```text
Old plan: CUDA kernel stores producer-loaded values through shared memory.
New plan: simulator models LDG_TO_RFQ and MOV_FROM_RFQ with queue tokens.
```

Hypothesis: RFQ should improve producer runahead and memory/compute overlap
without shared-memory handoff traffic. Moderate queue depths should improve
eligible-warp occupancy and DRAM utilization. Very deep queues may reduce
resident CTA count because RFQ entries consume register-file capacity.

## WASP RFQ Semantics to Preserve

- RFQ lives in register-file capacity, not shared memory.
- Each queue has `head`, `tail`, `min`, and `max` metadata.
- Producer RFQ load stalls when the target queue is full.
- Consumer RFQ read stalls when the source queue is empty or the FIFO head is
  reserved but not ready.
- Memory responses fill reserved RFQ entries instead of producer registers.
- Strict FIFO order is used by default.
- Queue metadata and queue register footprint are reported as stats.

## V3 Stream-FMA Mapping

The V3 workload keeps the V2 logical split: 256 threads/CTA, producer warps
0-3, consumer warps 4-7.

Map these warps to WASP-style pipeline stages:

```text
producer warp p = warp_id 0..3   -> original_warp_id p, stage_id 0
consumer warp p = warp_id 4..7   -> original_warp_id p-4, stage_id 1
```

Use one named RFQ per original warp pipeline. Each RFQ entry carries the three
stream-FMA operands for one warp-wide element group:

```text
q0: {a, b, c} operand tuple, stage 0 -> stage 1
payload_words_per_lane = 3
element_bits = 96
```

Producer stage:

```text
LDG3_TO_RFQ q0, [a + i], [b + i], [c + i]
```

Consumer stage:

```text
MOV3_FROM_RFQ r_a, r_b, r_c, q0
stream_fma_value(r_a, r_b, r_c)
STG [out + i], result
```

`LDG3_TO_RFQ` and `MOV3_FROM_RFQ` are V3 pseudo-instructions used by the
generated PTX path. Internally, one RFQ entry is reserved for the tuple. The
entry becomes ready only after all three global-load payloads are available.

## Step-by-Step Implementation Plan

### Step 1: Add RFQ Config Knobs

Files:

```text
gpgpu-sim_distribution/src/gpgpu-sim/gpu-sim.cc
gpgpu-sim_distribution/src/gpgpu-sim/gpu-sim.h
gpgpu-sim_distribution/src/gpgpu-sim/shader.h
```

Add knobs:

```text
-gpgpu_wasp_enable 0|1
-gpgpu_wasp_rfq_enable 0|1
-gpgpu_wasp_rfq_entries 32
-gpgpu_wasp_max_pipeline_stages 16
-gpgpu_wasp_rfq_count_register_pressure 1
-gpgpu_wasp_rfq_model_data 1
-gpgpu_wasp_rfq_strict_fifo 1
-gpgpu_wasp_rfq_assert_mask_match 1
-gpgpu_wasp_metadata_file <path>
```

Intention: baseline kernels remain unchanged when WASP/RFQ is disabled.

### Step 2: Add Kernel and Warp Stage Metadata

Files:

```text
gpgpu-sim_distribution/src/gpgpu-sim/wasp-rfq.h
gpgpu-sim_distribution/src/gpgpu-sim/wasp-rfq.cc
gpgpu-sim_distribution/src/gpgpu-sim/shader.h
gpgpu-sim_distribution/src/gpgpu-sim/shader.cc
```

Add:

```cpp
struct wasp_warp_stage_info {
    bool wasp_enabled;
    unsigned cta_id;
    unsigned original_warp_id;
    unsigned stage_id;
    unsigned num_pipeline_stages;
};

struct wasp_queue_spec {
    unsigned original_warp_id;
    unsigned src_stage_id;
    unsigned dst_stage_id;
    unsigned queue_id;
    unsigned entries;
    unsigned element_bits;
    unsigned payload_words_per_lane;
};
```

Use a small sidecar metadata file for V3 instead of building a WASP compiler.
For `stream_fma_v6_variant3`, metadata will specify two stages and one
three-word queue per original warp. The simulator will attach stage identity to
resident warps at CTA allocation.

Intention: make RFQ behavior explicit and avoid fragile inference from warp IDs
inside the scheduler.

### Step 3: Implement RFQ Table and Queue Object

Files:

```text
gpgpu-sim_distribution/src/gpgpu-sim/wasp-rfq.h
gpgpu-sim_distribution/src/gpgpu-sim/wasp-rfq.cc
```

Add:

```cpp
enum class rfq_entry_state { FREE, RESERVED, READY };

struct rfq_key {
    unsigned sm_id;
    unsigned cta_hw_id;
    unsigned original_warp_id;
    unsigned src_stage_id;
    unsigned dst_stage_id;
    unsigned queue_id;
};

struct rfq_token {
    rfq_key key;
    unsigned slot_index;
    unsigned sequence;
};
```

The `wasp_rfq` object will implement:

```text
full()
head_ready()
reserve_tail()
mark_ready()
consume_head()
occupancy()
ready_count()
pending_count()
```

Intention: model RFQ as a real per-SM, per-CTA FIFO with reserved, pending, and
ready states.

### Step 4: Add RFQ Instruction Representation

Files:

```text
gpgpu-sim_distribution/src/abstract_hardware_model.h
gpgpu-sim_distribution/src/abstract_hardware_model.cc
gpgpu-sim_distribution/src/cuda-sim/opcodes.def
gpgpu-sim_distribution/src/cuda-sim/opcodes.h
gpgpu-sim_distribution/src/cuda-sim/ptx.y
gpgpu-sim_distribution/src/cuda-sim/ptx.l
gpgpu-sim_distribution/src/cuda-sim/cuda-sim.cc
gpgpu-sim_distribution/src/cuda-sim/instructions.cc
```

Add simulator-visible pseudo instructions:

```text
ld.global.rfq.f32 qN, [addr];
mov.rfq.f32 %fX, qN;
ld.global.rfq.v3.f32 qN, [addr_a], [addr_b], [addr_c];
mov.rfq.v3.f32 %fA, %fB, %fC, qN;
```

The parser will store RFQ operand metadata in `warp_inst_t` / `ptx_instruction`:

```text
is_ldg_to_rfq
is_mov_from_rfq
rfq_queue_id
rfq_src_stage_id
rfq_dst_stage_id
rfq_payload_words_per_lane
```

Intention: make producer RFQ loads and consumer RFQ reads first-class
instructions in timing simulation. The scalar forms support simple
microbenchmarks; the `v3` forms support the stream-FMA `{a,b,c}` tuple through
one RFQ per original warp.

### Step 5: Connect Producer RFQ Loads to Memory Requests

Files:

```text
gpgpu-sim_distribution/src/gpgpu-sim/shader.cc
gpgpu-sim_distribution/src/gpgpu-sim/mem_fetch.h
gpgpu-sim_distribution/src/gpgpu-sim/mem_fetch.cc
gpgpu-sim_distribution/src/gpgpu-sim/scoreboard.cc
```

At issue time for `LDG_TO_RFQ`:

1. Resolve the target `rfq_key`.
2. If the queue is full, do not issue and count `rfq_full_stalls`.
3. Reserve a tail slot.
4. Attach an `rfq_token` to the generated memory request.
5. Do not reserve a producer destination register in the normal scoreboard.

For `LDG3_TO_RFQ`, reserve one tail slot and attach related tokens for the
three generated memory requests. The RFQ entry tracks a three-bit ready mask and
becomes consumer-visible only when all tuple words are ready.

On memory response:

1. If no RFQ token, use the existing writeback path.
2. If an RFQ token exists, fill the matching payload word in that queue slot.
3. Mark a scalar RFQ slot `READY`, or mark a tuple slot `READY` once all three
   payload words are available.
4. Store lane values if functional RFQ data mode is enabled.
5. Update RFQ scoreboard status.

Intention: producer runahead is limited by RFQ capacity, not by a producer
destination register dependency.

### Step 6: Connect Consumer RFQ Reads

Files:

```text
gpgpu-sim_distribution/src/gpgpu-sim/shader.cc
gpgpu-sim_distribution/src/gpgpu-sim/scoreboard.cc
gpgpu-sim_distribution/src/cuda-sim/instructions.cc
```

At issue time for `MOV_FROM_RFQ`:

1. Resolve the source `rfq_key`.
2. If the FIFO head is not ready, do not issue.
3. Count `rfq_empty_stalls` or `rfq_head_not_ready_stalls`.
4. If ready, consume the head entry.
5. Write the value into the consumer destination register.
6. Reserve/release the consumer destination register normally.

Intention: consumers are synchronized by queue readiness instead of CTA-wide
barriers or shared-memory handoff.

For `MOV3_FROM_RFQ`, one consume operation reads the FIFO head and writes three
consumer destination registers. The entry is freed only after all three tuple
words are copied to the consumer instruction destination operands.

### Step 7: Add RFQ Register Pressure and Occupancy Accounting

Files:

```text
gpgpu-sim_distribution/src/gpgpu-sim/shader.cc
gpgpu-sim_distribution/src/gpgpu-sim/shader.h
```

Register cost:

```text
rfq_regs_per_cta =
  original_warps_per_cta *
  rfq_entries *
  payload_words_per_lane *
  warp_size
```

For V3 stream-FMA:

```text
original_warps_per_cta = 4
queues_per_original_warp = 1
payload_words_per_lane = 3
element_bits = 96
rfq_regs_per_cta = 4 * rfq_entries * 3 * 32
```

Depth 32 therefore costs `12288` RFQ register slots per CTA. Add this cost to
CTA resource checks when `-gpgpu_wasp_rfq_count_register_pressure 1`.

The occupancy calculation must use the RFQ-adjusted register footprint, not the
original kernel register footprint alone:

```text
normal_regs_per_cta =
  regs_stage0 * producer_threads +
  regs_stage1 * consumer_threads

total_regs_per_cta_after_rfq =
  normal_regs_per_cta + rfq_regs_per_cta

ctas_per_sm_limit_by_regs_after_rfq =
  floor(gpgpu_shader_registers / total_regs_per_cta_after_rfq)

resident_ctas_per_sm_after_rfq =
  min(thread_limit, warp_limit, smem_limit, cta_limit,
      ctas_per_sm_limit_by_regs_after_rfq)
```

For the H200 config, `gpgpu_shader_registers = 65536`. With 256 threads/CTA,
the thread/warp limit is 8 CTAs/SM, but RFQ register pressure may reduce that.
For example, if the RFQ-adjusted register footprint permits only two CTAs at
depth 8 and only one CTA at depth 32, the simulator must report exactly that as
the after-RFQ occupancy limit and schedule no more CTAs than that limit allows.

The queue-depth sweep must report these fields for every depth:

```text
normal_regs_per_cta
rfq_regs_per_cta
total_regs_per_cta_after_rfq
ctas_per_sm_limit_by_regs_before_rfq
ctas_per_sm_limit_by_regs_after_rfq
resident_ctas_per_sm_before_rfq
resident_ctas_per_sm_after_rfq
active_warp_occupancy_before_rfq
active_warp_occupancy_after_rfq
```

Intention: queue depth must affect occupancy just as the WASP paper describes.

### Step 8: Add RFQ Stats and Deadlock Diagnostics

Files:

```text
gpgpu-sim_distribution/src/gpgpu-sim/shader.h
gpgpu-sim_distribution/src/gpgpu-sim/shader.cc
gpgpu-sim_distribution/src/gpgpu-sim/wasp-rfq.cc
```

Stats:

```text
rfq_reservations
rfq_memory_fills
rfq_consumes
rfq_full_stalls
rfq_empty_stalls
rfq_head_not_ready_stalls
rfq_max_occupancy
rfq_total_occupancy_cycles
rfq_ready_occupancy_cycles
rfq_pending_occupancy_cycles
rfq_register_slots_allocated
rfq_metadata_bits
ctas_per_sm_limit_before_rfq
ctas_per_sm_limit_after_rfq
wasp_stage_issue_count[stage_id]
```

Add a debug dump:

```text
SM CTA Worig Ssrc->Sdst Qid cap head tail occ ready pending full empty fills consumes
```

Intention: V3 results must show whether the RFQ is useful, full, empty, or
occupancy-limited.

### Step 9: Build V3 PTX Test Path

Files:

```text
01_github/259-dae-proj/stream_fma_v6/stream_fma_v6_variant3.cu
01_github/259-dae-proj/stream_fma_v6/Makefile
01_github/259-dae-proj/stream_fma_v6/01_analysis/04_variant3_rfq/
```

Because NVCC will not assemble custom RFQ PTX mnemonics directly, use this
flow:

1. Build a normal CUDA host harness and placeholder kernel launch.
2. Generate PTX for the placeholder kernel.
3. Generate an RFQ-patched PTX file containing `ld.global.rfq.f32` and
   `mov.rfq.f32` for scalar microbenchmarks, and `ld.global.rfq.v3.f32` /
   `mov.rfq.v3.f32` for stream-FMA tuple traffic.
4. Run GPGPU-Sim with PTX override so the host binary launches the kernel while
   the simulator parses the RFQ PTX.

Intention: avoid writing a full compiler while still evaluating real
simulator-side RFQ instructions. The user approved this handcoded generated-PTX
path for Variant 3.

### Step 10: Validate With RFQ Microbenchmarks First

Before the full streaming-FMA run, create and run:

```text
rfq_stream_copy: producer LDG_TO_RFQ, consumer MOV_FROM_RFQ + STG
rfq_full_stress: producer faster than consumer
rfq_empty_stress: consumer faster than producer or long memory latency
```

The user approved these RFQ microbenchmarks as required preconditions before
stream-FMA measurement.

Validation expectations:

```text
producer stalls on full queue
consumer stalls on empty or head-not-ready queue
memory responses mark entries ready
strict FIFO holds under out-of-order memory returns
baseline non-WASP kernels are unchanged when RFQ is disabled
```

Intention: debug RFQ correctness before running a long full-size benchmark.

### Step 11: Run Queue-Depth Sweep

Use H200 config:

```text
02_h200_config/config_h200_132sm_mshr512/
```

Queue depths:

```text
4, 8, 16, 32, 64, 128
```

Commands will be recorded after implementation, but every run must include:

```text
n = 1032192
compute_iters = 16
warmup = 0
repeats = 1
RFQ enabled
cross-warp co-issue disabled
```

Primary comparison:

```text
Variant 2 current smem handoff vs Variant 3 RFQ depth sweep
```

Intention: isolate the register-file queue effect before scheduler co-issue.

## Expected Metric Impact

- Shared-memory instruction count should drop versus V2 because RFQ replaces
  the smem handoff for `a/b/c`.
- Barriers should drop or disappear from the producer/consumer data path.
- Producer runahead should increase until queues become full.
- DRAM `CoL_Bus_Util` and effective bandwidth may increase if runahead exposes
  more memory-level parallelism.
- Deep RFQs may reduce `ctas_per_sm_limit_after_rfq` and active occupancy.

DRAM bandwidth remains:

```text
effective DRAM BW = 4800 GB/s * CoL_Bus_Util
```

## Output Artifacts After Approval

Planned outputs:

```text
00_doc/02_variant_data/variant_results/v3/metrics_v3.csv
00_doc/02_variant_data/variant_results/v3/results_v3.md
00_doc/02_variant_data/variant_results/v3/queue_depth_sweep_v3.md
00_doc/02_variant_data/comparisons/comparison_v2_v3.md
```

Raw logs, RFQ metadata files, generated RFQ PTX, and RFQ microbenchmark logs
will remain in:

```text
00_doc/02_variant_data/variant_results/v3/
```

## Correctness Risks and Validation Checks

- Risk: RFQ token is lost before memory response.
  - Check: `rfq_reservations == rfq_memory_fills` at kernel end.
- Risk: consumer and producer queue IDs do not match.
  - Check: dump RFQ key on full/empty deadlock.
- Risk: FIFO head is pending while later entries are ready.
  - Check: count `rfq_head_not_ready_stalls`.
- Risk: RFQ register pressure is undercounted.
  - Check: report before/after CTA-per-SM limits for every depth.
- Risk: functional data path is incomplete.
  - Check: start with `rfq_stream_copy`, then validate full stream-FMA output.
- Risk: baseline behavior changes.
  - Check: rerun V1/V2 sanity with `-gpgpu_wasp_enable 0`.

## Known Limitations

This design still does not implement the full WASP compiler, WASP-TMA,
out-of-order tagged RFQ consumption, multiple consumers per queue, or detailed
RFQ register-bank conflicts. It implements the core RFQ path needed for Variant
3: named queues, RFQ load destination, memory-response fill, RFQ read, queue
scoreboard, register-pressure accounting, and RFQ statistics.

## User Decisions Recorded

1. Use one RFQ per original warp for the `a/b/c` tuple, not three separate RFQs.
2. Include `rfq_stream_copy`, `rfq_full_stress`, and `rfq_empty_stress`
   microbenchmarks before stream-FMA.
3. Use handcoded generated RFQ PTX plus GPGPU-Sim PTX override instead of a full
   CUDA-to-WASP compiler transformation.

## Remaining Approval Gate

Implementation should start only after the user confirms this revised design is
approved for execution.
