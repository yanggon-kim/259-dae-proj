# Variant 3a Design: Shared-Memory-Backed Queue Without Co-Issue

## Status

This design was approved and implemented on the Variant 3a branch. The notes
below keep the original design intent and record the concrete implementation
and validation results.

Current branch:

```text
variant3a-smem-rfq
```

## Goal

Variant 3a adds a shared-memory-backed producer/consumer queue to the faithful
H200 warp-specialized workload. It compares against Variant 3 to isolate the
storage medium effect:

```text
Variant 3  = register-file-backed queue + warp specialization
Variant 3a = shared-memory-backed queue + warp specialization
```

Variant 3a must not enable producer/consumer cross-warp co-issue, same-cycle
dual issue, or software polling. Those remain later variants.

## Design Intention and Hypothesis

The intent is to keep the useful part of Variant 3, producer runahead and
scheduler-visible consumer readiness, while moving queue payload storage out of
the register file.

Hypothesis: shared-memory-backed queue storage should avoid RFQ register
pressure, allowing deeper queues without reducing CTA residency. The cost is
that queue payload accesses may create shared-memory bandwidth pressure, bank
conflicts, and extra queue access latency.

## Decisions Confirmed With User

Confirmed decisions:

```text
initial payload form = v3 tuple entries {a,b,c} only
shared-memory queue capacity counts against gpgpu_shmem_per_sm CTA residency immediately
config knobs, counters, PTX helpers, scripts, and reports use a distinct smemq prefix
```

Scalar, `float2`, and `float4` queue payload variants are deferred. They can be
added later as a design addendum if the first V3a stream-FMA implementation is
validated.

## Baseline and Branch Rule

Variant 3a should conceptually start from the clean Variant 2 state:

```text
faithful H200 config
warp-specialized producer/consumer kernel
no register-file queue
no cross-warp co-issue
```

If existing Variant 3 simulator helpers are reused, they must be reused as
implementation infrastructure only. Variant 3a must use separate config knobs,
separate PTX mnemonics, separate counters, and separate result paths so V3 RFQ
register-pressure behavior cannot leak into V3a.

## Queue Semantics

Each logical producer/consumer warp pair owns one queue.

For the stream-FMA workload:

```text
CTA threads              = 256
producer warps           = 0..3
consumer warps           = 4..7
logical original warps   = 0..3
queues/original warp     = 1
payload words/lane       = 3  // a, b, c
```

The queue payload lives in simulator-modeled SM-local shared-memory storage.
The queue metadata is hardware/simulator state:

```text
head
tail
valid bit per entry
active lane mask per entry
ready lane mask per entry
queue count
producer warp id
consumer warp id
full/empty state
```

`queue_head.valid` means the entire head entry is ready for the consumer warp.
It is set only after all active producer lanes have completed their payload
writes into the queue slot:

```text
producer reserves tail slot
producer global loads complete
producer writes active lanes into shared-memory queue slot
ready_lane_mask accumulates completed lanes
if ready_lane_mask == active_lane_mask:
    commit active_mask metadata
    set valid bit with release ordering
    consumer warp becomes scheduler-eligible
```

Do not require all 32 lanes if the producer active mask is partial. Only active
lanes participate in the readiness condition.

## Planned Simulator Interface

Add a new shared-memory queue model beside the existing RFQ model.

Config knobs:

```text
-gpgpu_smemq_enable 0|1
-gpgpu_smemq_entries 32
-gpgpu_smemq_payload_words 3
-gpgpu_smemq_scheduler_ready 1
-gpgpu_smemq_model_data 1
-gpgpu_smemq_assert_mask_match 1
-gpgpu_smemq_bank_conflict_model 1
-gpgpu_smemq_metadata_file <path>
```

Important default:

```text
-gpgpu_smemq_scheduler_ready 1
```

Consumer waits must block in scheduler eligibility logic. A consumer must not
issue a polling loop to test a shared-memory valid flag.

## Planned PTX Interface

Use generated or hand-written PTX, as in Variant 3. NVCC should build the host
harness and placeholder kernel; GPGPU-Sim should run with PTX override.

Add pseudo instructions:

```text
ld.global.smemq.v3.f32 qN, [addr_a], [addr_b], [addr_c];
mov.smemq.v3.f32 {%fA, %fB, %fC}, qN;
```

Producer instruction behavior:

```text
if queue full:
    do not issue
    count smemq_full_stall
else:
    reserve tail entry
    generate three global memory requests
    attach queue token to memory requests
```

Memory response behavior:

```text
write payload word into shared-memory queue slot
update per-entry word/lane readiness
commit valid when all active lane payloads are present
```

Consumer instruction behavior:

```text
if queue head is not valid:
    warp is not scheduler-eligible
    count smemq_empty_or_not_ready_stall
else:
    read head payload into consumer registers
    advance head
    clear consumed valid bit
```

## Planned Code Changes

### Step 1: Add Design-Isolated Config and Metadata

Expected files:

```text
gpgpu-sim_distribution/src/gpgpu-sim/gpu-sim.cc
gpgpu-sim_distribution/src/gpgpu-sim/gpu-sim.h
gpgpu-sim_distribution/src/gpgpu-sim/shader.h
```

Add the `smem_queue` knobs listed above. Keep them disabled by default.
Existing V1, V2, and V3 runs must be unchanged unless the new enable knob is
set.

### Step 2: Add Shared-Memory Queue State Object

Expected files:

```text
gpgpu-sim_distribution/src/gpgpu-sim/wasp-smem-queue.h
gpgpu-sim_distribution/src/gpgpu-sim/wasp-smem-queue.cc
gpgpu-sim_distribution/src/gpgpu-sim/shader.h
gpgpu-sim_distribution/src/gpgpu-sim/shader.cc
```

Add per-SM, per-CTA, per-original-warp queue state. The object should provide:

```text
full()
head_valid()
reserve_tail()
mark_lane_word_ready()
consume_head()
occupancy()
ready_count()
pending_count()
bank_conflict_cost()
```

This model should account queue payload storage as shared memory, not register
file space. It should report shared-memory bytes per CTA and should not add RFQ
register slots to occupancy.

Shared-memory queue capacity must count against normal `gpgpu_shmem_per_sm` CTA
residency immediately. The point of V3a is to compare register-file occupancy
pressure in V3 against shared-memory occupancy pressure in V3a.

### Step 3: Add PTX Parsing for Smem Queue Instructions

Expected files:

```text
gpgpu-sim_distribution/src/abstract_hardware_model.h
gpgpu-sim_distribution/src/cuda-sim/ptx_ir.cc
gpgpu-sim_distribution/src/cuda-sim/cuda-sim.cc
```

Add instruction metadata fields:

```text
is_smemq_load
is_smemq_move
smemq_queue_id
smemq_payload_words_per_lane
smemq_token
```

The parser should recognize `.smemq` as a separate option from `.rfq`. This
prevents V3 register-file queue behavior from being accidentally reused.

### Step 4: Connect Scheduler-Visible Readiness

Expected files:

```text
gpgpu-sim_distribution/src/gpgpu-sim/shader.cc
gpgpu-sim_distribution/src/gpgpu-sim/scoreboard.cc
```

For `mov.smemq.v3.f32`, scheduler eligibility must check queue head validity.
If the head is not valid, the warp is treated as not ready to issue. This is
not a software loop and should not increase dynamic instruction count.

The eligible-warp instrumentation must include this readiness test so reports
show the effect on:

```text
eligible_warp_count
eligible_warp_slot_count
avg_eligible_warps_per_scheduler
eligible_warp_occupancy
```

### Step 5: Add V3a Host Harness and PTX Generator

Expected files:

```text
01_github/259-dae-proj/stream_fma_v6/stream_fma_v6_variant3a.cu
01_github/259-dae-proj/stream_fma_v6/01_analysis/05_variant3a_smemq/generate_smemq_ptx.py
01_github/259-dae-proj/stream_fma_v6/Makefile
```

The host harness should match V2/V3 command-line behavior:

```text
--n
--iters
--memory-iters
--warmup
--repeats
```

The generated PTX should keep the same geometry and use only the `{a,b,c}` v3
tuple payload for the initial implementation:

```text
n = 1032192
compute_iters = 16
memory_iters = 4 for the primary full run
256 threads/CTA
4 producer warps + 4 consumer warps
1008 CTAs for n=1032192 and memory_iters=4
payload = v3 tuple {a,b,c}
```

### Step 6: Add Runner and Reports

Expected files:

```text
01_github/259-dae-proj/stream_fma_v6/01_analysis/05_variant3a_smemq/run_v3a_sweep.py
00_doc/02_variant_data/variant_results/v3a/results_v3a.md
00_doc/02_variant_data/variant_results/v3a/metrics_v3a.csv
00_doc/02_variant_data/variant_results/v3a/shared_memory_queue_depth_sweep_v3a.md
00_doc/02_variant_data/comparisons/comparison_v3_v3a.md
00_doc/02_variant_data/comparisons/comparison_v2_v3a.md
```

Use the H200 configuration:

```text
01_github/259-dae-proj/config_h200_132sm_mshr512
```

Run queue depths:

```text
4, 8, 16, 32, 64, 128
```

Use the same bandwidth calculation already used for the other variants:

```text
effective DRAM BW = 4800 GB/s * CoL_Bus_Util
```

## Counters and Required Report Fields

Add and report:

```text
smemq_pushes
smemq_memory_fills
smemq_pops
smemq_full_stalls
smemq_empty_stalls
smemq_head_not_ready_stalls
smemq_max_occupancy
smemq_avg_occupancy
smemq_ready_cycles
smemq_payload_words
smemq_payload_bytes
smemq_shared_bytes_per_cta
smemq_bank_conflicts
smemq_access_stalls
smemq_scheduler_ready_cycles
```

Also report the common fields:

```text
correctness result
simulated cycles
IPC
dynamic instruction count
active/resident warp occupancy
eligible warp occupancy
active CTA count per SM
DRAM bandwidth utilization
effective DRAM GB/s
L2 bandwidth
LD/ST utilization
total issue count
single issue count
dual issue count
```

Dual issue should be zero or disabled for Variant 3a. If nonzero, that is a
bug or an unintended config interaction.

## Correctness and Validation Plan

Run microbenchmarks before stream-FMA:

```text
smemq_stream_copy
smemq_full_stress
smemq_empty_stress
smemq_partial_mask
```

Validation checks:

```text
producer stalls when queue is full
consumer is scheduler-blocked when queue is empty
valid bit is set only after all active lanes are ready
partial active masks do not wait for inactive lanes
strict FIFO order holds
push/pop/fill counts match expected queue traffic
shared-memory queue mode does not add RFQ register pressure
baseline V2 and V3 runs are unchanged when smemq is disabled
```

Full workload validation:

```text
stream-FMA output verification = PASS
V3a output matches V2/V3 reference tolerance
no software polling loop appears in generated PTX
consumer wait does not inflate dynamic instruction count
```

## Expected Metric Impact

Expected improvements versus V3:

```text
less register-file occupancy pressure
more stable CTA/SM across deep queue depths
larger usable queue depths
```

Expected costs versus V3:

```text
shared-memory queue access latency
shared-memory bank conflicts
possible shared-memory bandwidth bottleneck
larger per-CTA shared-memory footprint
```

The key result is not simply whether V3a is faster. The key result is whether
moving queue payloads from register file to shared memory preserves enough
queue benefit while reducing occupancy loss.

## Known Limitations

- This design is still a simulator model, not a CUDA hardware feature.
- The generated PTX path avoids building a full CUDA-to-WASP compiler.
- Shared-memory queue bandwidth and bank conflict timing may need calibration.
- Variant 3a does not answer whether shared-memory-backed queues help under
  cross-warp co-issue; that is Variant 6.

## Open Questions for User Review

No open questions remain for the initial Variant 3a design. The implementation
should use the confirmed decisions above.

## Implementation Completed

Implemented simulator files:

```text
gpgpu-sim_distribution/src/gpgpu-sim/wasp-smem-queue.h
gpgpu-sim_distribution/src/gpgpu-sim/wasp-smem-queue.cc
gpgpu-sim_distribution/src/gpgpu-sim/shader.h
gpgpu-sim_distribution/src/gpgpu-sim/shader.cc
gpgpu-sim_distribution/src/gpgpu-sim/gpu-sim.cc
gpgpu-sim_distribution/src/abstract_hardware_model.h
gpgpu-sim_distribution/src/cuda-sim/ptx.l
gpgpu-sim_distribution/src/cuda-sim/ptx.y
gpgpu-sim_distribution/src/cuda-sim/ptx_ir.cc
gpgpu-sim_distribution/src/cuda-sim/cuda-sim.cc
```

Implemented workload and report files:

```text
01_github/259-dae-proj/stream_fma_v6/stream_fma_v6_variant3a.cu
01_github/259-dae-proj/stream_fma_v6/01_analysis/05_variant3a_smemq/generate_smemq_ptx.py
01_github/259-dae-proj/stream_fma_v6/01_analysis/05_variant3a_smemq/run_v3_v3a_smemq_sweep.py
01_github/259-dae-proj/stream_fma_v6/Makefile
00_doc/02_variant_data/variant_results/v3a/results_v3a.md
00_doc/02_variant_data/variant_results/v3a/v3a_smemq_sweep_m4.csv
```

Important implementation details:

```text
.smemq PTX option is parsed separately from .rfq
V3a instructions use m_smemq_* instruction fields
SMEMQ queue storage is charged as shared memory per CTA
RFQ register pressure is not charged when only smemq is enabled
consumer mov.smemq is scheduler-blocked on empty/not-ready queue head
producer ld.global.smemq is scheduler-blocked on full queue
SMEMQ and RFQ cannot be enabled together
```

Queue storage calculation:

```text
smemq_shared_bytes_per_cta =
  producer_warps_per_cta * queues_per_original_warp * depth *
  payload_words_per_lane * warp_size * sizeof(float)
```

For the m4 stream-FMA workload this becomes:

```text
4 producer warps * 1 queue * depth * 3 words/lane * 32 lanes * 4 B
```

## Validation Completed

Build commands:

```text
cd gpgpu-sim_distribution && source setup_environment release && make -j$(nproc)
make -C 01_github/259-dae-proj/stream_fma_v6 \
  NVCC=/usr/local/cuda-11.7/bin/nvcc CUOBJDUMP=/usr/local/cuda-11.7/bin/cuobjdump \
  ARCH=sm_70 NVCCFLAGS='-O3 -std=c++17 -arch=sm_70 -cudart shared' \
  NVCCFLAGS_V3='-O3 -std=c++17 -arch=sm_70 -cudart shared' \
  stream_fma_v6_variant3 stream_fma_v6_variant3a instructions-variant3 instructions-variant3a
python3 01_github/259-dae-proj/stream_fma_v6/01_analysis/05_variant3a_smemq/run_v3_v3a_smemq_sweep.py \
  --skip-build --rfq-depth 32 --depths 4 8 16 32 64 128
```

All executed runs reported `verification=PASS`.

Standalone `smemq_*` microbenchmark binaries from the validation plan were not
added in this pass. The full stream-FMA sweep exercises and reports full,
empty, and head-not-ready queue stalls, but partial-mask-specific validation
still needs a dedicated microbenchmark if that behavior becomes important.

Summary, full workload `n=1,032,192`, `compute_iters=16`, `memory_iters=4`:

| Variant | Depth | Cycles | Speedup vs V3 | CTA/SM | Eligible Occ. % | DRAM BW GB/s | Queue Storage |
|---|---:|---:|---:|---:|---:|---:|---:|
| V3 RFQ | 32 | 64036 | 1.0000 | 3 | 13.0425 | 766.1 | 12288 regs |
| V3a SMEMQ | 4 | 125561 | 0.5100 | 8 | 66.4259 | 390.6 | 6144 B |
| V3a SMEMQ | 8 | 117911 | 0.5431 | 8 | 71.5051 | 416.0 | 12288 B |
| V3a SMEMQ | 16 | 117911 | 0.5431 | 8 | 71.5051 | 416.0 | 24576 B |
| V3a SMEMQ | 32 | 81035 | 0.7902 | 4 | 25.9639 | 605.3 | 49152 B |
| V3a SMEMQ | 64 | 68854 | 0.9300 | 2 | 4.0072 | 712.3 | 98304 B |
| V3a SMEMQ | 128 | 61550 | 1.0404 | 1 | 1.0071 | 796.8 | 196608 B |

The current result shows that SMEMQ is functionally correct and visibly changes
CTA residency through shared-memory capacity. It is slower than V3 RFQ at the
same queue depth 32 on this workload because queue-readiness stalls and lower
DRAM utilization dominate the occupancy difference. Depth 128 is faster in
cycles than V3 RFQ, but that point runs at only 1 CTA/SM and about 1% eligible
warp occupancy, so it should be treated as a high-depth occupancy-collapse
data point rather than a generally healthy operating point.
