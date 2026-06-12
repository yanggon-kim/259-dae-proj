# Variant 6 Design: SMEMQ + Producer/Consumer Cross-Warp Co-Issue

## Status

Implemented. User approved execution after this design was written.

Branches:

```text
parent workspace:        variant6-smem-rfq
gpgpu-sim_distribution: variant6-smem-rfq
```

## Goal

Variant 6 combines the two useful mechanisms already implemented separately:

```text
Variant 4:  producer/consumer cross-warp co-issue scheduler
Variant 3a: shared-memory-backed queue with scheduler-visible readiness
```

Primary comparison:

```text
Variant 6 vs Variant 4
```

The goal is to test whether moving queue payload storage out of the register
file can preserve V4 co-issue behavior while improving CTA residency, active
warps, eligible readiness windows, and end-to-end cycles.

## Design Intention and Hypothesis

V4 proves the co-issue scheduler works, but the RFQ payload is modeled as
register-file queue storage. Large RFQ depth can reduce CTA residency.

V3a proves that SMEMQ can move queue payload storage into shared-memory-backed
storage with scheduler-visible readiness, but it does not enable cross-warp
co-issue.

V6 hypothesis:

```text
SMEMQ storage + scheduler-visible ready bits + V4 co-issue
can outperform RFQ + V4 when RFQ register pressure limits occupancy
or when SMEMQ creates cleaner consumer-ready windows.
```

The risk is that shared-memory queue traffic, queue metadata latency, or lower
consumer readiness quality can dominate and make V6 slower.

## Approved Starting Point

Start from the current V4 simulator branch state, not from clean V2:

```text
base scheduler behavior = V4 producer/consumer co-issue
queue storage to compare = RFQ in V4 vs SMEMQ in V6
```

Reuse V3a SMEMQ implementation infrastructure where possible:

```text
wasp-smem-queue.{h,cc}
smemq PTX instruction recognition
smemq scheduler-visible readiness check
smemq counters and occupancy accounting
```

Do not reuse RFQ register-pressure accounting for V6 SMEMQ payload storage.
V6 queue storage must count against shared memory, not register file capacity.

## Queue and Scheduler Semantics

Each logical producer/consumer warp pair owns one SMEMQ queue:

```text
producer warp 0 -> consumer warp 4
producer warp 1 -> consumer warp 5
producer warp 2 -> consumer warp 6
producer warp 3 -> consumer warp 7
```

For each queue entry:

```text
producer reserves tail slot
producer global loads complete
payload words are written into SM-local shared-memory-backed queue storage
active-lane ready mask reaches active-lane mask
head/tail/valid metadata commits
consumer warp becomes scheduler-visible ready
```

The queue logical pair is only for data transfer. The V4/V6 co-issue scheduler
must still be allowed to pair any ready producer warp with any ready consumer
warp, even if they are not logical queue partners.

Consumer wait must not use software polling. If `mov.smemq.v3.f32` reaches an
empty/not-valid queue head, the consumer warp is not eligible to issue until the
head entry becomes valid.

## Planned Implementation Steps

### Step 1: Config Integration

Use the existing V4 and V3a knobs together:

```text
-gpgpu_wasp_pc_coissue_enable 1
-gpgpu_smemq_enable 1
-gpgpu_smemq_scheduler_ready 1
-gpgpu_smemq_entries <depth>
-gpgpu_smemq_payload_words 3
-gpgpu_smemq_metadata_file <metadata>
```

Expected code areas:

```text
gpgpu-sim_distribution/src/gpgpu-sim/gpu-sim.cc
gpgpu-sim_distribution/src/gpgpu-sim/shader.h
gpgpu-sim_distribution/src/gpgpu-sim/shader.cc
```

Intent:

```text
allow V4 co-issue and SMEMQ readiness to be enabled simultaneously
ensure RFQ and SMEMQ storage/occupancy accounting stay separate
preserve V3, V3a, and V4 behavior when V6 knobs are disabled
```

### Step 2: Co-Issue Candidate Readiness Uses SMEMQ

Extend the V4 co-issue readiness path so a consumer warp with a
`mov.smemq.v3.f32` next instruction is ready only when its queue head is valid.

Expected code areas:

```text
gpgpu-sim_distribution/src/gpgpu-sim/shader.cc
gpgpu-sim_distribution/src/gpgpu-sim/shader.h
```

Intent:

```text
V6 co-issue should pair ready producer instructions with consumer instructions
whose SMEMQ data is actually available
eligible_warp_occupancy must include the same SMEMQ readiness filter
```

Required behavior:

```text
producer can co-issue with consumer only if both pass normal readiness checks
consumer blocked on empty SMEMQ does not count as eligible
producer/consumer logical pair is not required for scheduler co-issue
```

### Step 3: Queue Backpressure and Accounting

Keep V3a queue full/empty behavior and counters, and make them visible in V6
reports alongside V4 issue counters.

Required SMEMQ counters:

```text
smemq_push_count
smemq_pop_count
smemq_full_stall
smemq_empty_stall
smemq_max_occupancy
smemq_queue_storage_bytes_per_cta
smemq_payload_words_transferred
smemq_ready_cycles
smemq_bank_conflicts if available
```

Intent:

```text
explain whether V6 wins because of occupancy, readiness, queue depth, or
co-issue effectiveness
```

### Step 4: V6 Kernel/PTX Workloads

Use the existing V3a SMEMQ stream-FMA workload so V6 directly compares against
V4 and V3a:

```text
threads/CTA = 256
producer warps = 4
consumer warps = 4
payload = v3 tuple {a,b,c}
stream-FMA workload = memory_iters=4, compute_iters=128, CTA count=512
```

Also run the V4 kernel-family `sepconv_a` workload. This is not optional or
fallback-only; it is required because it is the current best V4-positive
co-issue workload:

```text
sepconv_a
```

Reason:

```text
sepconv_a was the best V4-fit RFQ kernel-family point at 1.023x over V3
stream-FMA tests V6 on the original project workload
```

PTX requirements:

```text
producer instruction: ld.global.smemq.v3.f32
consumer instruction: mov.smemq.v3.f32
metadata: producer_warps_per_cta=4, queues_per_original_warp=1,
          payload_words_per_lane=3
```

### Step 5: Evaluation Matrix

Primary V6 depth sweep:

```text
depth = 8, 16, 32, 64, 128
```

Run at:

```text
H200 config
single launch
--warmup 0 --repeats 1
```

Primary comparisons:

```text
V4 RFQ depth 32 vs V6 SMEMQ depth 32
V4 RFQ best known point vs V6 SMEMQ best depth
V3a SMEMQ no-coissue vs V6 SMEMQ coissue at matching depth
```

Required kernel-family comparison:

```text
sepconv_a V4 RFQ vs sepconv_a V6 SMEMQ
```

## Required Metrics

Every V6 result row must include:

```text
cycles
speedup vs V4
CTA/SM
gpu_tot_occupancy
eligible_warp_occupancy
avg_eligible_warps_per_scheduler
DRAM bw_util and effective GB/s
IPC
pc_coissue_attempts
pc_coissue_success
pc_coissue_success_rate
single_issue_count
dual_issue_count
single_issue_rate
dual_issue_success_rate
co-issue failure buckets
smemq depth
smemq max occupancy
smemq full stalls
smemq empty stalls
smemq storage bytes/CTA
verification status
```

DRAM bandwidth normalization:

```text
effective DRAM BW = 4800 GB/s * average bw_util
```

## Validation Plan

Before full sweeps:

```text
1. Build GPGPU-Sim on variant6-smem-rfq.
2. Run a one-CTA SMEMQ PTX smoke test with co-issue disabled.
3. Run the same one-CTA SMEMQ PTX smoke test with co-issue enabled.
4. Confirm verification=PASS in both.
5. Confirm V4 RFQ runs still work when SMEMQ is disabled.
6. Confirm V3a SMEMQ no-coissue runs still work when co-issue is disabled.
```

Full validation:

```text
stream-FMA primary workload passes numerical verification
sepconv V6 workload passes numerical verification if used
co-issue counters are nonzero when V6 co-issue is enabled
smemq counters are nonzero when SMEMQ is enabled
eligible occupancy changes consistently with SMEMQ readiness
```

## Output Artifacts

Create:

```text
00_doc/02_variant_data/variant_results/v6/metrics_v6.csv
00_doc/02_variant_data/variant_results/v6/results_v6.md
00_doc/02_variant_data/variant_results/v6/shared_memory_rfq_sweep_v6.md
00_doc/02_variant_data/comparisons/comparison_v4_v6.md
```

If sepconv V6 is used, also create:

```text
00_doc/02_variant_data/variant_results/v6/results_v6_kernel_family.md
00_doc/02_variant_data/variant_results/v6/metrics_v6_kernel_family.csv
```

## Expected Outcomes

Positive V6 evidence:

```text
V6 cycles < V4 cycles at matching workload
CTA/SM higher than RFQ at comparable queue depth
eligible warp occupancy does not collapse from SMEMQ readiness
pc_coissue_success and dual_issue_success_rate remain comparable to V4
smemq full/empty stalls are low or explainable
```

Negative V6 evidence:

```text
V6 has higher CTA/SM but worse cycles due to SMEMQ stalls
consumer readiness is too sparse for co-issue
shared-memory queue traffic becomes the bottleneck
dual_issue_success_rate drops sharply versus V4
```

## Open Questions for User Review

Default assumptions unless changed:

```text
primary V6 stream workload = memory_iters=4, compute_iters=128, CTA count=512
primary SMEMQ payload = v3 tuple {a,b,c}
primary depth sweep = 8, 16, 32, 64, 128
required kernel-family workload = sepconv_a
```

User decision:

```text
Run both stream-FMA and sepconv_a for V6 evaluation.
```

## Implementation Notes

Simulator branch:

```text
gpgpu-sim_distribution: variant6-smem-rfq
```

Code changes:

```text
src/gpgpu-sim/shader.cc
  - allow cross-warp co-issue when either RFQ or SMEMQ is enabled
  - use SMEMQ producer-warp metadata for role classification when SMEMQ is active

src/gpgpu-sim/gpu-sim.cc
  - clarify the co-issue option text as RFQ-or-SMEMQ capable
```

Tooling and PTX in this consolidated package:

```text
01_gpgpusim_design/tools/run_stream_fma_reference.sh
01_gpgpusim_design/ptx_overrides/smemq_based/
01_gpgpusim_design/configs/smemq_dual_issue_depth32/
```

Packaged result summaries:

```text
01_gpgpusim_design/results_reference/metrics_smemq_dual_issue.csv
01_gpgpusim_design/results_reference/results_smemq_dual_issue.md
01_gpgpusim_design/results_reference/smemq_dual_issue_depth_sweep.md
```
