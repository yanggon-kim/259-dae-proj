# Variant Guide

This file maps the experiment variants to the files and simulator features in this package.

## V1: Baseline Stream-FMA

Purpose: establish the homogeneous CUDA baseline.

Implementation:

- Source: `../00_kernels/stream_fma_v6/stream_fma_v6.cu`
- Simulator changes: none
- Config: `configs/h200_132sm_mshr512/`
- Results: `results_reference/results_v1_baseline.md`

## V2: Warp-Specialized Stream-FMA

Purpose: split the same Stream-FMA work into producer and consumer warps using CUDA/shared-memory handoff.

Implementation:

- Source: `kernels/stream_fma_reference/stream_fma_v6_v2_m4_c128_cta512.cu`
- Producer warps: 0-3
- Consumer warps: 4-7
- Simulator changes: none
- Results: `results_reference/results_v2_warp_specialized.md`

## V3: RFQ

Purpose: model a WASP-style register-file queue for producer/consumer handoff.

Implementation:

- PTX instructions: `ld.global.rfq.v3.f32`, `mov.rfq.v3.f32`
- Queue state: `gpgpusim_overlay/src/gpgpu-sim/wasp-rfq.*`
- Parser/execution changes: `gpgpusim_overlay/src/cuda-sim/*`
- Scheduler/CTA accounting: `gpgpusim_overlay/src/gpgpu-sim/shader.*`
- Config: `configs/rfq_based_depth32/`
- PTX override: `ptx_overrides/rfq_based/`
- Results: `results_reference/results_rfq.md`

Key knobs:

```text
-gpgpu_wasp_enable 1
-gpgpu_wasp_rfq_enable 1
-gpgpu_wasp_rfq_entries 32
-gpgpu_wasp_rfq_count_register_pressure 1
-gpgpu_wasp_metadata_file ../../ptx_overrides/rfq_based/stream_fma_v6_rfq_based.metadata
```

## V3a: SMEM-Based Queue

Purpose: move queue payload storage from register-file capacity to shared-memory capacity while keeping valid/ready metadata visible to the scheduler.

Implementation:

- PTX instructions: `ld.global.smemq.v3.f32`, `mov.smemq.v3.f32`
- Queue state: `gpgpusim_overlay/src/gpgpu-sim/wasp-smem-queue.*`
- Parser/execution changes: `gpgpusim_overlay/src/cuda-sim/*`
- Shared-memory CTA accounting: `shader_core_config::smemq_shared_bytes_per_cta`
- Config: `configs/smemq_based_depth32/`
- PTX override: `ptx_overrides/smemq_based/`
- Results: `results_reference/results_smemq.md`

Key knobs:

```text
-gpgpu_smemq_enable 1
-gpgpu_smemq_entries 32
-gpgpu_smemq_scheduler_ready 1
-gpgpu_smemq_model_data 1
-gpgpu_smemq_metadata_file ../../ptx_overrides/smemq_based/stream_fma_v6_smemq_based.metadata
```

## V4: RFQ + Dual-Issue

Purpose: test whether a producer warp instruction and a consumer warp instruction can issue in the same cycle with RFQ storage.

Implementation:

- Scheduler path: `scheduler_unit::try_pc_coissue`
- Producer/consumer classification: metadata-driven warp role detection
- Issue counters: `pc_coissue_*`, `single_issue_count`, `dual_issue_count`
- Config: `configs/rfq_dual_issue_depth32/`
- Results: `results_reference/results_rfq_dual_issue.md`
- Kernel search results: `results_reference/results_rfq_dual_issue_kernel_search.md`

Key knob:

```text
-gpgpu_wasp_pc_coissue_enable 1
```

## V6: SMEM-Based Queue + Dual-Issue

Purpose: combine SMEM-based queue storage with the same producer/consumer dual-issue scheduler path.

Implementation:

- Uses the SMEM-based queue parser/execution and scheduler-visible readiness logic.
- Enables `scheduler_unit::try_pc_coissue` when SMEM-based queues are active.
- Uses SMEM metadata for producer-warp classification when SMEM-based queues are enabled.
- Config: `configs/smemq_dual_issue_depth32/`
- Results: `results_reference/results_smemq_dual_issue.md`

Key knobs:

```text
-gpgpu_smemq_enable 1
-gpgpu_smemq_entries 32
-gpgpu_smemq_scheduler_ready 1
-gpgpu_wasp_pc_coissue_enable 1
```

## Notes

- V1 and V2 are kernel/config baselines and do not require the simulator overlay.
- V3 and later require the simulator overlay.
- V4 and V6 use the same dual-issue scheduler feature but different queue storage backends.
- Depth sweeps change only the queue-depth knob and the matching run directory.
