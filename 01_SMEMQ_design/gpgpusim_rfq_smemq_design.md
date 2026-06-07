# RFQ/SMEMQ GPGPU-Sim Design

## Goal

This package compares two scheduler-visible producer/consumer queues on the
same warp-specialized stream-FMA workload:

```text
RFQ-based implementation   = queue payload stored in register-file capacity
SMEMQ-based implementation = queue payload stored in shared-memory capacity
```

Both implementations keep the same CTA geometry, producer/consumer warp split,
H200-like GPGPU-Sim configuration, and generated queue PTX. Neither
implementation adds cross-warp co-issue or same-cycle dual issue.

## Simulator Patch

`gpgpusim_rfq_smemq.patch` adds the simulator support needed for both queue
types:

- PTX parser support for `.rfq` and `.smemq` pseudo modifiers.
- PTX override support through `PTX_SIM_USE_PTX_FILE` and
  `PTX_SIM_KERNELFILE`.
- Optional ptxinfo override support through `PTX_SIM_PTXINFO_FILE`.
- RFQ state in `wasp-rfq.*`.
- SMEMQ state in `wasp-smem-queue.*`.
- Scheduler eligibility checks so consumer warps issue only when queue heads
  are ready.
- CTA residency accounting for RFQ register pressure and SMEMQ shared-memory
  pressure.

The patch was generated against GPGPU-Sim commit:

```text
a4ce3feac901c97a4b4601f679e43cf3589c79de
```

## Queue Semantics

Each logical producer/consumer warp pair owns one FIFO queue. For this
workload, producer warps are `0..3`, consumer warps are `4..7`, and each queue
entry carries one `{a,b,c}` tuple per active lane.

RFQ behavior:

- Producer queue loads reserve a tail slot and attach a queue token to memory
  requests.
- The entry becomes ready when all active-lane payload responses arrive.
- Consumer queue moves read the ready head entry and advance the FIFO.
- Queue payload words count against register-file capacity when RFQ register
  pressure accounting is enabled.

SMEMQ behavior:

- Producer queue loads reserve a tail slot in the shared-memory-backed queue.
- Ready metadata is scheduler-visible; consumers do not spin on software flags.
- Queue payload bytes count against shared-memory CTA residency immediately.
- The valid bit is set only after all active producer lanes have completed the
  payload for the queue entry.

## Configuration Knobs

RFQ-based run:

```text
-gpgpu_wasp_enable 1
-gpgpu_wasp_rfq_enable 1
-gpgpu_wasp_rfq_entries 32
-gpgpu_wasp_rfq_count_register_pressure 1
-gpgpu_wasp_rfq_model_data 1
-gpgpu_wasp_rfq_strict_fifo 1
-gpgpu_wasp_rfq_assert_mask_match 1
-gpgpu_wasp_metadata_file ../../ptx_overrides/rfq_based/stream_fma_v6_rfq_based.metadata
```

SMEMQ-based run:

```text
-gpgpu_smemq_enable 1
-gpgpu_smemq_entries 32
-gpgpu_smemq_scheduler_ready 1
-gpgpu_smemq_model_data 1
-gpgpu_smemq_assert_mask_match 1
-gpgpu_smemq_bank_conflict_model 0
-gpgpu_smemq_metadata_file ../../ptx_overrides/smemq_based/stream_fma_v6_smemq_based.metadata
```

Shared metadata format:

```text
num_pipeline_stages = 2
producer_warps_per_cta = 4
queues_per_original_warp = 1
payload_words_per_lane = 3
element_bits = 96
```

## H200 Configuration

The base H200-like configuration is in `configs/h200_base/`. The key memory
settings are:

```text
SMs = 132
HBM channels = 24
DRAM bus width = 32 B/channel
DDR ratio = 2
DRAM clock = 3125 MHz
FR-FCFS scheduler queue size = 256
DRAM return queue size = 512
```

The full DRAM bandwidth used for utilization conversion is `4800 GB/s`.

## Measurement Interpretation

At depth 32 and the packaged workload size, SMEMQ stores queue payload outside
the register file. That raises the measured CTA residency from 3 CTA/SM to
4 CTA/SM relative to RFQ, increasing active and eligible warp occupancy. The
included reference run shows SMEMQ reducing total cycles from `80,183` to
`63,937`, a speedup of about `1.254x`, while preserving `verification=PASS`.

The design tradeoff is that SMEMQ spends shared-memory capacity and can create
shared-memory pressure, while RFQ spends register-file capacity and can reduce
resident CTA count earlier as queue depth increases.
