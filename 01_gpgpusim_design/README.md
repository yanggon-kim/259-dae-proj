# GPGPU-Sim Warp-Specialization Design

This directory is the consolidated reproduction package for the GPGPU-Sim implementation used in the warp-specialization study. It replaces the old split directories `01_SMEMQ_design/` and `02_dual_issue_design/`.

The package does not vendor the full upstream GPGPU-Sim tree. Instead, `gpgpusim_overlay/` contains the actual modified source files, stored with their original GPGPU-Sim relative paths. Applying the overlay to a compatible GPGPU-Sim 4.2.0-style checkout gives the simulator implementation used for the experiments.

## Directory Layout

```text
gpgpusim_overlay/       # Source replacements/additions for GPGPU-Sim
configs/                # H200-like base config and variant-specific configs
design_notes/           # Per-variant design intent and implementation notes
kernels/                # CUDA host/kernel reference harnesses
ptx_overrides/          # Queue PTX, ptxinfo, and metadata files
results_reference/      # Reference logs, CSVs, and markdown summaries
report_support/         # Report-ready design/result text and diagrams
tools/                  # Overlay installer, run helper, parsers
VARIANT_GUIDE.md        # Variant-by-variant design map
```

## Variant Map

| Label | Report term | Main implementation change |
|---|---|---|
| V1 | Baseline Stream-FMA | Homogeneous Stream-FMA CUDA kernel. No simulator changes. |
| V2 | Warp-specialized Stream-FMA | Producer/consumer CUDA geometry with shared-memory handoff. No simulator changes. |
| V3 | RFQ | Adds `ld.global.rfq` and `mov.rfq`, RFQ storage, RFQ stalls, RFQ register-pressure CTA accounting. |
| V3a | SMEM-based queue | Adds `ld.global.smemq` and `mov.smemq`, shared-memory queue storage, scheduler-visible valid/ready metadata. |
| V4 | RFQ + dual-issue | Adds producer/consumer dual-issue on top of RFQ. |
| V6 | SMEM-based queue + dual-issue | Enables the same dual-issue path for SMEM-based queues. |

## Apply the Simulator Overlay

```bash
cd <this-repo>
01_gpgpusim_design/tools/install_gpgpusim_overlay.sh <gpgpu-sim_distribution>

cd <gpgpu-sim_distribution>
export CUDA_INSTALL_PATH=/usr/local/cuda-11.7
source setup_environment release
make -j$(nproc)
```

The overlay contains the modified GPGPU-Sim files directly. No patch application is required.

## Build the Reference Kernel

```bash
cd <this-repo>/01_gpgpusim_design/kernels/stream_fma_reference
make NVCC=/usr/local/cuda-11.7/bin/nvcc ARCH=sm_70
```

The default Stream-FMA experiment uses:

```text
n=524288
memory_iters=4
compute_iters=128
CTA count=512
threads/CTA=256
producer warps/CTA=4
consumer warps/CTA=4
queue depth=32 for RFQ/SMEM-based queue reference points
```

## Run Reference Cases

From `01_gpgpusim_design/`:

```bash
tools/run_stream_fma_reference.sh rfq
tools/run_stream_fma_reference.sh smemq
tools/run_stream_fma_reference.sh rfq-dual
tools/run_stream_fma_reference.sh smemq-dual
```

The script creates run directories under `01_gpgpusim_design/runs/`, copies the relevant config, and uses the PTX override environment variables:

```text
PTX_SIM_USE_PTX_FILE=1
PTX_SIM_KERNELFILE=<queue PTX>
PTX_SIM_PTXINFO_FILE=<queue ptxinfo>
```

Expected correctness line:

```text
verification=PASS
```

## Configuration Sets

| Config directory | Purpose |
|---|---|
| `configs/h200_132sm_mshr512/` | Corrected H200-like base configuration. |
| `configs/ws_base/` | Warp-specialized CUDA baseline configuration. |
| `configs/rfq_based_depth32/` | RFQ depth-32, dual-issue disabled. |
| `configs/smemq_based_depth32/` | SMEM-based queue depth-32, dual-issue disabled. |
| `configs/rfq_dual_issue_depth32/` | RFQ depth-32 with producer/consumer dual-issue enabled. |
| `configs/smemq_dual_issue_depth32/` | SMEM-based queue depth-32 with producer/consumer dual-issue enabled. |

For queue-depth sweeps, change `-gpgpu_wasp_rfq_entries` or `-gpgpu_smemq_entries` in the selected config and keep the matching metadata/PTX structure.

## Important Metrics

The simulator overlay reports:

- `eligible_warp_occupancy`
- `avg_eligible_warps_per_scheduler`
- RFQ queue stalls, occupancy, and register-pressure CTA adjustment
- SMEM-based queue stalls, shared-memory footprint, metadata bits, and CTA adjustment
- `pc_coissue_attempts`, `pc_coissue_success`, and failure buckets
- `single_issue_count`, `dual_issue_count`, and dual-issue success rate

Effective DRAM bandwidth in the reports uses:

```text
24 HBM channels * 32 B/channel * 2 DDR ratio * 3125 MHz = 4800 GB/s
effective DRAM BW = 4800 GB/s * average bw_util
```

## Reference Results

Headline results are stored in `results_reference/` and report-ready explanations are in `report_support/`.

For the Main Stream-FMA point, the key results are:

| Configuration | Cycles | CTA/SM | Active Occ. | Eligible Occ. | DRAM BW |
|---|---:|---:|---:|---:|---:|
| RFQ depth 32 | 80183 | 3 | 37.5% | 11.5139% | 310.704 GB/s |
| SMEM-based queue depth 32 | 63937 | 4 | 50.0% | 18.3275% | 389.664 GB/s |
| RFQ + dual-issue depth 32 | 80304 | 3 | 37.5% | 11.5402% | 310.272 GB/s |
| SMEM-based queue + dual-issue depth 32 | 63343 | 4 | 50.0% | 18.0341% | 393.312 GB/s |

Speedup is always:

```text
speedup = baseline cycles / measured cycles
```

For the combined SMEM-based queue plus dual-issue result, the depth-8, depth-16, and depth-32 points tie at `63343` cycles.
