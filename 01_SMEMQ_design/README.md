# RFQ/SMEMQ Reproduction Guide

This directory packages the RFQ-based and SMEMQ-based stream-FMA measurements
for reproduction in another GPGPU-Sim environment.

## Contents

```text
gpgpusim_rfq_smemq.patch        # RFQ/SMEMQ simulator implementation patch
gpgpusim_rfq_smemq_design.md    # Design and implementation notes
configs/h200_base/              # Corrected H200 Tier-1 base config
configs/rfq_based_depth32/      # RFQ-based run config
configs/smemq_based_depth32/    # SMEMQ-based run config
kernel_reference/               # CUDA host and warp-specialized reference kernel
ptx_overrides/                  # Generated queue PTX, metadata, and ptxinfo files
results_reference/              # Reference logs and aggregate metrics
tools/extract_reference_metrics.py
```

## Workload

| Parameter | Value |
|---|---:|
| CTA count | 512 |
| Threads/CTA | 256 |
| Producer warps/CTA | 4 |
| Consumer warps/CTA | 4 |
| `memory_iters` | 4 |
| `compute_iters` | 128 |
| Queue depth | 32 |
| `n` | 524,288 |

## Build GPGPU-Sim

Apply the simulator patch to a clean GPGPU-Sim checkout based on commit
`a4ce3feac901c97a4b4601f679e43cf3589c79de` or the matching upstream dev tree.

```bash
cd <gpgpu-sim>
git apply <259-dae-proj>/01_SMEMQ_design/gpgpusim_rfq_smemq.patch
export CUDA_INSTALL_PATH=/usr/local/cuda-11.7
source setup_environment release
make -j$(nproc)
```

## Build the Kernel Harness

```bash
cd <259-dae-proj>/01_SMEMQ_design/kernel_reference
make NVCC=/usr/local/cuda-11.7/bin/nvcc ARCH=sm_70
```

The CUDA file is the host/geometry harness. The queue behavior comes from the
PTX overrides in `ptx_overrides/`.

## Run RFQ-Based Measurement

```bash
cd <259-dae-proj>/01_SMEMQ_design
mkdir -p runs/rfq_based_depth32
cp configs/rfq_based_depth32/gpgpusim.config runs/rfq_based_depth32/
cp configs/rfq_based_depth32/config_volta_islip.icnt runs/rfq_based_depth32/
cp configs/h200_base/accelwattch*.xml runs/rfq_based_depth32/

cd runs/rfq_based_depth32
PTX_SIM_USE_PTX_FILE=1 \
PTX_SIM_KERNELFILE=../../ptx_overrides/rfq_based/stream_fma_v6_rfq_based.ptx \
PTX_SIM_PTXINFO_FILE=../../ptx_overrides/rfq_based/stream_fma_v6_rfq_based.ptxinfo \
../../kernel_reference/stream_fma_v6_v2_m4_c128_cta512 \
  --n 524288 --iters 128 --memory-iters 4 --warmup 0 --repeats 1 \
  > full_run.log 2>&1
```

## Run SMEMQ-Based Measurement

```bash
cd <259-dae-proj>/01_SMEMQ_design
mkdir -p runs/smemq_based_depth32
cp configs/smemq_based_depth32/gpgpusim.config runs/smemq_based_depth32/
cp configs/smemq_based_depth32/config_volta_islip.icnt runs/smemq_based_depth32/
cp configs/h200_base/accelwattch*.xml runs/smemq_based_depth32/

cd runs/smemq_based_depth32
PTX_SIM_USE_PTX_FILE=1 \
PTX_SIM_KERNELFILE=../../ptx_overrides/smemq_based/stream_fma_v6_smemq_based.ptx \
PTX_SIM_PTXINFO_FILE=../../ptx_overrides/smemq_based/stream_fma_v6_smemq_based.ptxinfo \
../../kernel_reference/stream_fma_v6_v2_m4_c128_cta512 \
  --n 524288 --iters 128 --memory-iters 4 --warmup 0 --repeats 1 \
  > full_run.log 2>&1
```

## Extract Metrics

```bash
cd <259-dae-proj>/01_SMEMQ_design
tools/extract_reference_metrics.py \
  results_reference/rfq_based_depth32/full_run.log \
  results_reference/smemq_based_depth32/full_run.log
```

Reference headline metrics:

| Implementation | Cycles | CTA/SM | Active Occ. | Eligible Occ. | DRAM BW |
|---|---:|---:|---:|---:|---:|
| RFQ-based | 80,183 | 3 | 37.5% | 11.5139% | 310.704 GB/s |
| SMEMQ-based | 63,937 | 4 | 50.0% | 18.3275% | 389.664 GB/s |

The H200 bandwidth conversion used for these reports is:

```text
24 HBM channels * 32 B/channel * 2 DDR ratio * 3125 MHz = 4800 GB/s
effective_dram_bw = 4800 GB/s * dram_bw_util
```

Expected correctness line in each log:

```text
verification=PASS
```
