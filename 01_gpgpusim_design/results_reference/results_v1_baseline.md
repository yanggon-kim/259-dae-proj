# Variant 1 Results

## Summary

Variant 1 establishes the baseline for later warp-specialization and RFQ experiments: the unmodified `stream_fma_v6` kernel running under the faithful H200-like GPGPU-Sim configuration.

- Kernel: `stream_fma_v6_m4_kernel`
- Kernel source: `01_github/259-dae-proj/stream_fma_v6/stream_fma_v6.cu`
- Config source: `02_h200_config/config_h200_132sm_mshr512`
- Run directory: `00_doc/02_variant_data/variant_results/v1`
- GPGPU-Sim commit: `a4ce3feac901c97a4b4601f679e43cf3589c79de`
- Project commit: N/A; `01_github/259-dae-proj` is not a Git repository in this workspace.

## Build

The baseline binary was built with CUDA 11.7, compute 7.0 PTX, and shared cudart so GPGPU-Sim can intercept CUDA runtime calls:

```bash
/usr/local/cuda-11.7/bin/nvcc -O3 -std=c++17 \
  -gencode arch=compute_70,code=compute_70 -cudart shared \
  -o 00_doc/02_variant_data/variant_results/v1/stream_fma_v6_gpgpusim \
  01_github/259-dae-proj/stream_fma_v6/stream_fma_v6.cu
```

`-lineinfo` was intentionally omitted because this GPGPU-Sim PTX parser rejected generated `.loc ... inlined_at` records.

## Commands

Environment setup:

```bash
export CUDA_INSTALL_PATH=/usr/local/cuda-11.7
cd gpgpu-sim_distribution
source setup_environment release
cd ../00_doc/02_variant_data/variant_results/v1
```

Sanity run:

```bash
./stream_fma_v6_gpgpusim --n 32768 --iters 16 --memory-iters 4 --warmup 0 --repeats 1
```

Full baseline run:

```bash
./stream_fma_v6_gpgpusim --n 1032192 --iters 16 --memory-iters 4 --warmup 0 --repeats 1
```

## Full Run Metrics

| Metric | Value |
| --- | ---: |
| Correctness | PASS |
| Simulated cycles | 102287 |
| IPC | 691.2428 |
| Dynamic instruction count | 70705152 |
| Warp instruction count | 2233728 |
| Issued CTAs | 1008 |
| Average issued CTAs per SM | 7.6364 |
| Active/resident warp occupancy | 92.5667% |
| Eligible warp occupancy | 75.0682% |
| Average eligible warps per scheduler | 12.0109 |
| L2 bandwidth | 319.6872 GB/s |
| L2 accesses / misses | 516096 / 516096 |
| L2 miss rate | 1.0000 |
| L2 reservation fails | 620976 |
| Global read / write transactions | 387072 / 129024 |
| Load / store instructions | 3096576 / 1032192 |
| Shared-memory instructions | 0 |
| Shared-memory bank conflicts | 0 |
| L1D bank conflicts | 8043299 |
| Single issue count | 2233728 |
| Same-warp dual issue count | 0 |
| Producer/consumer co-issue count | N/A |
| Queue depth / full stalls / empty stalls | N/A |

## Required Report Fields

Active/resident warp occupancy is `gpu_tot_occupancy = 92.5667%`. This is GPGPU-Sim's resident active-warp-slot metric, not an eligible-warp metric.

Eligible warp occupancy is now measured by added scheduler counters in `gpgpu-sim_distribution/src/gpgpu-sim/shader.cc` and `shader.h`. A warp is counted eligible after scheduler ordering and before issue when it is not exited, not barrier-waiting, has a valid next instruction at the current SIMT stack PC, and passes scoreboard dependency checks. The full run reports `avg_eligible_warps_per_scheduler = 12.0109` and `eligible_warp_occupancy = 75.0682%`, where the denominator is the scheduler-owned 16 warp slots.

FP/INT/LD-ST/Tensor utilization and active CTA count per SM are not directly exposed by the current log format. The closest CTA proxy is `gpu_tot_issued_cta = 1008`, or 7.6364 issued CTAs per SM over 132 SMs.

DRAM bandwidth utilization is reported from the simulator memory-partition details as `CoL_Bus_Util = 0.099903`, or 4800 GB/s x 0.099903 = 479.5 GB/s effective DRAM bandwidth. `Either_Row_CoL_Bus_Util = 0.145407` is retained as a supporting bus counter. L2 utilization is represented by `L2_BW_total = 319.6872 GB/Sec`, `L2_cache_data_port_util = 0.000`, and `L2_cache_fill_port_util = 0.079`.

## Validation

The full run with eligible-warp instrumentation reported `verification=PASS` and preserved the previous cycle, IPC, dynamic instruction, issue, and cache totals. A sanity run with the same instrumentation also passed and reported `eligible_warp_occupancy = 1.3106%`.

## Memory-Iteration Sweep

Full runs were also collected for `--memory-iters` 1, 2, 4, and 8 to measure how the kernel memory-iteration template affects eligible warp occupancy and DRAM bandwidth utilization. Detailed results are in `memory_iters_sweep_v1.md` and `memory_iters_sweep_v1.csv`.

The peak DRAM data-bus utilization was `memory_iters=1`: `CoL_Bus_Util = 0.214388`, or 1029.0624 GB/s effective DRAM BW. The peak eligible warp occupancy was `memory_iters=4`: 75.0682%, with 12.0109 average eligible warps per scheduler. `memory_iters=8` launched only 504 CTAs and was limited to 6 resident CTAs/SM by registers, so active occupancy fell to 45.8412% and eligible occupancy fell to 32.0689%.

## Conclusion

Variant 1 is implemented and validated as the normalization baseline. It uses the faithful H200-like configuration and the normal baseline kernel with no scheduler, queue, or kernel specialization changes.
