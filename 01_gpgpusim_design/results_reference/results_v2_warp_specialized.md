# Variant 2 Results

## Summary

Variant 2 is the true double-buffered producer/consumer warp-specialized version of `stream_fma_v6`. It keeps the Variant 1 launch geometry and workload: 1008 CTAs, 256 threads/CTA, 1024 elements/CTA, `--n 1032192 --iters 16 --memory-iters 4 --warmup 0 --repeats 1`.

No simulator change is used for Variant 2. The only simulator addition is the eligible-warp instrumentation already added during Variant 1.

## Implementation

- Kernel used: `stream_fma_v6_v2_m4_kernel`.
- Producer warps: warp IDs 0-3.
- Consumer warps: warp IDs 4-7.
- Tile shape: 1024 elements/CTA split into 8 chunks of 128 elements.
- Handoff: two shared-memory buffers, each holding `a`, `b`, and `c` for one 128-element chunk.
- Pipeline: producers load chunk `k+1` while consumers compute and store chunk `k`.
- Synchronization: one `__syncthreads()` per chunk.
- Shared memory: 3072 bytes per CTA.
- Not included: scheduler co-issue, register-file queue, queue-depth counters, or RFQ backpressure.

## Build and Run

```bash
/usr/local/cuda-11.7/bin/nvcc -O3 -std=c++17 \
  -gencode arch=compute_70,code=compute_70 -cudart shared \
  -o 00_doc/02_variant_data/variant_results/v2/stream_fma_v6_variant2_gpgpusim \
  01_github/259-dae-proj/stream_fma_v6/stream_fma_v6_variant2.cu
```

Full run command:

```bash
./stream_fma_v6_variant2_gpgpusim --n 1032192 --iters 16 --memory-iters 4 --warmup 0 --repeats 1
```

## Full-Run Metrics

| Metric | Value |
| --- | ---: |
| Correctness | PASS |
| Simulated cycles | 128023 |
| IPC | 937.2716 |
| Dynamic instructions | 119992320 |
| Warp instructions | 3911040 |
| Issued CTAs | 1008 |
| Resident CTA/SM limit | 8 |
| Active warp occupancy | 90.3453% |
| Eligible warp occupancy | 1.0776% |
| Avg eligible warps/scheduler | 0.1724 |
| DRAM `CoL_Bus_Util` | 0.079820 |
| Effective DRAM BW | 383.1360 GB/s |
| L2 BW | 255.4216 GB/s |
| L2 accesses / misses | 516096 / 516096 |
| L2 reservation fails | 361379 |
| Global read / write transactions | 387072 / 129024 |
| Load / store instructions | 3096576 / 1032192 |
| Shared-memory instructions | 6193152 |
| Shared-memory bank conflicts | 0 |
| L1D bank conflicts | 0 |
| Single-issue warp instructions | 3911040 |
| Same-warp dual-issue instructions | 0 |
| Producer/consumer co-issue | N/A |

Effective DRAM bandwidth uses the same H200 full-bandwidth normalization as Variant 1:

```text
24 HBM channels x 32 B/channel x 2 DDR ratio x 3125 MHz = 4800 GB/s
4800 GB/s x 0.079820 = 383.1360 GB/s
```

## Validation

The sanity run (`--n 32768`) and full run (`--n 1032192`) both passed verification. Raw logs are `sanity_run.log` and `full_run.log`; the machine-readable summary is `metrics_v2.csv`.

## Interpretation

Compared with Variant 1, true Variant 2 is slower by simulated cycles. It preserves the same global read/write transaction counts, but adds shared-memory transfer instructions and CTA barriers. The eligible-warp occupancy drops sharply, indicating that the barrier-coupled producer/consumer handoff is limiting scheduler-ready work in the current baseline GPGPU-Sim configuration.
