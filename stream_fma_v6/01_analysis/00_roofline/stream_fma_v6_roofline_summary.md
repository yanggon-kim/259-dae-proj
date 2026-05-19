# stream_fma_v6 Roofline Summary

## Scope

This profiles `stream_fma_v6_m1_kernel` with `n=1032192` for the roofline. v6 uses
a conventional grid-size-from-work launch like v5, but each thread processes
`memory_iters` elements in one CTA tile. There is no outer `round` loop. Tensor
Core throughput is excluded. FLOPs use NCU counters: `2 * FFMA + FADD`.

## Hardware Ceilings

- GPU: NVIDIA GeForce RTX 5080
- Spec CUDA-core FP32 peak: 56.349 TFLOP/s
- Spec DRAM bandwidth: 960.0 GB/s
- NCU clock-adjusted CUDA-core FP32 ceiling: 55.910 TFLOP/s
- NCU clock-adjusted DRAM ceiling: 941.4 GB/s

## Compute-Iters Sweep

| compute_iters | AI algorithm | v6 m1 GFLOP/s | v2 m1 GFLOP/s | DRAM GB/s | DRAM peak % | long score % | active occ % | eligible % |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.125 | 137.3 | 132.7 | 823.9 | 87.13 | 95.88 | 84.88 | 0.85 |
| 1 | 0.500 | 544.4 | 538.7 | 816.9 | 86.49 | 91.43 | 84.98 | 1.17 |
| 2 | 0.875 | 954.7 | 936.9 | 818.7 | 86.73 | 92.91 | 84.63 | 1.38 |
| 4 | 1.625 | 1769.3 | 1715.0 | 816.9 | 86.63 | 89.66 | 85.74 | 1.91 |
| 8 | 3.125 | 3353.0 | 3374.1 | 805.0 | 85.41 | 86.62 | 84.97 | 2.37 |
| 16 | 6.125 | 6613.2 | 6585.6 | 810.1 | 85.70 | 86.17 | 85.76 | 3.47 |
| 32 | 12.125 | 13064.0 | 12849.4 | 808.4 | 85.70 | 82.22 | 84.73 | 6.15 |
| 64 | 24.125 | 25255.2 | 24557.8 | 785.4 | 82.97 | 56.84 | 83.57 | 22.90 |
| 128 | 48.125 | 38507.2 | 36153.0 | 600.3 | 63.50 | 23.00 | 83.72 | 50.06 |

## Interpretation

For `memory_iters=1`, v6 and v5 have the same one-thread-per-element mapping.
The v6 memory sweep changes CTAs instead of outer rounds: larger
`memory_iters` means fewer CTAs and more elements per thread in the same launch.
