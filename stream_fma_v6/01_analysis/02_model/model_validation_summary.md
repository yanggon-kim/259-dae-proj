# stream_fma_v6 Model Validation

## Model

Inputs are `n`, `compute_iters`, `memory_iters`, `threads_per_cta`, and hardware
parameters. The model computes work, traffic, CTA mapping, warp occupancy, and
the bottleneck time:

```text
ops = n * (6 * compute_iters + 2)
algorithm_bytes = n * 16
ctas = n / (threads_per_cta * memory_iters)
predicted_time = max(compute_time, dram_time)
TOPS = ops / predicted_time / 1e12
```

The calibrated v6 model uses 12 modeled DRAM bytes per element because NCU
measures this kernel as read-dominated DRAM traffic, and it uses exposed
efficiency terms for scalar FMA issue and streaming DRAM bandwidth.

## Accuracy

| sweep | v6 model time MAPE | peak roofline time MAPE |
| --- | ---: | ---: |
| compute_iters, memory_iters=1 | 1.17% | 14.79% |
| memory_iters, compute_iters=16 | 1.55% | 13.06% |

Memory-sweep occupancy MAPE is `2.17%`.
DRAM BW utilization MAPE is `1.10%` for the compute sweep
and `1.61%` for the memory sweep.
Eligible-warp occupancy MAPE is `16.67%` for the compute
sweep and `3.06%` for the memory sweep.

## Compute-Iters Sweep

| compute_iters | measured TOPS | model TOPS | peak roofline TOPS | NCU DRAM peak % | model DRAM peak % | NCU eligible % | model eligible % | model time error % | bottleneck |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0 | 0.137 | 0.134 | 0.118 | 87.13 | 85.50 | 0.85 | 1.18 | 2.31 | dram |
| 1 | 0.544 | 0.537 | 0.471 | 86.49 | 85.50 | 1.17 | 1.21 | 1.45 | dram |
| 2 | 0.955 | 0.939 | 0.824 | 86.73 | 85.50 | 1.38 | 1.25 | 1.67 | dram |
| 4 | 1.769 | 1.744 | 1.530 | 86.63 | 85.50 | 1.91 | 1.38 | 1.45 | dram |
| 8 | 3.353 | 3.354 | 2.942 | 85.41 | 85.50 | 2.37 | 1.82 | -0.03 | dram |
| 16 | 6.613 | 6.574 | 5.766 | 85.70 | 85.50 | 3.47 | 3.24 | 0.60 | dram |
| 32 | 13.064 | 13.013 | 11.415 | 85.70 | 85.50 | 6.15 | 7.98 | 0.39 | dram |
| 64 | 25.255 | 25.892 | 22.712 | 82.97 | 85.50 | 22.90 | 23.85 | -2.46 | dram |
| 128 | 38.507 | 38.578 | 45.307 | 63.50 | 63.86 | 50.06 | 46.74 | -0.18 | compute |

## Memory-Iters Sweep

| memory_iters | CTAs | measured TOPS | model TOPS | peak roofline TOPS | NCU DRAM peak % | model DRAM peak % | measured occ % | model occ % | NCU eligible % | model eligible % | model time error % |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 4032 | 6.669 | 6.574 | 5.766 | 86.63 | 85.50 | 85.00 | 84.36 | 3.41 | 3.24 | 1.45 |
| 2 | 2016 | 6.491 | 6.574 | 5.766 | 84.42 | 85.50 | 85.84 | 81.88 | 3.13 | 3.14 | -1.26 |
| 4 | 1008 | 6.531 | 6.574 | 5.766 | 84.85 | 85.50 | 78.43 | 77.33 | 2.79 | 2.97 | -0.65 |
| 8 | 504 | 6.386 | 6.574 | 5.766 | 82.94 | 85.50 | 68.30 | 69.60 | 3.73 | 3.72 | -2.85 |

## Interpretation

The peak roofline baseline mispredicts this kernel because it combines ideal
hardware ceilings with algorithm-level DRAM bytes. In the memory-bound region it
is pessimistic relative to NCU, which reports this kernel as mostly read DRAM
traffic; in the high-compute region it becomes optimistic because scalar FMA
issue efficiency is below the CUDA-core peak. The v6 model adds kernel-specific
scalar-FMA efficiency, NCU-like DRAM traffic, and a warp occupancy estimate
based on CTA count and register pressure. This keeps the parameters physical
while matching the measured sweep more closely.
