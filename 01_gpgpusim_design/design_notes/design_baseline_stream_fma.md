# Design: Baseline Stream-FMA

## Goal

Establish the baseline streaming-FMA workload before adding warp specialization or queue hardware. This point measures the original homogeneous kernel on the H200-like GPGPU-Sim configuration.

## Implementation

- Source: `../../00_kernels/stream_fma_v6/stream_fma_v6.cu`
- Simulator changes: none
- Configuration: `../configs/h200_132sm_mshr512/`
- Main workload family: Stream-FMA with configurable `memory_iters` and `compute_iters`

## Measurement Role

This point is used to understand baseline active occupancy, eligible-warp occupancy, DRAM bandwidth utilization, and sensitivity to `memory_iters`. It is not a queue or dual-issue experiment.

## Result Files

```text
../results_reference/results_v1_baseline.md
../results_reference/metrics_v1_baseline.csv
```
