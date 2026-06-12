# Design: Warp-Specialized Stream-FMA

## Goal

Introduce producer/consumer warp specialization while keeping the workload and launch geometry close to the baseline Stream-FMA kernel. This isolates the cost and benefit of splitting work across warp roles without adding RFQ, SMEM-based queue hardware, or dual-issue.

## Implementation

- Source: `../kernels/stream_fma_reference/stream_fma_v6_v2_m4_c128_cta512.cu`
- Simulator changes: none
- Threads/CTA: `256`
- Warps/CTA: `8`
- Producer warps: `0-3`
- Consumer warps: `4-7`
- Handoff mechanism: CUDA shared memory and CTA-level synchronization

## Workload

```text
n=524288
memory_iters=4
compute_iters=128
CTA count=512
```

## Measurement Role

This point provides the CUDA warp-specialized baseline used before replacing the software/shared-memory handoff with RFQ or SMEM-based queue pseudo-instructions.

## Result Files

```text
../results_reference/results_v2_warp_specialized.md
../results_reference/metrics_v2_warp_specialized.csv
```
