# Agent Result: GPGPU-Sim Dual-Issue and SMEM-Based Queue

## 1. Supported Report Sections

- Section 6.1, Design and Contradictory Finding: SMEM-Based Queue
- Section 6.2, Design and Contradictory Finding: Dual Issue
- Section 7.1, Methodology: GPGPU-Sim Configuration
- Section 7.2, Methodology: Kernels
- Section 8.1, Evaluation: GPGPU-Sim

## 2. Task Scope

This note reports the GPGPU-Sim results for producer/consumer dual-issue and for the combined SMEM-based queue plus dual-issue design. The terminology is report-facing: RFQ means register-file queue, SMEM-based queue means queue payload storage charged to shared memory with scheduler-visible readiness metadata, and dual-issue means issuing one producer warp instruction and one consumer warp instruction in the same cycle when the scheduler and structural checks allow it.

## 3. Environment and Workload

Main Stream-FMA Result workload: `n=524288`, `memory_iters=4`, `compute_iters=128`, CTA count `512`.

All runs use the H200-like GPGPU-Sim configuration with `132` SMs and the same generated producer/consumer stream-FMA PTX shape. Each CTA has `256` threads, `8` warps/CTA, `4` producer warps, and `4` consumer warps. DRAM bandwidth is reported as:

```text
effective DRAM BW = average DRAM bw_util * 4800 GB/s
```

Speedup is reported as:

```text
speedup = baseline cycles / measured cycles
```

A speedup above `1.0x` is faster than the baseline. For the dual-issue-only paragraph, the baseline is RFQ without dual issue at depth `32`, which took `80183` cycles.

## 4. RFQ Dual-Issue-Only Result

This comparison isolates the effect of dual-issue only. Both runs use RFQ storage, depth `32`, the same queue payload, the same CTA geometry, and the same H200-like configuration.

| Configuration | Queue | Depth | Cycles | Speedup vs RFQ Without Dual-Issue | CTA/SM | Active Occ. | Eligible Occ. | DRAM BW | Dual-Issue Successes | Dual-Issue Rate | Verify |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| RFQ without dual-issue | RFQ | 32 | 80183 | 1.0000x | 3 | 37.5% | 11.5139% | 310.704 GB/s | 0 | 0.000000 | PASS |
| RFQ with dual-issue | RFQ | 32 | 80304 | 0.9985x | 3 | 37.5% | 11.5402% | 310.272 GB/s | 7456 | 0.000499 | PASS |

The RFQ dual-issue result shows that the scheduler feature is active, because `7456` producer/consumer pairs were issued together. However, this did not improve the Stream-FMA workload: `80183 / 80304 = 0.9985x`, meaning the RFQ dual-issue run was about `0.15%` slower than RFQ without dual-issue. CTA residency stayed fixed at `3` CTA/SM and active occupancy stayed at `37.5%`, so this experiment isolates the dual-issue mechanism without a queue-storage change. The eligible occupancy increased only slightly, from `11.5139%` to `11.5402%`, and effective DRAM bandwidth was effectively unchanged. The measured dual-issue rate was only `0.0499%`, so structural hazards, scoreboard constraints, and dispatch-port pressure prevented most candidate producer/consumer pairs from actually issuing together.

## 5. SMEM-Based Queue Plus Dual-Issue Result

This comparison uses RFQ without dual-issue at depth `32` as the speedup baseline (`80183` cycles), then reports SMEM-based queue plus dual-issue over queue depths `8`, `16`, and `32`.

| Configuration | Queue | Depth | Cycles | Speedup vs RFQ Without Dual-Issue | Speedup vs RFQ With Dual-Issue | CTA/SM | Active Occ. | Eligible Occ. | DRAM BW | Dual-Issue Successes | Dual-Issue Rate | Queue Storage/CTA | Verify |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| SMEM-based queue with dual-issue | SMEM-based | 8 | 63343 | 1.2658x | 1.2678x | 8 | 100.0% | 18.0341% | 393.312 GB/s | 6258 | 0.000419 | 12288 B | PASS |
| SMEM-based queue with dual-issue | SMEM-based | 16 | 63343 | 1.2658x | 1.2678x | 8 | 100.0% | 18.0341% | 393.312 GB/s | 6258 | 0.000419 | 24576 B | PASS |
| SMEM-based queue with dual-issue | SMEM-based | 32 | 63343 | 1.2658x | 1.2678x | 4 | 50.0% | 18.0341% | 393.312 GB/s | 6258 | 0.000419 | 49152 B | PASS |

Depths `8`, `16`, and `32` are tied at `63343` cycles. Relative to RFQ without dual-issue, the combined SMEM-based queue plus dual-issue design gives `80183 / 63343 = 1.2658x` speedup. Relative to RFQ with dual-issue, it gives `80304 / 63343 = 1.2678x` speedup. The residency effect is the dominant difference: at depth `32`, RFQ is limited to `3` CTA/SM, while the SMEM-based queue reaches `4` CTA/SM; at depths `8` and `16`, the SMEM-based queue reaches `8` CTA/SM. Active occupancy therefore rises from the RFQ value of `37.5%` to `50.0%` at depth `32` and `100.0%` at depths `8` and `16`. Eligible occupancy also rises from about `11.5%` in the RFQ runs to `18.0341%`, and effective DRAM bandwidth rises from about `310 GB/s` to `393.312 GB/s`.

For reference, the SMEM-based queue without dual-issue at depth `32` took `63937` cycles. Therefore, adding dual-issue on top of the SMEM-based queue improved performance by only `63937 / 63343 = 1.0094x`. The best combined points are depths `8`, `16`, and `32`, all tied at `63343` cycles. Compared with RFQ with dual-issue at depth `32`, this is `1.2678x` faster. Compared with the SMEM-based queue without dual-issue at depth `32`, it is only `1.0094x` faster. This means most of the observed improvement comes from the SMEM-based queue improving CTA residency, active occupancy, eligible occupancy, and DRAM bandwidth, while the incremental contribution of dual-issue is small for this Stream-FMA workload.

## 6. Report-Ready Paragraph

For the Main Stream-FMA workload (`n=524288`, `memory_iters=4`, `compute_iters=128`, `512` CTAs), producer/consumer dual-issue alone did not improve the RFQ design. RFQ without dual-issue completed in `80183` cycles, while RFQ with dual-issue completed in `80304` cycles, giving `80183 / 80304 = 0.9985x` speedup. Although the simulator reported `7456` successful dual-issue events, the dual-issue rate was only `0.0499%`, CTA residency remained `3` CTA/SM, active occupancy remained `37.5%`, eligible occupancy changed only from `11.5139%` to `11.5402%`, and effective DRAM bandwidth stayed near `310 GB/s`. In contrast, combining the SMEM-based queue with dual-issue completed in `63343` cycles at depths `8`, `16`, and `32`, giving `80183 / 63343 = 1.2658x` speedup over RFQ without dual-issue and `80304 / 63343 = 1.2678x` over RFQ with dual-issue. The depth-32 SMEM-based queue point reached `4` CTA/SM and `50.0%` active occupancy, while depths `8` and `16` reached `8` CTA/SM and `100.0%` active occupancy. Eligible occupancy increased to `18.0341%`, and effective DRAM bandwidth increased to `393.312 GB/s`. However, the SMEM-based queue without dual-issue already took `63937` cycles, so the incremental dual-issue benefit on top of SMEM-based storage was only `63937 / 63343 = 1.0094x`. These results suggest that, for this Stream-FMA workload, SMEM-based queue storage is the main source of performance improvement because it recovers CTA residency and exposes more eligible warps; dual-issue is functional but contributes only a small additional gain.

## 7. Limitations

- These are GPGPU-Sim timing-model results, not real H200 hardware measurements.
- The Stream-FMA instruction mix produced a very low dual-issue success rate, so this workload does not strongly stress the best-case dual-issue path.
- The speedups use simulator cycle counts from the same workload and configuration; they should not be mixed with runs that use a different CTA count, memory iteration count, compute iteration count, or GPU configuration.
