# V6 SMEMQ Co-Issue Results

V6 enables producer/consumer cross-warp co-issue with SMEMQ storage and scheduler-visible queue readiness. DRAM bandwidth uses `4800 GB/s * average bw_util`.

Stream-FMA V4 RFQ depth-32 baseline: `80304` cycles.
Sepconv_a V4 RFQ depth-32 baseline: `50078` cycles.

## Stream-FMA

| Run | Depth | Cycles | Speedup vs V4 | Speedup vs V3a Control | CTA/SM | Active Occ % | Eligible Occ % | DRAM BW GB/s | Co-Issue Success | Dual Issue Rate | Empty Stalls | Head-Not-Ready Stalls | Verify |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| stream_smemq_no_coissue_depth32 | 32 | 63937 | 1.256 | 1.000 | 4 | 50.00 | 18.328 | 389.66 | 0 | 0.000000 | 72448 | 17109452 | PASS |
| stream_v6_smemq_depth8 | 8 | 63343 | 1.268 | 1.009 | 8 | 100.00 | 18.034 | 393.31 | 6258 | 0.000419 | 72448 | 17284706 | PASS |
| stream_v6_smemq_depth16 | 16 | 63343 | 1.268 | 1.009 | 8 | 100.00 | 18.034 | 393.31 | 6258 | 0.000419 | 72448 | 17284706 | PASS |
| stream_v6_smemq_depth32 | 32 | 63343 | 1.268 | 1.009 | 4 | 50.00 | 18.034 | 393.31 | 6258 | 0.000419 | 72448 | 17284706 | PASS |
| stream_v6_smemq_depth64 | 64 | 79644 | 1.008 | 0.803 | 2 | 25.00 | 4.628 | 312.82 | 3513 | 0.000235 | 29559 | 7064527 | PASS |
| stream_v6_smemq_depth128 | 128 | 130365 | 0.616 | 0.490 | 1 | 12.50 | 1.503 | 191.09 | 0 | 0.000000 | 23658 | 3812128 | PASS |

## Sepconv_A

| Run | Depth | Cycles | Speedup vs V4 | Speedup vs V3a Control | CTA/SM | Active Occ % | Eligible Occ % | DRAM BW GB/s | Co-Issue Success | Dual Issue Rate | Empty Stalls | Head-Not-Ready Stalls | Verify |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| sepconv_a_v6_smemq_depth32 | 32 | 50078 | 1.000 |  | 4 | 50.00 | 4.931 | 995.40 | 24013 | 0.014926 | 34432 | 7593340 | PASS |

## Interpretation

Best stream-FMA V6 depths are `8`, `16`, and `32`; all finish at `63343` cycles and `1.268x` speedup vs the V4 RFQ depth-32 stream baseline.

Depths `64` and `128` lose the benefit because SMEMQ storage reduces CTA/SM to `2` and `1`, lowering eligible occupancy and DRAM bandwidth.

`sepconv_a` ties V4 exactly. It validates that SMEMQ + co-issue executes the same queue/co-issue pattern, but this `128`-CTA workload fits in one wave on the `132`-SM H200 config, so the higher V6 CTA/SM limit does not create a residency advantage.

Read V6 benefit through both cycle count and the issue counters: `pc_coissue_success` and `dual_issue_success_rate` show whether scheduler-visible SMEMQ readiness creates actual dual-issue events, while `smemq_empty_stalls` and `smemq_head_not_ready_stalls` explain queue-readiness delays.
