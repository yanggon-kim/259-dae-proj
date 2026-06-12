# V6 Shared-Memory Queue Sweep

Workload: stream-FMA, `n=524288`, `memory_iters=4`, `compute_iters=128`, one launch.

Baseline: V4 RFQ depth 32 = `80304` cycles, `3` CTA/SM, `11.5402%` eligible occupancy, `310.27 GB/s` effective DRAM bandwidth.

| Variant | Queue | Depth | Cycles | Speedup vs V4 RFQ d32 | CTA/SM | Active Occ % | Eligible Occ % | DRAM BW GB/s | Co-Issue Success | Dual Issue Rate | Queue Storage/CTA |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| V3a control | SMEMQ | 32 | 63937 | 1.256 | 4 | 50.0 | 18.3275 | 389.66 | 0 | 0.000000 | 49152 B |
| V6 | SMEMQ | 8 | 63343 | 1.268 | 8 | 100.0 | 18.0341 | 393.31 | 6258 | 0.000419 | 12288 B |
| V6 | SMEMQ | 16 | 63343 | 1.268 | 8 | 100.0 | 18.0341 | 393.31 | 6258 | 0.000419 | 24576 B |
| V6 | SMEMQ | 32 | 63343 | 1.268 | 4 | 50.0 | 18.0341 | 393.31 | 6258 | 0.000419 | 49152 B |
| V6 | SMEMQ | 64 | 79644 | 1.008 | 2 | 25.0 | 4.6280 | 312.82 | 3513 | 0.000235 | 98304 B |
| V6 | SMEMQ | 128 | 130365 | 0.616 | 1 | 12.5 | 1.5027 | 191.09 | 0 | 0.000000 | 196608 B |

## Findings

Depths 8, 16, and 32 are the useful V6 region for this workload. They all produce `63343` cycles, about `1.268x` faster than V4 RFQ depth 32 and `1.009x` faster than the SMEMQ no-coissue control at depth 32.

Depth 64 and 128 show the expected SMEMQ shared-memory capacity tradeoff. Queue storage grows to `98304 B/CTA` and `196608 B/CTA`, reducing CTA/SM to `2` and `1`; eligible occupancy and DRAM bandwidth fall with the reduced residency.

The V6 co-issue benefit is real but small for stream-FMA: at depths 8-32, `pc_coissue_success = 6258` and `dual_issue_success_rate = 0.000419`. Most attempted pairs still fail on dispatch-port pressure, so the speedup is mainly from SMEMQ avoiding RFQ register-pressure residency loss, with a small added co-issue contribution.
