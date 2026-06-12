# Variant 3a SMEMQ Results

Workload: `n=1,032,192`, `compute_iters=16`, `memory_iters=4`, one launch.
CSV: `v3a_smemq_sweep_m4.csv`

| Variant | Depth | Cycles | Speedup vs V3 | Eligible Occ. % | DRAM BW GB/s | CTA/SM | Queue Storage | Queue Max Occ. | Full Stalls | Empty Stalls |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v3_rfq | 32 | 64036 | 1.0000 | 13.0425 | 766.1 | 3 | 12288 | 8 | 0 | 0 |
| v3a_smemq | 4 | 125561 | 0.5100 | 66.4259 | 390.6 | 8 | 6144 | 4 | 40575935 | 320580 |
| v3a_smemq | 8 | 117911 | 0.5431 | 71.5051 | 416.0 | 8 | 12288 | 8 | 0 | 320580 |
| v3a_smemq | 16 | 117911 | 0.5431 | 71.5051 | 416.0 | 8 | 24576 | 8 | 0 | 320580 |
| v3a_smemq | 32 | 81035 | 0.7902 | 25.9639 | 605.3 | 4 | 49152 | 8 | 0 | 234389 |
| v3a_smemq | 64 | 68854 | 0.9300 | 4.0072 | 712.3 | 2 | 98304 | 8 | 0 | 44154 |
| v3a_smemq | 128 | 61550 | 1.0404 | 1.0071 | 796.8 | 1 | 196608 | 8 | 0 | 44505 |
| v3a_smemq | 144 | 61550 | 1.0404 | 1.0071 | 796.8 | 1 | 221184 | 8 | 0 | 44505 |
| v3a_smemq | 151 | 61550 | 1.0404 | 1.0071 | 796.8 | 1 | 231936 | 8 | 0 | 44505 |

Depths 128, 144, and 151 produce the same performance counters because this `memory_iters=4` workload only reaches `smemq_max_occupancy=8`; the larger depths change only modeled shared-memory footprint and keep residency at one CTA/SM.
