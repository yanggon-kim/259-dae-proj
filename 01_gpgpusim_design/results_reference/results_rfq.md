# Variant 3 Results

Variant 3 replaces the Variant 2 shared-memory handoff with simulator-modeled register-file queues. The CUDA host geometry is the same as V2, and the RFQ PTX override is used for simulator runs.

## V2/V3 Full-Run Comparison

| Variant | RFQ depth | Cycles | Speedup | Eligible occ. | Active occ. | DRAM BW | Shared instr. | RFQ full stalls |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| V2 SMEM | N/A | 124680 | 1.0000 | 1.1257% | 90.5134% | 393.4080 GB/s | 6193152 | N/A |
| V3 RFQ | 8 | 90553 | 1.3769 | 37.9123% | 52.0665% | 541.4400 GB/s | 0 | 0 |

## RFQ Depth Sweep

| Depth | Correct | Cycles | Speedup vs V2 | Eligible occ. | DRAM BW | CTA/SM | RFQ full stalls | RFQ head stalls | RFQ max occ. |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | PASS | 130734 | 0.9537 | 1.2105% | 375.1680 GB/s | 7 | 351166039 | 345613870 | 1 |
| 2 | PASS | 126801 | 0.9833 | 28.8477% | 386.8320 GB/s | 7 | 292499394 | 119841745 | 2 |
| 4 | PASS | 92306 | 1.3507 | 43.4810% | 531.3600 GB/s | 6 | 45080294 | 55546998 | 4 |
| 8 | PASS | 90553 | 1.3769 | 37.9123% | 541.4400 GB/s | 5 | 0 | 46353906 | 8 |
| 16 | PASS | 81035 | 1.5386 | 25.9639% | 605.2800 GB/s | 4 | 0 | 39153971 | 8 |
| 32 | PASS | 64036 | 1.9470 | 13.0425% | 766.0800 GB/s | 3 | 0 | 26161289 | 8 |
| 48 | PASS | 68854 | 1.8108 | 4.0072% | 712.3200 GB/s | 2 | 0 | 31009890 | 8 |
| 64 | PASS | 68854 | 1.8108 | 4.0072% | 712.3200 GB/s | 2 | 0 | 31009890 | 8 |
| 96 | PASS | 61550 | 2.0257 | 1.0071% | 796.8000 GB/s | 1 | 0 | 11042693 | 8 |
| 128 | PASS | 61550 | 2.0257 | 1.0071% | 796.8000 GB/s | 1 | 0 | 11042693 | 8 |
| 149 | PASS | 61550 | 2.0257 | 1.0071% | 796.8000 GB/s | 1 | 0 | 11042693 | 8 |

Best passing RFQ depth by simulated cycles: `96`.

Effective DRAM bandwidth uses: 24 HBM channels x 32 B/channel x 2 DDR ratio x 3125 MHz = 4800 GB/s; effective BW = 4800 GB/s x reported CoL_Bus_Util.

Raw logs are under `rfq_depth_sweep_m4/depth_<N>/full_run.log` and `v2_compare_m4/full_run.log`.

## RFQ Depth Sweep With 64 Normal Regs/Thread

This run uses generated ptxinfo with `regs=64` for the V3 kernels. The CUDA host geometry and RFQ PTX handoff are unchanged; only the modeled normal register footprint changes.

| Depth | Correct | Cycles | Speedup vs V2 | Regs/thread | Normal regs/CTA | Total regs/CTA | Eligible occ. | DRAM BW | CTA/SM | RFQ full stalls | RFQ head stalls | RFQ max occ. |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | PASS | 67559 | 1.8455 | 64 | 16384 | 16768 | 1.0602% | 726.2400 GB/s | 3 | 65369478 | 63896851 | 1 |
| 2 | PASS | 66561 | 1.8732 | 64 | 16384 | 17152 | 1.1170% | 736.8000 GB/s | 3 | 57771001 | 61774431 | 2 |
| 4 | PASS | 63649 | 1.9589 | 64 | 16384 | 17920 | 1.3366% | 770.4000 GB/s | 3 | 43962458 | 56851175 | 4 |
| 8 | PASS | 64036 | 1.9470 | 64 | 16384 | 19456 | 13.0425% | 766.0800 GB/s | 3 | 0 | 26161289 | 8 |
| 16 | PASS | 68854 | 1.8108 | 64 | 16384 | 22528 | 4.0072% | 712.3200 GB/s | 2 | 0 | 31009890 | 8 |
| 24 | PASS | 68854 | 1.8108 | 64 | 16384 | 25600 | 4.0072% | 712.3200 GB/s | 2 | 0 | 31009890 | 8 |
| 32 | PASS | 68854 | 1.8108 | 64 | 16384 | 28672 | 4.0072% | 712.3200 GB/s | 2 | 0 | 31009890 | 8 |
| 40 | PASS | 68854 | 1.8108 | 64 | 16384 | 31744 | 4.0072% | 712.3200 GB/s | 2 | 0 | 31009890 | 8 |
| 48 | PASS | 61550 | 2.0257 | 64 | 16384 | 34816 | 1.0071% | 796.8000 GB/s | 1 | 0 | 11042693 | 8 |
| 64 | PASS | 61550 | 2.0257 | 64 | 16384 | 40960 | 1.0071% | 796.8000 GB/s | 1 | 0 | 11042693 | 8 |

Best passing RFQ depth by simulated cycles: `48`.

Effective DRAM bandwidth uses: 24 HBM channels x 32 B/channel x 2 DDR ratio x 3125 MHz = 4800 GB/s; effective BW = 4800 GB/s x reported CoL_Bus_Util.

Raw logs are under `rfq_depth_sweep_m4_regs64/depth_<N>/full_run.log`.
