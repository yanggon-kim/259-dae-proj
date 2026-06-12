# Variant 4 Results

## Scope

Variant 4 adds scheduler-level producer/consumer cross-warp co-issue on top of
the Variant 3 RFQ model. The queue storage, RFQ readiness, register-pressure
accounting, and kernel geometry are unchanged.

Branch: `variant4-coissue-rfq` in
`/home/yanggon/05_259_project/00_version0/gpgpu-sim_distribution`.

Packaged summary files:

```text
01_gpgpusim_design/results_reference/results_rfq_dual_issue.md
01_gpgpusim_design/results_reference/metrics_rfq_dual_issue.csv
```

## Workload

- Config: H200 RFQ depth 32
- Kernel: generated RFQ stream-FMA V3 PTX
- `n=524288`
- `compute_iters=128`
- `memory_iters=4`
- `CTA count=512`
- `threads/CTA=256`

## Primary V3 vs V4 Result

| Variant | Co-Issue | Cycles | Speedup vs V3 | CTA/SM | Active Occ. | Eligible Occ. | DRAM BW |
|---|---:|---:|---:|---:|---:|---:|---:|
| V3 RFQ | off | 80,183 | 1.000x | 3 | 37.5000% | 11.5139% | 310.704 GB/s |
| V4 RFQ | on | 80,304 | 0.998x | 3 | 37.5000% | 11.5402% | 310.272 GB/s |

DRAM bandwidth uses the H200 normalization:

```text
24 HBM channels * 32 B/channel * 2 DDR ratio * 3125 MHz = 4800 GB/s
effective_dram_bw = 4800 GB/s * average bw_util
```

## Co-Issue Metrics

| Metric | V3 | V4 |
|---|---:|---:|
| `pc_coissue_attempts` | 0 | 67,196,465 |
| `pc_coissue_success` | 0 | 7,456 |
| `pc_coissue_success_rate` | 0.000000 | 0.000111 |
| `single_issue_count` | 14,956,544 | 14,941,632 |
| `dual_issue_count` | 0 | 7,456 |
| `single_issue_rate` | 1.000000 | 0.999501 |
| `dual_issue_success_rate` | 0.000000 | 0.000499 |
| `single_issue_fallback` | 14,956,544 | 14,941,632 |
| `same_warp_dual_issue_fallback` | 0 | 0 |

Failure breakdown for V4:

| Failure Bucket | Count |
|---|---:|
| `pc_coissue_fail_scoreboard` | 2,992,997 |
| `pc_coissue_fail_fu_conflict` | 426,397 |
| `pc_coissue_fail_lsu_lsu_conflict` | 143,305 |
| `pc_coissue_fail_structural_conflict` | 0 |
| `pc_coissue_fail_dispatch_port` | 63,769,615 |
| `pc_coissue_fail_operand_collector` | 0 |
| `pc_coissue_fail_register_bank_conflict` | 0 |
| `pc_coissue_fail_result_bus` | 0 |

## Interpretation

The V4 scheduler is functional: it produces nonzero producer/consumer co-issue
and passes verification. However, this initial stream-FMA workload does not
speed up because successful dual-issue groups are only `0.0499%` of all issue
groups. Most failed attempts are dispatch-port limited, meaning the candidate
pair is visible to the scheduler but cannot both enter the required output
pipeline slots in that cycle.

The measured cycle delta is small: V4 is `0.1509%` slower than V3 on this
workload. The next useful V4 step is the dummy ideal producer-load /
consumer-FMA validation kernel from the plan, because it should create cleaner
producer/consumer ready windows and expose whether the low success rate is a
workload issue or a scheduler policy issue.

## Additional RFQ Tiled GEMM Microkernel

I also tested a small RFQ tiled GEMM-like microkernel to see whether a different
producer/consumer instruction mix exposes stronger V4 co-issue behavior. The
kernel uses 256 threads/CTA, 4 producer warps, 4 consumer warps, and handcoded
RFQ PTX where producers enqueue `{A, B, bias}` tuples and consumers perform FMA
reuse.

Detailed RFQ tiled GEMM microkernel results are summarized in
`report_support/gpgpusim_rfq_dual_issue_microkernel_search.md`.

| Workload | V3 Cycles | V4 Cycles | V4 Speedup | V4 Co-Issue Success | V4 Dual Issue Rate | Result |
|---|---:|---:|---:|---:|---:|---|
| `k64_r16_cta256` | 98,492 | 99,500 | 0.990x | 91,584 | 1.3926% | slower |
| `k64_r4_cta128` | 49,249 | 49,119 | 1.003x | 27,848 | 2.0779% | tiny win |
| `k128_r4_cta128` | 92,485 | 92,448 | 1.000x | 85,836 | 3.2691% | tie |

Conclusion: this GEMM-like RFQ microkernel creates more visible co-issue events
than the initial stream-FMA point, but it still does not materially improve V4.
The best speedup is only `1.003x`; dispatch-window failures remain large.

## Kernel Family Search

I then tested RFQ software-pipelined load/FMA, 2D stencil, separable
convolution, and gather/poly kernels. Detailed results are in
`results_reference/results_rfq_dual_issue_kernel_search.md`; raw metrics are in
`results_reference/metrics_rfq_dual_issue_kernel_search.csv`.

| Family | Best Point | V4 Speedup | Classification | V4 Co-Issue Success | V4 Dual Issue Rate |
|---|---|---:|---|---:|---:|
| `pipe_fma` | A | 0.991x | regression | 77,450 | 2.6718% |
| `stencil2d` | A | 1.004x | tie/noise | 24,030 | 1.4937% |
| `sepconv` | A | 1.023x | strong win | 24,013 | 1.4926% |
| `gather_poly` | A | 1.002x | tie/noise | 13,144 | 0.3841% |

The useful positive result is `sepconv_a`: V4 improves from 51,238 cycles to
50,078 cycles (`1.023x`). `sepconv_b` is also positive at `1.012x`. This makes
separable convolution the best current V4-fit kernel family. The larger
`gather_poly_b` point was stopped after the V3 half ran for over 23 minutes
without final counters, so it is excluded from the summary table.
