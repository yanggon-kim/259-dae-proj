# V4 Kernel Family Search Results

Purpose: try RFQ workloads that should expose producer LSU and consumer SP/FMA readiness in the same scheduler windows.

All runs use the H200 RFQ depth-32 configuration. V3 is RFQ without cross-warp co-issue; V4 enables the same RFQ kernel with producer/consumer co-issue. DRAM bandwidth is `4800 GB/s * average bw_util`.

| Family | Point | CTAs | Tiles | Reuse | V3 Cycles | V4 Cycles | Speedup | Class | V4 Co-Issue | V4 Dual Issue Rate | V4 Dispatch Fail | V4 Eligible Occ % | V4 DRAM BW GB/s | Verify |
|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---|
| pipe_fma | A | 128 | 128 | 4 | 92759 | 93595 | 0.991 | regression | 77450 | 0.026718 | 18955722 | 5.766 | 1064.64 | PASS |
| pipe_fma | B | 128 | 128 | 16 | 100297 | 101901 | 0.984 | regression | 96032 | 0.014097 | 11719258 | 4.424 | 977.76 | PASS |
| stencil2d | A | 128 | 64 | 4 | 52023 | 51798 | 1.004 | tie/noise | 24030 | 0.014937 | 8914629 | 4.888 | 962.08 | PASS |
| stencil2d | B | 256 | 64 | 8 | 94695 | 94664 | 1.000 | tie/noise | 123268 | 0.027682 | 107592709 | 16.869 | 1052.72 | PASS |
| sepconv | A | 128 | 64 | 4 | 51238 | 50078 | 1.023 | strong win | 24013 | 0.014926 | 9085975 | 4.931 | 995.40 | PASS |
| sepconv | B | 256 | 64 | 8 | 95136 | 93974 | 1.012 | weak win | 123893 | 0.027826 | 107747542 | 16.932 | 1060.80 | PASS |
| gather_poly | A | 128 | 128 | 4 | 171326 | 171024 | 1.002 | tie/noise | 13144 | 0.003841 | 27392 | 0.263 | 777.04 | PASS |

## Interpretation

Best observed point: `sepconv_a` with V4 speedup `1.023x`.

Use the failure buckets to separate real co-issue benefit from scheduler visibility. A useful V4-fit kernel should raise both `pc_coissue_success` and `dual_issue_success_rate` while reducing cycle count. If co-issue success rises but cycles do not improve, the added ready-window opportunities are still being absorbed by dispatch, scoreboard, or FU conflicts.

Raw logs are stored under `runs/h200_v4_kernel_family_search/`.

## Timed-Out Points

The following planned points were stopped before final simulator counters and are excluded from the table:

- `gather_poly_b_v3`: partial log retained as `full_run.timeout.log`.
