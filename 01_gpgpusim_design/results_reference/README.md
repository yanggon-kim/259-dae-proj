# Reference Results

This directory stores compact reference outputs for reproducing and checking the GPGPU-Sim experiments.

## Main Stream-FMA Workload

```text
n=524288
memory_iters=4
compute_iters=128
CTA count=512
threads/CTA=256
producer warps/CTA=4
consumer warps/CTA=4
```

## Key Files

| File | Contents |
|---|---|
| `results_v1_baseline.md`, `metrics_v1_baseline.csv` | Baseline homogeneous Stream-FMA. |
| `results_v2_warp_specialized.md`, `metrics_v2_warp_specialized.csv` | Warp-specialized CUDA/shared-memory handoff. |
| `results_rfq.md` | RFQ result summary. |
| `results_smemq.md` | SMEM-based queue result summary. |
| `results_rfq_dual_issue.md`, `metrics_rfq_dual_issue.csv` | RFQ plus producer/consumer dual-issue. |
| `results_smemq_dual_issue.md`, `metrics_smemq_dual_issue.csv` | SMEM-based queue plus dual-issue. |
| `smemq_dual_issue_depth_sweep.md` | SMEM-based queue plus dual-issue depth sweep. |
| `results_rfq_dual_issue_kernel_search.md` | RFQ microkernel search for better dual-issue workloads. |
| `rfq_smemq_reference/` | Packaged RFQ/SMEM-based queue reference logs. |
| `dual_issue_reference/` | Packaged dual-issue reference logs. |

Use `../tools/extract_rfq_smemq_metrics.py` and `../tools/extract_dual_issue_metrics.py` to parse logs.
