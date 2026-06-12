# Design Notes

These files describe the design intent and implementation plan for each experiment family.

| File | Scope |
|---|---|
| `design_baseline_stream_fma.md` | V1 baseline Stream-FMA kernel and metrics. |
| `design_warp_specialized_stream_fma.md` | V2 CUDA warp-specialized producer/consumer geometry. |
| `design_rfq.md` | V3 RFQ simulator implementation. |
| `design_smemq.md` | V3a SMEM-based queue simulator implementation. |
| `design_rfq_dual_issue.md` | V4 RFQ plus producer/consumer dual-issue. |
| `design_smemq_dual_issue.md` | V6 SMEM-based queue plus producer/consumer dual-issue. |

Use `../VARIANT_GUIDE.md` for the concise mapping from variants to configs, PTX overrides, and result files.
