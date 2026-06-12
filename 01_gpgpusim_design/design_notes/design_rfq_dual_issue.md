# Variant 4 Design: RFQ + Producer/Consumer Cross-Warp Co-Issue

## Goal

Variant 4 measures the scheduler benefit of producer/consumer cross-warp
co-issue on top of the Variant 3 register-file queue. The RFQ storage,
readiness, and occupancy model stays unchanged; only scheduler issue selection
is extended.

## Scope

- Base branch: `variant4-coissue-rfq` in `gpgpu-sim_distribution/`.
- Queue model: RFQ only, using existing `ld.global.rfq.v3.f32` producer loads
  and `mov.rfq.v3.f32` consumer reads.
- Kernel role model: first `producer_warps_per_cta` warps in each CTA are
  producers; remaining warps are consumers. The value comes from RFQ metadata
  and defaults to half the CTA warps.
- SMEMQ and no-queue co-issue are not part of V4.

## Scheduler Plan

1. Add `-gpgpu_wasp_pc_coissue_enable`.
2. At the start of each scheduler cycle, after normal warp ordering, find a
   ready producer warp and a ready consumer warp from the prioritized list.
3. A candidate is ready only if it passes control-flow PC check, scoreboard
   check, RFQ/SMEMQ readiness checks, and output pipeline availability.
4. Co-issue succeeds only when the two instructions use different execution
   unit classes. This prevents LSU+LSU pair issue and records it as a
   structural/FU conflict.
5. On success, issue both warps in the same scheduler cycle, step both warp
   issue states, and skip the normal single-warp scheduler path for that cycle.
6. If no pair succeeds, fall back to existing same-warp dual issue and single
   issue behavior.

## Metrics

Add final-report counters for:

- `pc_coissue_attempts`
- `pc_coissue_success`
- `pc_coissue_success_rate`
- scoreboard, FU/LSU, dispatch-port, operand-collector, register-bank, and
  result-bus failure buckets
- `same_warp_dual_issue_fallback`
- `single_issue_fallback`
- `total_issue_count`
- `single_issue_count`
- `dual_issue_count`
- `single_issue_rate`
- `dual_issue_success_rate`

The operand collector, register bank, and result bus buckets are exposed for
report consistency. The first V4 implementation classifies scheduler-visible
failures directly and leaves deeper pipeline-stage failures at zero unless a
later phase wires those units into the pair-attempt checker.

## Validation

Build GPGPU-Sim after code edits. Run a short RFQ workload with
`-gpgpu_wasp_pc_coissue_enable 1` and confirm:

- functional verification passes
- `pc_coissue_attempts` and `pc_coissue_success` are printed
- `dual_issue_count` includes successful producer/consumer pairs
- Variant 3 behavior is preserved when the new flag is disabled
