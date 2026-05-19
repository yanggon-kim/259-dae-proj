# Eligible-Warp Model Extension

This note records the planned and implemented extension to the `stream_fma_v6`
performance model. The original model predicts TOPS with a bottleneck equation:

```text
compute_time = ops / effective_fpu_tops
dram_time = model_dram_bytes / effective_dram_bw
predicted_time = max(compute_time, dram_time)
```

This distinguishes compute-bound and memory-bound behavior, but it does not
separate hidden memory latency from exposed memory stalls. To make that more
explicit, the extension adds a model-level overlap decomposition:

```text
overlapped_time = min(compute_time, dram_time)
exposed_memory_time = max(0, dram_time - compute_time)
exposed_compute_time = max(0, compute_time - dram_time)
compute_exposure_fraction = compute_time / predicted_time
```

If `exposed_memory_time` is positive, the kernel is memory dominated. If
`exposed_compute_time` is positive, the kernel is compute dominated. This is a
roofline-level interpretation, not a cycle-accurate memory-latency timeline.

## Eligible Warp Occupancy

Active warp occupancy counts resident active warps, including stalled warps.
Eligible warp occupancy counts active warps that are ready to issue an
instruction at the scheduler. The extension predicts eligible warps by applying
a calibrated readiness fraction to the existing active-warp estimate:

```text
memory_batch_relief = max(0, log2(memory_iters) - 2)

eligible_ready_fraction =
    clamp(
        eligible_floor
        + eligible_compute_gain * compute_exposure_fraction^eligible_compute_alpha
        + eligible_memory_batch_gain * memory_batch_relief,
        0,
        1
    )

eligible_warps_per_sm = active_warps_per_sm * eligible_ready_fraction
eligible_warps_per_scheduler_cycle = eligible_warps_per_sm / smsps_per_sm
eligible_warps_pct = eligible_warps_per_sm / max_warps_per_sm * 100
```

The calibrated constants are:

```text
smsps_per_sm = 4
eligible_floor = 0.014
eligible_compute_gain = 0.54
eligible_compute_alpha = 1.75
eligible_memory_batch_gain = 0.015
```

The `eligible_floor` captures a small amount of scheduler readiness even in very
memory-bound cases. The compute term increases readiness as the kernel gains
more independent FMA work. The memory batching term handles the observed
`memory_iters=8` case, where per-thread batching changes the scheduler behavior
even though the kernel remains memory dominated.

## Validation Targets

The model is validated against existing NCU metrics already collected in the
sweep CSVs:

```text
smsp__warps_eligible.avg.per_cycle_active
smsp__warps_eligible.avg.pct_of_peak_sustained_active
```

The validation script writes measured-vs-model eligible warp columns into:

```text
01_analysis/02_model/model_validation_compute_iters.csv
01_analysis/02_model/model_validation_memory_iters.csv
```

and generates:

```text
01_analysis/02_model/eligible_warps_vs_ncu_compute_iters.png
01_analysis/02_model/eligible_warps_vs_ncu_memory_iters.png
```

These plots compare NCU eligible warp occupancy with the model prediction. They
should be interpreted as a calibrated scheduler-readiness estimate, not as a
replacement for detailed warp-level simulation.
