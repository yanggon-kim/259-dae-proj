# stream_fma_v6 Model

This directory contains a mechanistic performance model for
`stream_fma_v6`. The model inputs are:

- `n`
- `compute_iters`
- `memory_iters`
- `threads_per_cta`
- hardware parameters such as FP32 TOPS, DRAM bandwidth, SM count, register
  file size, and warp/CTA limits

The main equations are:

```text
ctas = n / (threads_per_cta * memory_iters)
ops = n * (6 * compute_iters + 2)
algorithm_bytes = n * 16
operational_intensity = ops / algorithm_bytes
predicted_time = max(compute_time, dram_time)
TOPS = ops / predicted_time / 1e12
```

`stream_fma_v6_model.py` provides the reusable model. `validate_v6_model.py`
compares predictions with the existing NCU sweeps and can refresh those sweeps:

```bash
python3 01_kernel/stream_fma_v6/01_analysis/02_model/stream_fma_v6_model.py \
  --n 1032192 --iters 16 --memory-iters 4

python3 01_kernel/stream_fma_v6/01_analysis/02_model/validate_v6_model.py

python3 01_kernel/stream_fma_v6/01_analysis/02_model/validate_v6_model.py \
  --refresh-ncu
```

The validation output includes CSV tables, plots, and
`model_validation_summary.md`.
