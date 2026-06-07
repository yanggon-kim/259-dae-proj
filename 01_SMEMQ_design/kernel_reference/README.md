# RFQ/SMEMQ Kernel Reference Workload

This folder contains the portable CUDA reference workload used to define the
RFQ-based and SMEMQ-based stream-FMA experiment shape. It intentionally
contains the warp-specialized CUDA kernel only. It does not contain queue
pseudo-instruction PTX.

## Workload Parameters

| Parameter | Value |
|---|---:|
| Kernel shape | V2 warp-specialized stream-FMA |
| `memory_iters` | 4 |
| `compute_iters` | 128 |
| Queue depth for RFQ/SMEMQ experiments | 32 |
| CTA count | 512 |
| Threads/CTA | 256 |
| Elements/CTA | 1024 |
| `n` | 524,288 |
| Producer warps/CTA | 4 |
| Consumer warps/CTA | 4 |

The CUDA file launches the warp-specialized shared-memory handoff kernel:

```text
stream_fma_v6_v2_m4_kernel
```

The RFQ-based and SMEMQ-based measurements use generated PTX based on this
same producer and consumer geometry. In those measurements, the
producer/consumer handoff is replaced by RFQ or SMEMQ pseudo instructions. The
queue depth value, `32`, is recorded here for reproducibility, but this CUDA
reference does not allocate either queue.

## Files

```text
stream_fma_v6_v2_m4_c128_cta512.cu  # CUDA host + V2 warp-specialized kernels
Makefile                            # Minimal local build/run targets
README.md                           # This description
```

## Build

For local CUDA runs:

```bash
make
```

For GPGPU-Sim CUDA 11.7 / SM70 runs:

```bash
make NVCC=/usr/local/cuda-11.7/bin/nvcc ARCH=sm_70
```

## Run

The defaults already match the experiment:

```bash
./stream_fma_v6_v2_m4_c128_cta512
```

Equivalent explicit command:

```bash
./stream_fma_v6_v2_m4_c128_cta512 \
  --n 524288 \
  --iters 128 \
  --memory-iters 4 \
  --warmup 0 \
  --repeats 1
```

Expected geometry line:

```text
CTAs=512  threads/CTA=256  elements/CTA=1024  chunks/CTA=8
```

Correctness is reported by:

```text
verification=PASS
```

## Notes for Other Agents

- Use this file as the baseline CUDA geometry when generating or comparing
  RFQ-based and SMEMQ-based PTX.
- Do not treat this source as either queue implementation. It is the
  warp-specialized source geometry that the generated queue PTX replaces.
- Keep `n = CTA_count * 256 * memory_iters`. For this experiment,
  `512 * 256 * 4 = 524288`.
- Use `../configs/` for the H200-like GPGPU-Sim configurations and
  `../ptx_overrides/` for queue PTX, metadata, and ptxinfo overrides.
