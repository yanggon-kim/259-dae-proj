# Dual-Issue Kernel Reference

This folder contains the CUDA harness for the `WS_BASE` and `DUAL_ISSUE_PC`
comparison. The same binary supports both implementations:

- `--variant 2`: warp-specialized V2 kernel for `WS_BASE`.
- `--variant 3`: RFQ-shaped V3 kernel for `DUAL_ISSUE_PC`.

## Default Workload

| Parameter | Value |
|---|---:|
| `n` | 524,288 |
| `memory_iters` | 4 |
| `compute_iters` | 128 |
| CTA count | 512 |
| Threads/CTA | 256 |
| Producer warps/CTA | 4 |
| Consumer warps/CTA | 4 |
| V2 shared storage | 3 KiB |
| V3 m4 queue depth | 8 chunks |
| V3 m4 shared storage | 12 KiB |

## Build

```bash
make NVCC=/usr/local/cuda-11.7/bin/nvcc ARCH=sm_70
```

The Makefile intentionally omits `-lineinfo` for GPGPU-Sim v4.2.0 parser
compatibility.

## Local Commands

```bash
make run-v2
make run-v3
./stream_fma_v6_v2_m4_c128_cta512 --variant 3 --n 524288 --iters 128 --memory-iters 4
```

For full reproduction, run the binary from a directory containing one of the
packaged GPGPU-Sim configs under `../configs/`.
