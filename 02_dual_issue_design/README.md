# Producer-Consumer Dual-Issue Reproduction Guide

This directory packages the GPGPU-Sim producer-consumer dual-issue experiment
for reproducing only two implementations:

- `WS_BASE`: H200-like baseline hardware with the warp-specialized V2
  stream-FMA kernel.
- `DUAL_ISSUE_PC`: producer-consumer dual-issue/RFQ hardware with the V3
  stream-FMA kernel.

The headline result is that the dual-issue mechanism is active, but this
full-scale kernel does not expose enough legal same-cycle producer/consumer
instructions to beat `WS_BASE`.

## Contents

```text
configs/ws_base/                         # H200-like config, pair issue disabled
configs/dual_issue_pc/                   # H200-like config with dual-issue/RFQ knobs
gpgpusim_dual_issue_sources/             # Modified GPGPU-Sim source replacements
kernel_reference/                        # CUDA harness with --variant 2 and --variant 3
report/                                  # Report-ready markdown and block diagram
results_reference/                       # Reference logs, CSV, and summary
tools/extract_dual_issue_metrics.py      # Log parser
```

## Environment

Expected software stack:

```text
GPGPU-Sim v4.2.0-compatible tree
CUDA 11.7
NVCC target: sm_70
```

The kernel Makefile intentionally omits `-lineinfo`; GPGPU-Sim v4.2.0 rejects
the CUDA 11.7 `.loc ... inlined_at` PTX emitted by that option.

## Prepare GPGPU-Sim

From a clean GPGPU-Sim v4.2.0-compatible checkout, copy the packaged modified
source files into place:

```bash
cd <gpgpu-sim>
cp <259-dae-proj>/02_dual_issue_design/gpgpusim_dual_issue_sources/src/gpgpu-sim/gpu-sim.cc src/gpgpu-sim/gpu-sim.cc
cp <259-dae-proj>/02_dual_issue_design/gpgpusim_dual_issue_sources/src/gpgpu-sim/shader.cc src/gpgpu-sim/shader.cc
cp <259-dae-proj>/02_dual_issue_design/gpgpusim_dual_issue_sources/src/gpgpu-sim/shader.h src/gpgpu-sim/shader.h
export CUDA_INSTALL_PATH=/usr/local/cuda-11.7
source setup_environment release
make -j$(nproc)
```

This package uses source replacements rather than a patch because the local
dual-issue simulator history starts from an already-modified snapshot and does
not record a clean upstream parent.

## Build Kernel Harness

```bash
cd <259-dae-proj>/02_dual_issue_design/kernel_reference
make NVCC=/usr/local/cuda-11.7/bin/nvcc ARCH=sm_70
```

The same binary runs both implementations:

- `--variant 2` selects `WS_BASE`.
- `--variant 3` selects `DUAL_ISSUE_PC`.

## Run `WS_BASE`

```bash
cd <259-dae-proj>/02_dual_issue_design
mkdir -p runs/ws_base
cp -a configs/ws_base/. runs/ws_base/
cp kernel_reference/stream_fma_v6_v2_m4_c128_cta512 runs/ws_base/

cd runs/ws_base
./stream_fma_v6_v2_m4_c128_cta512 \
  --variant 2 \
  --n 524288 \
  --iters 128 \
  --memory-iters 4 \
  --warmup 0 \
  --repeats 1 \
  > full_run.log 2>&1
```

## Run `DUAL_ISSUE_PC`

```bash
cd <259-dae-proj>/02_dual_issue_design
mkdir -p runs/dual_issue_pc
cp -a configs/dual_issue_pc/. runs/dual_issue_pc/
cp kernel_reference/stream_fma_v6_v2_m4_c128_cta512 runs/dual_issue_pc/

cd runs/dual_issue_pc
./stream_fma_v6_v2_m4_c128_cta512 \
  --variant 3 \
  --n 524288 \
  --iters 128 \
  --memory-iters 4 \
  --warmup 0 \
  --repeats 1 \
  > full_run.log 2>&1
```

Correctness in both logs should report:

```text
verification=PASS
```

## Extract Metrics

For packaged reference logs:

```bash
cd <259-dae-proj>/02_dual_issue_design
tools/extract_dual_issue_metrics.py \
  results_reference/ws_base_full_run.log \
  results_reference/dual_issue_pc_full_run.log
```

For newly generated logs:

```bash
tools/extract_dual_issue_metrics.py \
  runs/ws_base/full_run.log \
  runs/dual_issue_pc/full_run.log
```

## Reference Result

Full workload:

```text
--n 524288 --iters 128 --memory-iters 4 --warmup 0 --repeats 1
```

Speedup denominator:

```text
speedup = cycles(WS_BASE) / cycles(DUAL_ISSUE_PC)
```

| Implementation | Cycles | Instructions | Pair successes | Correct | Speedup vs `WS_BASE` |
|---|---:|---:|---:|---|---:|
| `WS_BASE` | 54,110 | 281,149,440 | 0 | PASS | 1.000x |
| `DUAL_ISSUE_PC` | 59,165 | 281,149,440 | 188,694 | PASS | 0.915x |

The dual-issue mechanism issues real producer-consumer pairs, but most pairing
attempts fail due to scoreboard timing or missing producer windows. The
full-scale kernel therefore runs slower than the warp-specialized baseline.

## Scout Check

The smaller scout case is useful to confirm the mechanism can help in an
underfilled run:

```text
--n 16384 --iters 128 --memory-iters 4 --warmup 0 --repeats 1
```

| Implementation | Cycles | Pair successes | Correct | Speedup vs `WS_BASE` |
|---|---:|---:|---|---:|
| `WS_BASE` | 28,201 | 0 | PASS | 1.000x |
| `DUAL_ISSUE_PC` | 28,101 | 2,620 | PASS | 1.004x |

This confirms functionality, but the positive effect is too small to carry to
the full 512-CTA run.
