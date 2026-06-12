# GPGPU-Sim Source Overlay

This directory contains the actual modified GPGPU-Sim source files used for the project. Paths match the destination paths inside a GPGPU-Sim source tree.

## Included Implementation Areas

```text
libcuda/cuda_runtime_api.cc
src/abstract_hardware_model.h
src/cuda-sim/cuda-sim.cc
src/cuda-sim/ptx.l
src/cuda-sim/ptx.y
src/cuda-sim/ptx_ir.cc
src/cuda-sim/ptx_loader.cc
src/gpgpu-sim/gpu-sim.cc
src/gpgpu-sim/CMakeLists.txt
src/gpgpu-sim/shader.cc
src/gpgpu-sim/shader.h
src/gpgpu-sim/wasp-rfq.cc
src/gpgpu-sim/wasp-rfq.h
src/gpgpu-sim/wasp-smem-queue.cc
src/gpgpu-sim/wasp-smem-queue.h
```

## What This Adds

- RFQ pseudo-instructions and timing/functional queue model.
- SMEM-based queue pseudo-instructions and timing/functional queue model.
- Queue metadata loading for producer/consumer geometry.
- RFQ register-pressure CTA residency accounting.
- SMEM-based shared-memory CTA residency accounting.
- Scheduler-visible queue-head readiness checks.
- Eligible-warp occupancy reporting.
- Producer/consumer dual-issue path and failure counters.

## Install

```bash
cd <this-repo>
01_gpgpusim_design/tools/install_gpgpusim_overlay.sh <gpgpu-sim_distribution>
```

Then rebuild GPGPU-Sim:

```bash
cd <gpgpu-sim_distribution>
export CUDA_INSTALL_PATH=/usr/local/cuda-11.7
source setup_environment release
make -j$(nproc)
```
