# 259 DAE Project

This repository packages the CUDA kernels, GPGPU-Sim configuration, simulator-source overlay, generated PTX, and reference results used for the warp-specialization experiments. The main reproducibility package is:

```text
01_gpgpusim_design/
```

That directory replaces the previous split `01_SMEMQ_design/` and `02_dual_issue_design/` packages. It contains one consolidated GPGPU-Sim implementation for the baseline, warp-specialized, RFQ, SMEM-based queue, RFQ dual-issue, and SMEM-based queue plus dual-issue studies.

## Repository Layout

```text
00_kernels/
  stream_fma_v6/          # Original Stream-FMA CUDA benchmark and analysis files
  pc_pair_issue/          # Early producer/consumer pair-issue microbenchmarks

01_gpgpusim_design/
  gpgpusim_overlay/       # Actual modified GPGPU-Sim source files
  configs/                # H200-like and variant-specific simulator configs
  kernels/                # Reference CUDA host/kernel harnesses
  ptx_overrides/          # Generated RFQ and SMEM-based queue PTX/metadata
  results_reference/      # Reference logs, CSVs, and result summaries
  report_support/         # Report-ready design and result markdown
  tools/                  # Overlay install and run helpers

config_h200_132sm_mshr512/
  # Standalone H200-like configuration reference
```

## Quick Start

Prepare a compatible GPGPU-Sim checkout:

```bash
export CUDA_INSTALL_PATH=/usr/local/cuda-11.7
cd <gpgpu-sim_distribution>
source setup_environment release
```

Install the project simulator overlay:

```bash
cd <this-repo>
01_gpgpusim_design/tools/install_gpgpusim_overlay.sh <gpgpu-sim_distribution>
cd <gpgpu-sim_distribution>
make -j$(nproc)
```

Build and run a reference Stream-FMA case:

```bash
cd <this-repo>/01_gpgpusim_design
tools/run_stream_fma_reference.sh rfq
tools/run_stream_fma_reference.sh smemq
tools/run_stream_fma_reference.sh rfq-dual
tools/run_stream_fma_reference.sh smemq-dual
```

Expected correctness line:

```text
verification=PASS
```

## Documentation Map

- `01_gpgpusim_design/README.md`: full overlay install, build, and reproduction workflow.
- `01_gpgpusim_design/VARIANT_GUIDE.md`: mapping from the original V1/V2/V3/V3a/V4/V6 names to the packaged baseline, warp-specialized, RFQ, SMEM-based queue, and dual-issue cases.
- `01_gpgpusim_design/design_notes/`: implementation intent and source-edit summaries for each design point.
- `01_gpgpusim_design/results_reference/`: reference logs, CSV metrics, and result summaries.
- `01_gpgpusim_design/report_support/`: report-ready text and figures for the RFQ, SMEM-based queue, and dual-issue implementation.

Use `README.md` as the top-level entry point; detailed simulator instructions live under `01_gpgpusim_design/`.
