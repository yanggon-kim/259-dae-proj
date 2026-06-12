# Agent Result: RFQ Microkernel Search for Dual-Issue Benefit

## 1. Supported Report Sections

- Section 6.2, Design and Contradictory Finding: Dual Issue
- Section 7.2, Methodology: Kernels
- Section 8.1, Evaluation: GPGPU-Sim Dual-Issue
- Section 9, Opportunity-Revealing Finding and Future Work

## 2. Task Scope

The main Stream-FMA workload showed that RFQ plus producer/consumer dual-issue was functional but did not improve performance. To test whether this was a kernel-shape problem, we added RFQ-based microkernels designed to create more overlap between producer global-load instructions and consumer FMA instructions. Each microkernel was run twice: once with RFQ only and once with RFQ plus producer/consumer dual-issue. Speedup is computed as:

```text
speedup = RFQ-only cycles / RFQ-plus-dual-issue cycles
```

## 3. Environment and Common Setup

All runs use the H200-like GPGPU-Sim configuration and RFQ depth `32`. The kernels use `256` threads/CTA, with `4` producer warps and `4` consumer warps. The producer side fills RFQ entries using generated or hand-written `ld.global.rfq.v3.f32` instructions. The consumer side reads the queue using `mov.rfq.v3.f32` and then performs the microkernel-specific compute sequence. Effective DRAM bandwidth is reported as:

```text
effective DRAM BW = average DRAM bw_util * 4800 GB/s
```

The original microkernel source files were generated during the experiment as:

```text
rfq_tiled_gemm_micro.cu
v4_kernel_family_search.cu
```

The current consolidated package keeps the resulting metrics and report text in:

```text
01_gpgpusim_design/results_reference/results_rfq_dual_issue_kernel_search.md
01_gpgpusim_design/results_reference/metrics_rfq_dual_issue_kernel_search.csv
```

The source code for those exploratory microkernels is not part of this
consolidated package. The packaged, directly runnable kernel harness is the
Stream-FMA reference under `01_gpgpusim_design/kernels/stream_fma_reference/`.

## 4. Main Stream-FMA Reference

The reference workload was the same Main Stream-FMA point used elsewhere: `n=524288`, `memory_iters=4`, `compute_iters=128`, CTA count `512`, RFQ depth `32`.

| Workload | RFQ-Only Cycles | RFQ + Dual-Issue Cycles | Speedup | Dual-Issue Successes | Dual-Issue Rate | Result |
|---|---:|---:|---:|---:|---:|---|
| Main Stream-FMA | 80183 | 80304 | 0.9985x | 7456 | 0.000499 | slight slowdown |

This result motivated the microkernel search. The simulator successfully issued producer/consumer pairs, but the success rate was too low to overcome scheduler and dispatch overheads.

## 5. Added Microkernels

### RFQ Tiled GEMM

This kernel is a small GEMM-like producer/consumer workload. Producer warps load `{A, B, bias}` tuples into RFQ. Consumer warps consume the tuple, accumulate one output element, and apply a configurable FMA reuse loop. The intent was to keep consumers compute-heavy while producers continue feeding RFQ entries.

### Pipe-FMA

This kernel keeps the stream-style `{a,b,c}` tuple shape but changes the loop body to create repeated consumer FMA reuse after each RFQ receive. It tests whether a cleaner producer-load plus consumer-FMA rhythm improves dual-issue opportunities.

### Stencil2D

This kernel uses nearby offset accesses, such as `a[base]`, `b[base+1]`, and `c[base+128]`, followed by weighted FMA operations. It approximates a simple stencil pattern where producer warps fetch neighboring values and consumer warps perform a compact compute chain.

### SepConv

This kernel models a separable-convolution-like three-tap computation using neighboring values such as `a[base]`, `b[base+1]`, and `c[base+2]`. It was intended to produce regular memory access and short consumer compute bursts that can align with producer load issue windows.

### Gather-Poly

This kernel uses an index array to gather `{a,b,c}` from indirect locations, then applies a small polynomial/FMA sequence. It tests whether irregular producer memory behavior creates useful ready windows for consumer work.

## 6. Results

| Kernel Family | Point | Parameters | RFQ-Only Cycles | RFQ + Dual-Issue Cycles | Speedup | Dual-Issue Successes | Dual-Issue Rate | Effective DRAM BW | Result |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| RFQ tiled GEMM | k64_r16_cta256 | CTAs=256, K=64, reuse=16 | 98492 | 99500 | 0.990x | 91584 | 0.013926 | 1001.76 GB/s | regression |
| RFQ tiled GEMM | k64_r4_cta128 | CTAs=128, K=64, reuse=4 | 49249 | 49119 | 1.003x | 27848 | 0.020779 | 1014.24 GB/s | tie/noise |
| RFQ tiled GEMM | k128_r4_cta128 | CTAs=128, K=128, reuse=4 | 92485 | 92448 | 1.000x | 85836 | 0.032691 | 1078.08 GB/s | tie/noise |
| pipe_fma | A | CTAs=128, tiles=128, reuse=4 | 92759 | 93595 | 0.991x | 77450 | 0.026718 | 1064.64 GB/s | regression |
| pipe_fma | B | CTAs=128, tiles=128, reuse=16 | 100297 | 101901 | 0.984x | 96032 | 0.014097 | 977.76 GB/s | regression |
| stencil2d | A | CTAs=128, tiles=64, reuse=4 | 52023 | 51798 | 1.004x | 24030 | 0.014937 | 962.08 GB/s | tie/noise |
| stencil2d | B | CTAs=256, tiles=64, reuse=8 | 94695 | 94664 | 1.000x | 123268 | 0.027682 | 1052.72 GB/s | tie/noise |
| sepconv | A | CTAs=128, tiles=64, reuse=4 | 51238 | 50078 | 1.023x | 24013 | 0.014926 | 995.40 GB/s | best win |
| sepconv | B | CTAs=256, tiles=64, reuse=8 | 95136 | 93974 | 1.012x | 123893 | 0.027826 | 1060.80 GB/s | weak win |
| gather_poly | A | CTAs=128, tiles=128, reuse=4 | 171326 | 171024 | 1.002x | 13144 | 0.003841 | 777.04 GB/s | tie/noise |

## 7. Interpretation

The best observed dual-issue point is `sepconv` point A:

```text
speedup = 51238 / 50078 = 1.023x
```

This is stronger than the Main Stream-FMA result, which was `0.9985x`, but the gain is still modest. The microkernel search shows that dual-issue is not automatically profitable just because producer/consumer pairs are issued. Several kernels had many dual-issue successes but still showed no meaningful cycle reduction. For example, RFQ tiled GEMM and pipe-FMA produced nonzero dual-issue events, but dispatch-port pressure, scoreboard constraints, and functional-unit conflicts absorbed most of the potential benefit.

The positive `sepconv` result suggests that dual-issue benefits workloads where producer memory operations and consumer FMA operations become scheduler-ready in nearby cycles and do not repeatedly compete for the same structural resources. The negative GEMM and pipe-FMA results are also useful: they show that simply adding more consumer FMA reuse is not sufficient if the RFQ handoff or dispatch window prevents producer and consumer instructions from pairing often enough.

## 8. Report-Ready Paragraph

Because the Main Stream-FMA workload showed almost no RFQ dual-issue benefit, we added RFQ-based microkernels to search for a kernel shape with stronger producer/consumer overlap. The tested kernels included a tiled GEMM-like RFQ microkernel, pipe-FMA, stencil2D, separable convolution, and gather-polynomial patterns. All kernels used the same H200-like simulator configuration, RFQ depth `32`, `256` threads/CTA, `4` producer warps, and `4` consumer warps. The best point was the separable-convolution-like kernel with `128` CTAs, `64` tiles, and compute reuse `4`: RFQ-only execution took `51238` cycles, while RFQ plus dual-issue took `50078` cycles, giving `51238 / 50078 = 1.023x` speedup. Other kernels were weaker: RFQ tiled GEMM reached only `1.003x`, stencil2D reached `1.004x`, gather-polynomial reached `1.002x`, and pipe-FMA regressed. These results show that dual-issue is functional and can improve a better-matched producer/consumer workload, but the effect is limited by low successful pairing rates and structural scheduling constraints.

## 9. Limitations

- The tested kernels are microbenchmarks, not full application workloads.
- The best speedup found so far is only `1.023x`; this supports dual-issue as a possible opportunity but not as a large standalone gain.
- One planned gather-polynomial point timed out and is excluded from the result table.
- The results isolate RFQ dual-issue behavior and do not include SMEM-based queue residency effects.
