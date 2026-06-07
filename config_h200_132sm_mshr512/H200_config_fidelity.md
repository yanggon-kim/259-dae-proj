# config_h200_132sm_mshr512 — Real-H200 Fidelity Notes

**Purpose of this directory:** a *solid, real-H200-faithful* GPGPU-Sim configuration. This file is the authority on how each knob maps to the real NVIDIA H200 (Hopper GH100), and records fidelity upgrades. For dual-issue *performance* tuning (a deliberately non-faithful, separate axis), see `../../01_doc/02_config_recommendations/dual_issue_config_recommendations.md`.

Simulator: GPGPU-Sim v4.2.0 @ commit a4ce3fe. Compile target stays `sm_70` (PTX-intercept compatibility; see limitations).

## Per-knob → real-H200 mapping

| Aspect | Config value | Real H200 | Faithful? |
|---|---|---|:--:|
| SM count | `gpgpu_n_clusters 132` | 132 SMs | ✓ |
| Core clock | 1980 MHz | ~1980 MHz boost | ✓ |
| DRAM clock | 3125 MHz ×2 = 6.25 Gbps | HBM3e ~6.0–6.5 Gbps/pin | ✓ |
| HBM bandwidth | 24 ch × 32 B × 2 × 3125 MHz = 4800 GB/s | 4.8 TB/s | ✓ |
| L2 cache | 256×128×32 × 48 subpart ≈ 50.3 MB | ~50 MB | ✓ (flat, not split — see limits) |
| L1/shared | 256 KB unified/SM, 228 KB smem | Hopper 256 KB/SM, 228 KB smem | ✓ |
| L1 MSHR | 512 | (tuned to expose DRAM latency) | ✓ |
| FP32 units | `num_sp_units 4` (1/subpart × 32-wide) | 128/SM = 32/subpart | ✓ |
| INT32 units | `num_int_units 4` | 64/SM = 16/subpart | ✓ |
| FP64 units | `num_dp_units 4` | 64/SM = 16/subpart | ✓ |
| SFU | `num_sfu_units 4` | 16/SM = 4/subpart | ✓ |
| Schedulers | `num_sched_per_core 4`, `sub_core_model 1` | 4 sub-partitions/SM | ✓ |
| **FP32 initiation interval** | **`initiation_fp 1,1,1,1,4`** | full-rate (32 FP32/subpart) → II=1 | ✓ (fixed 2026-05-24) |
| **INT32 initiation interval** | `initiation_int 2,2,2,2,8,4` | half-rate (16 INT32/subpart) → II=2 | ✓ |
| **FP64 initiation interval** | **`initiation_dp 2,2,2,2,130`** | ½-rate datacenter (16 FP64/subpart) → II=2 | ✓ (fixed 2026-05-24) |
| **SFU initiation interval** | `initiation_sfu 8` | quarter-rate (4 SFU/subpart) → II=8 | ✓ |
| Register banks | `num_reg_banks 16` | banked RF (modeling choice) | ~ |
| RF port throughput | `reg_file_port_throughput 2` | operand collector; see note | ~ (see limits) |
| Issue width | `max_insn_issue_per_warp 1` | Hopper single-issues per warp by default | ✓ |
| Compute capability | 7.0 (Volta) | 9.0 (Hopper) | ✗ deliberate (see limits) |

## Initiation-interval semantics

`-ptx_opcode_initiation_*` = minimum core cycles between successive warp-instructions entering a function unit = `warp_size / lanes_per_subpartition`. Arrays are `[ADD, MAX/MIN, MUL, MAD/FMA, DIV(,SHFL)]`.

- FP32: 32 lanes/subpart → 32/32 = **II=1**.
- INT32: 16 lanes/subpart → 32/16 = **II=2**.
- FP64: 16 lanes/subpart → 32/16 = **II=2**.
- SFU: 4 lanes/subpart → 32/4 = **II=8**.

## 2026-05-24 fidelity upgrade

The config was originally cloned from GPGPU-Sim's **Volta** `SM7_QV100` template, which carried Volta's function-unit throughputs. Volta ≠ Hopper on two pipes:

| Unit | Volta V100 (/SM) | Hopper H100/H200 (/SM) | old II (Volta) | new II (Hopper) |
|---|---|---|---:|---:|
| FP32 | 64 | **128** | 2 | **1** |
| FP64 | 32 | **64** (datacenter ½-rate) | 4 | **2** |
| INT32 | 64 | 64 | 2 | 2 (unchanged) |
| SFU | 16 | 16 | 8 | 8 (unchanged) |

Changes applied:
- `-ptx_opcode_initiation_fp 2,2,2,2,4` → `1,1,1,1,4`
- `-ptx_opcode_initiation_dp 4,4,4,4,130` → `2,2,2,2,130`

Cross-check: GPGPU-Sim's shipped **Ampere `SM86_RTX3070`** config uses `initiation_fp 1` (Ampere/Hopper also have 128 FP32/SM), confirming II=1 is the correct full-rate value; Volta's `SM7_*` configs use `initiation_fp 2` (64 FP32/SM). Sources: NVIDIA Hopper Architecture whitepaper (128 FP32 / 64 INT32 / 64 FP64 / 16 SFU per SM; FP64 = ½ FP32 rate, 34 vs 67 TFLOPS), GPGPU-Sim `configs/tested-cfgs/SM86_RTX3070`.

## 2026-06-02 investigation — high-occupancy memory congestion (interconnect RULED OUT)

A "fewer warps = faster" inversion was found at high occupancy (full writeup:
`../../00_doc/02_variant_data/v3/occupancy_noc_confound_v3.md`): the streaming-FMA
kernel runs ~2× slower at 8 resident CTAs/core (64 warps/SM) than at 1, with
effective load latency exploding 500 → 13,637 cycles. The latency surfaces in the
`avg_icnt2mem_latency` counter (70 → ~12,900), so the **interconnect was the prime
suspect**. It was tested and **ruled out:**

| 8 CTA/core, RFQ-off | cycles | icnt2mem latency |
|---|---|---|
| stock NoC (`num_vcs=1`, 1980 MHz, flit 40) | 114,070 | 12,892 |
| NoC clock 3× + `num_vcs` 4 | 105,278 | 10,520 |
| above + flit 128 + `internal_speedup` 4 + 512 buffers | **105,278** | **10,520** |

The strong router upgrade was **byte-identical** to the weak one, and the clock/VC
changes moved cycles < 8 %. The packets are not network-throughput-limited — they
are **back-pressured**, waiting for the memory partition to accept them.

**Real bottleneck — the memory subsystem, not the NoC.** At 8 CTA/core:
- **L1D `Reservation_fails` ≈ 170k–207k per core** (vs 4,096 accesses) — the L1
  miss-handling resources (MSHRs / miss queue) are saturated and loads retry.
- **L1 & L2 miss rate = 1.000** (pure streaming, zero reuse) — every access
  traverses the full L1→L2→DRAM path.
- **DRAM only ~11 % utilised** with `gpu_stall_dramfull` ~800k — DRAM is *starved*
  because the L1-miss / memory-partition queues can't feed it; the icnt2mem
  latency is a *symptom* of this downstream backpressure.

So the high-occupancy slowdown comes from the **L1-miss / memory-partition
acceptance path** (MSHRs, L2 ports, `-gpgpu_dram_partition_queues`,
`-gpgpu_frfcfs_dram_sched_queue_size`) saturating under a 132-SM, 100 %-miss
streaming load. It is partly physical (finite MSHRs/queues do throttle), but the
DRAM starvation at 11 % indicates the memory-partition provisioning is too small
for this scaled-up config.

**Throttle isolated (one category bumped at a time, 8 CTA/core):**

| bumped | cycles | gain | note |
|---|---|---|---|
| stock | 114,070 | — | |
| L2 (MSHR/miss-q/data-port) | 115,134 | 0 % | not a bottleneck |
| L1 (MSHR/miss-q/data-port) | 97,780 | ~35 % | |
| DRAM queues (partition/sched/return) | 90,422 | ~51 % | |
| all of the above | 67,618 | 100 % | |

So the binding resources are **(1) the L1 miss-handling path (MSHR + miss-queue)
and (2) the DRAM partition/scheduler queues** (`-gpgpu_dram_partition_queues`
64→256, `-gpgpu_frfcfs_dram_sched_queue_size` 64→256,
`-gpgpu_dram_return_queue_size` 192→512) — in series. **L2 is not a bottleneck.**

**Faithful-only fix tested** — deepen DRAM queues (partition 64→256, sched 64→256,
return 192→512) + L1 miss-queue (64→256), **keep L1 MSHR=512** and L2 stock:

| config | 8 CTA/core | 1 CTA/core | inversion |
|---|---|---|---|
| stock | 114,070 | 55,709 | 2.05× |
| **faithful-only (MSHR=512)** | **86,234** | 54,914 | **1.57×** |
| bracket (unfaithful, MSHR=2048) | 67,618 | 56,918 | 1.19× |

The defensible change recovers **~60 %** of the lost performance and cuts the
inversion 2.05×→1.57×. The remainder is the L1 MSHR=512 limit (resv-fails stay
17.2 M vs the bracket's 1.1 M at MSHR=2048) — **partly real**: a 100 %-miss
streaming kernel at 64 warps/SM genuinely thrashes a finite L1 MSHR, as real
hardware would. So the inversion is **~60 % fidelity bug, ~40 % intrinsic**.

**Recommended config change (defensible, not yet applied to this baseline):**
`-gpgpu_dram_partition_queues 256:256:256:256`,
`-gpgpu_frfcfs_dram_sched_queue_size 256`, `-gpgpu_dram_return_queue_size 512`,
and L1 miss-queue 64→256 (`-gpgpu_cache:dl1 …A:512:8,256:0,32`). Keep MSHR=512
(faithful). Applying it changes the shared baseline and invalidates prior
cycle/occupancy metrics, so it is held pending decision. Until then, **compare
variants at equal resident-warp count** to avoid the confound.

## Known fidelity limitations (not fixable by config knobs)

1. **Compute capability 7.0, not 9.0** — kept for `sm_70` PTX interception (cuSPARSE / Hopper ISA aren't PTX-interceptable through GPGPU-Sim). No Hopper-specific ISA (TMA, `wgmma`, async-copy, distributed shared memory). Deliberate, documented project policy.
2. **No operand-reuse / register-file cache** — real Hopper has register caching in the collector units that raises effective RF bandwidth; GPGPU-Sim models a plain operand collector. `reg_file_port_throughput 2` is the validated value and is adequate for single issue (it only bottlenecks forced dual issue). This is *why* the knob isn't the lever for Hopper fidelity.
3. **Flat 50 MB L2, not split-partition** — real Hopper L2 is two ~30 MB halves with asymmetric near/far latency; the sim models a uniform L2.
4. **No hardware prefetcher / async pipelines** — contributes to the residual sim-vs-real bw_util gap on large matrices (documented in `02_doc/`).

## Relationship to other configs

- This (`05_gpgpusim_h200`) is now the **live real-H200-faithful** config.
- `04_gpgpusim_h200/00_config_b200/config_h200_132sm_mshr512/` is the **historical frozen ADOPTED baseline** and still carries the old Volta FU init intervals (FP32 II=2, FP64 II=4). The two now intentionally diverge; 04 is preserved for reproducing the original `02_doc/` result set.

## Effect on prior results

The SpMV bw_util numbers in `02_doc/` (e.g. cant 20.5%, 132,019 cyc) were measured **before** this upgrade (FP32 II=2). Since SpMV is memory-bound, the FP32 II 2→1 change is expected to leave bw_util essentially unchanged. Verification re-run of cant under the upgraded config confirms this:

| Metric | Pre-upgrade (FP32 II=2) | Post-upgrade (FP32 II=1) | Δ |
|---|---:|---:|---:|
| gpu_sim_cycle | 132,019 | **131,648** | −0.28 % |
| bw_util | 0.2050 | **0.2056** | +0.06 pp |
| correctness | PASSED | PASSED | — |

→ Essentially unchanged, as expected for a memory-bound kernel. The FP32 throughput fix is correct for fidelity (and matters for compute-bound workloads) without disturbing the SpMV memory-bound conclusion. (`/tmp/h200_fp1_cant/cant.log`.)

---
*Generated 2026-05-24. This directory = real-H200 reference. Dual-issue tuning lives in `01_doc/02_config_recommendations/dual_issue_config_recommendations.md`.*
