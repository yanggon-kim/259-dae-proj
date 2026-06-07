# H200 memory-fidelity Tier-1 change — handoff note (2026-06-02)

**Read this before running anything on `config_h200_132sm_mshr512`.** It records a
memory-subsystem investigation, the Tier-1 config edits just applied, and what is
still pending. Companion docs:
- Full investigation + experiments: `../../00_doc/02_variant_data/v3/occupancy_noc_confound_v3.md`
- Per-knob H200 mapping + the dated fidelity entries: `H200_config_fidelity.md` (this dir)

---

## ⚠️ CURRENT STATE (important)

- **Tier-1 edits ARE applied** to `gpgpusim.config` in this directory (see "What changed" below).
- They are **NOT yet validated on this applied config**, and **NOT yet committed.**
  (The change was interrupted right before the validate+commit step.)
- The edits are byte-for-byte the same knob values that were already validated in a
  scratch dir (`gpgpu-sim_distribution/wasp_rfq/faithfix/`), so the expected result is
  known (below) — but a clean re-run on THIS config + a git commit are still TODO.
- **MSHR is intentionally kept at 512.** L2 and the interconnect are intentionally
  unchanged. Only the four Tier-1 memory knobs were touched.

---

## Why this change exists (the problem)

A skeptical follow-up to the Variant-3 results ("how can fewer warps be faster?")
uncovered a confound:

1. The V3 depth-sweep "deeper RFQ queue → faster" was **not** producer/consumer
   decoupling. At **fixed occupancy, queue depth is cycle-neutral**. Depth only
   changed speed by changing *occupancy* (RFQ register pressure → fewer resident CTAs).
2. The real driver was an **inverted occupancy curve**: this config runs the
   streaming-FMA kernel **~2× slower at 8 CTAs/core (64 warps/SM) than at 1** —
   the opposite of normal GPU latency-hiding.
3. Root cause, isolated by experiment: **memory-subsystem back-pressure**, NOT the
   interconnect and NOT DRAM bandwidth:
   - Interconnect was the prime suspect (icnt2mem latency inflated ~185×) but was
     **tested and RULED OUT** — strengthening the NoC (3× clock, 4 VCs, 4× flit,
     4× router throughput) changed cycles < 8 %; the high icnt2mem latency is a
     *symptom* (back-pressure), not the cause.
   - DRAM bandwidth was only **~7–11 % utilised** (i.e. starved), so it is not the limit.
   - The binding resources, found by bumping one category at a time at 8 CTA/core:
     | bumped alone | 8-CTA cycles | gain | 
     |---|---|---|
     | stock | 114,070 | — |
     | L2 (MSHR/miss-q/data-port) | 115,134 | **0 %** (not a bottleneck) |
     | L1 (MSHR/miss-q/data-port) | 97,780 | ~35 % |
     | DRAM queues (partition/sched/return) | 90,422 | ~51 % |
     | all of the above | 67,618 | 100 % |
   - So **two serial throttles: the L1 miss-handling path (MSHR + miss-queue) and
     the DRAM partition/scheduler queues. L2 is innocent.**
4. **Root: Volta-template values, never scaled with the machine.** This config was
   cloned from the Volta `SM7_QV100` template (lineage in `H200_config_fidelity.md`),
   and the **interconnect** (`config_volta_islip.icnt`: `num_vcs=1`, flit 40, NoC
   clock = core clock), the **L1 MSHR** (512), the **L1 miss-queue** (64), and the
   **DRAM partition/scheduler/return queues** (64/64/192) were all inherited
   verbatim. These are GPGPU-Sim's standard defaults — unchanged Volta→Ampere and
   **not scaled per GPU**. So while the H200 fidelity work scaled the SM count, FP
   rates, L2 size, and HBM bandwidth (Volta ~80 SMs / ~0.9 TB/s → H200 132 SMs /
   48 sub-partitions / 4.8 TB/s), **the NoC and the memory-partition queues kept
   their small Volta-template sizes** and are structurally too small to sustain
   HBM3e bandwidth at high occupancy. Tier 1 below resizes the binding ones (the
   L1 miss-queue and the DRAM queues); the NoC and MSHR are discussed separately.

---

## What changed (Tier 1 only — the validated, defensible fix)

All in `gpgpusim.config`, this directory. **Sizing rationale: bandwidth-delay
product** — to sustain ~100 GB/s/sub-partition at ~300-cycle round-trip you need
~300 outstanding 32 B requests per sub-partition, so queues of 64 structurally
cannot sustain HBM3e bandwidth; ~256 can. (NVIDIA does not publish the actual H200
queue/MSHR depths, so this is principled estimation, not datasheet-matching.)

| knob | stock | now (Tier 1) |
|---|---|---|
| `-gpgpu_cache:dl1 …A:512:8,`**`64`**`:0,32` | L1 miss-queue 64 | **256** (`…A:512:8,256:0,32`) |
| `-gpgpu_dram_partition_queues` | 64:64:64:64 | **256:256:256:256** |
| `-gpgpu_frfcfs_dram_sched_queue_size` | 64 | **256** |
| `-gpgpu_dram_return_queue_size` | 192 | **512** |

**Kept unchanged on purpose:**
- **L1 MSHR = 512** (`…A:512:8,…`) — this config is deliberately `mshr512`; real
  Hopper L1 MSHR depth is unpublished, so inflating it would be a guess.
- **L2** (`-gpgpu_cache:dl2 …`) — proven not a bottleneck.
- **Interconnect** (`config_volta_islip.icnt` `num_vcs=1`, flit 40, NoC clock
  1980) and `-gpgpu_clock_domains` — genuinely under-provisioned per-port but
  **not the binding constraint** (ruled out), so left at standard defaults.

---

## Expected validation result (from the identical scratch run)

The same four knob values were run at full scale (`--n 1032192 --iters 16`),
RFQ-off, in `wasp_rfq/faithfix/`:

| config | 8 CTA/core | 1 CTA/core | inversion | 8-CTA DRAM BW | 8-CTA L1D resv-fails |
|---|---|---|---|---|---|
| stock | 114,070 | 55,709 | 2.05× | ~11 % | ~25 M |
| **Tier 1 (this config)** | **86,234** | 54,914 | **1.57×** | ~13 % | 17.2 M |
| (ref) bracket, unfaithful MSHR=2048 | 67,618 | 56,918 | 1.19× | ~16 % | 1.1 M |

So Tier 1 recovers **~60 %** of the lost performance and cuts the inversion
**2.05× → 1.57×**. The residual ~1.57× is held by the faithful L1 MSHR=512 and is
**partly physical** (a 100 %-miss streaming kernel at 64 warps/SM genuinely
thrashes a finite L1 MSHR — real hardware does too). The truly-physical floor is
~1.19× even at MSHR=2048.

---

## TODO for the next agent (pending steps)

1. **Re-validate on THIS applied config** (not the scratch copy):
   ```bash
   cd <repo>/gpgpu-sim_distribution
   export CUDA_INSTALL_PATH=/usr/local/cuda-11.7
   source setup_environment release
   mkdir -p wasp_rfq/tier1val && cd wasp_rfq/tier1val
   cp ../stream_fma_v3_rfq ../stream_fma_v3_rfq.ptx .
   cp <repo>/02_h200_config/config_h200_132sm_mshr512/* .
   echo "-gpgpu_deadlock_detect 1" >> gpgpusim.config
   # Primary check — default (full) occupancy = 8 CTA/core for this kernel:
   env PTX_SIM_USE_PTX_FILE=1 PTX_SIM_KERNELFILE=$PWD/stream_fma_v3_rfq.ptx \
       ./stream_fma_v3_rfq --n 1032192 --warmup 0 --repeats 1
   # Optional low-occupancy point (1 CTA/core) via the standard register knob:
   #   add  -gpgpu_shader_registers 8192  to gpgpusim.config, then re-run the above.
   ```
   Confirm `verification=PASS` and that the default (8 CTA/core) run is ≈ 86k
   cycles — down from ≈ 114k on the stock config — with DRAM bandwidth risen
   (~11 % → ~13 %) and L1D reservation-fails dropped (~25 M → ~17 M). Reducing
   `-gpgpu_shader_registers` to force 1 CTA/core should give ≈ 55k, i.e. the
   high-vs-low-occupancy gap shrinks from ~2.05× to ~1.57×. (Each run ~5 min.)
2. **Commit** this config change (workspace repo, `02_h200_config/`), and update
   `H200_config_fidelity.md` (mark the 2026-06-02 entry "applied" with these numbers).
3. **Regenerate prior cycle metrics:** all V1/V2/V3 runs before this change used the
   stock (under-sized) memory config; their *cycle/occupancy* numbers must be
   re-generated on this config before cross-variant cycle comparisons are trusted.
   Functional/verification results are unaffected.

---

## Standing methodology (applies regardless of this fix)

Because ~1.19–1.57× of the occupancy effect is **physical and survives any config
fix**, cross-variant **cycle** comparisons should be made at **equal resident-warp
count** (equal occupancy). Resident occupancy can be equalized with the standard
GPGPU-Sim register knob **`-gpgpu_shader_registers`** (lower it to reduce CTAs/core),
or per-kernel at compile time with **`-maxrregcount`**. Always report both occupancy
figures (resident `gpu_tot_occupancy` and eligible-warp occupancy) per variant so
the comparison point is explicit.

---

## What is NOT done (deliberately deferred — Tier 2/3)

- **Interconnect** (`num_vcs`, flit, NoC clock): under-provisioned but not binding;
  left at standard defaults. Bump only if you want it to not be *artificially*
  limiting (it currently isn't).
- **L1 MSHR > 512**: would close more of the gap (toward 1.19×) but is an
  unvalidated guess for H200; not done.
- **L2 data-port / `-gpgpu_dram_timing_opt` (HBM3e timings)**: refinements; the
  aggregate 4.8 TB/s is already correct via buswidth×burst×freq×channels.
