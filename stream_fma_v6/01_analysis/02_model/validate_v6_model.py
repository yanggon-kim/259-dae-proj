#!/usr/bin/env python3
"""Validate stream_fma_v6 model predictions against NCU measurements."""

from __future__ import annotations

import argparse
import csv
import math
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt

from stream_fma_v6_model import (
    HardwareParams,
    KernelParams,
    KernelResources,
    make_peak_roofline_hardware,
    predict,
)


MODEL_DIR = Path(__file__).resolve().parent
ANALYSIS_DIR = MODEL_DIR.parent
V6_DIR = ANALYSIS_DIR.parent
ROOT = V6_DIR.parent.parent
RUN_V6_ANALYSIS = ANALYSIS_DIR / "00_roofline" / "scripts" / "run_v6_analysis.py"
COMPUTE_CSV = ANALYSIS_DIR / "01_sweep_test" / "compute_iters_sweep_memory_iters_1.csv"
MEMORY_CSV = ANALYSIS_DIR / "01_sweep_test" / "memory_iters_sweep_compute_iters_16.csv"


def run(cmd: list[str]) -> None:
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open() as f:
        return list(csv.DictReader(f))


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def mape(actual: list[float], predicted: list[float]) -> float:
    errors = []
    for a, p in zip(actual, predicted):
        if a != 0.0 and not math.isnan(a) and not math.isnan(p):
            errors.append(abs((p - a) / a))
    return sum(errors) / len(errors) * 100.0 if errors else math.nan


def pct_error(actual: float, predicted: float) -> float:
    return (predicted - actual) / actual * 100.0 if actual else math.nan


def measured_values(row: dict[str, str]) -> dict[str, float]:
    duration_s = float(row["duration_ns"]) * 1.0e-9
    flops = float(row["flops"])
    dram_bytes = float(row["dram_bytes"])
    algorithm_bytes = float(row["algorithm_bytes"])
    return {
        "time_s": duration_s,
        "tops": flops / duration_s / 1.0e12,
        "oi_algorithm": flops / algorithm_bytes,
        "oi_ncu": flops / dram_bytes if dram_bytes else math.nan,
        "dram_gbps": float(row["dram_GBps"]),
        "dram_peak_pct": float(row["dram_peak_pct"]),
        "occupancy_pct": float(row["achieved_occupancy_pct"]),
        "active_warps": float(row["active_warps_per_sm_cycle"]),
        "eligible_warps_pct": float(row["eligible_warps_pct"]),
        "eligible_warps_per_scheduler_cycle": float(
            row["eligible_warps_per_scheduler_cycle"]
        ),
        "long_scoreboard_pct": float(row["long_scoreboard_pct"]),
    }


def prediction_row(row: dict[str, str], hw: HardwareParams) -> dict[str, object]:
    memory_iters = int(row["memory_iters"])
    kernel = KernelParams(
        n=int(row["n"]),
        compute_iters=int(row["compute_iters"]),
        memory_iters=memory_iters,
        threads_per_cta=int(row["threads_per_cta"]),
    )
    resources = KernelResources.for_memory_iters(memory_iters)
    model = predict(kernel, hw, resources)
    roofline = predict(kernel, make_peak_roofline_hardware(hw), resources)
    measured = measured_values(row)

    return {
        "n": kernel.n,
        "compute_iters": kernel.compute_iters,
        "memory_iters": kernel.memory_iters,
        "ctas": model.mapping.ctas,
        "elements_per_cta": model.mapping.elements_per_cta,
        "registers_per_thread": resources.registers_per_thread,
        "measured_time_ns": measured["time_s"] * 1.0e9,
        "model_time_ns": model.predicted_time_s * 1.0e9,
        "roofline_time_ns": roofline.predicted_time_s * 1.0e9,
        "model_time_error_pct": pct_error(measured["time_s"], model.predicted_time_s),
        "roofline_time_error_pct": pct_error(measured["time_s"], roofline.predicted_time_s),
        "measured_tops": measured["tops"],
        "model_tops": model.predicted_tops,
        "roofline_tops": roofline.predicted_tops,
        "model_tops_error_pct": pct_error(measured["tops"], model.predicted_tops),
        "roofline_tops_error_pct": pct_error(measured["tops"], roofline.predicted_tops),
        "oi_algorithm": model.operational_intensity_algorithm,
        "model_oi_dram": model.operational_intensity_dram,
        "measured_oi_ncu": measured["oi_ncu"],
        "measured_dram_gbps": measured["dram_gbps"],
        "model_predicted_dram_gbps": model.predicted_dram_gbps,
        "model_effective_dram_gbps": model.effective_dram_bw_gbps,
        "measured_dram_utilization_pct": measured["dram_peak_pct"],
        "model_dram_utilization_pct": model.predicted_dram_utilization_pct,
        "model_effective_dram_utilization_pct": model.effective_dram_utilization_pct,
        "model_dram_gbps_error_pct": pct_error(measured["dram_gbps"], model.predicted_dram_gbps),
        "model_dram_utilization_error_pct": pct_error(
            measured["dram_peak_pct"], model.predicted_dram_utilization_pct
        ),
        "measured_active_warps_per_sm": measured["active_warps"],
        "model_active_warps_per_sm": model.occupancy.estimated_active_warps_per_sm,
        "measured_occupancy_pct": measured["occupancy_pct"],
        "model_occupancy_pct": model.occupancy.estimated_achieved_occupancy * 100.0,
        "theoretical_occupancy_pct": model.occupancy.theoretical_occupancy * 100.0,
        "measured_eligible_warps_pct": measured["eligible_warps_pct"],
        "model_eligible_warps_pct": model.scheduler.estimated_eligible_warps_pct,
        "measured_eligible_warps_per_scheduler_cycle": measured[
            "eligible_warps_per_scheduler_cycle"
        ],
        "model_eligible_warps_per_scheduler_cycle": model.scheduler.estimated_eligible_warps_per_scheduler_cycle,
        "model_eligible_ready_fraction": model.scheduler.eligible_ready_fraction,
        "model_compute_exposure_fraction": model.scheduler.compute_exposure_fraction,
        "model_overlapped_time_ns": model.scheduler.overlapped_time_s * 1.0e9,
        "model_exposed_memory_time_ns": model.scheduler.exposed_memory_time_s * 1.0e9,
        "model_exposed_compute_time_ns": model.scheduler.exposed_compute_time_s * 1.0e9,
        "resident_ctas_per_sm": model.occupancy.resident_ctas_per_sm,
        "resident_warps_per_sm": model.occupancy.resident_warps_per_sm,
        "register_limit_ctas_per_sm": model.occupancy.register_limit_ctas_per_sm,
        "bottleneck": model.bottleneck,
        "measured_long_scoreboard_pct": measured["long_scoreboard_pct"],
    }


def validate_rows(rows: list[dict[str, str]], hw: HardwareParams) -> list[dict[str, object]]:
    return [prediction_row(row, hw) for row in rows]


def plot_compute(rows: list[dict[str, object]]) -> None:
    x = [int(r["compute_iters"]) for r in rows]
    fig, ax = plt.subplots(figsize=(8.5, 5.2), dpi=160)
    ax.plot(x, [float(r["measured_tops"]) for r in rows], marker="o", label="NCU measured")
    ax.plot(x, [float(r["model_tops"]) for r in rows], marker="s", label="v6 model")
    ax.plot(
        x,
        [float(r["roofline_tops"]) for r in rows],
        marker="^",
        linestyle="--",
        label="peak roofline",
    )
    ax.set_xscale("symlog", linthresh=1)
    ax.set_xlabel("compute_iters")
    ax.set_ylabel("TOPS")
    ax.set_title("stream_fma_v6 m1: model vs NCU")
    ax.grid(True, linestyle=":", linewidth=0.7)
    ax.legend()
    fig.tight_layout()
    fig.savefig(MODEL_DIR / "model_vs_ncu_compute_iters.png")
    plt.close(fig)


def plot_memory(rows: list[dict[str, object]]) -> None:
    x = [int(r["memory_iters"]) for r in rows]
    fig, ax = plt.subplots(figsize=(8.5, 5.2), dpi=160)
    ax.plot(x, [float(r["measured_tops"]) for r in rows], marker="o", label="NCU measured")
    ax.plot(x, [float(r["model_tops"]) for r in rows], marker="s", label="v6 model")
    ax.plot(
        x,
        [float(r["roofline_tops"]) for r in rows],
        marker="^",
        linestyle="--",
        label="peak roofline",
    )
    ax.set_xlabel("memory_iters")
    ax.set_ylabel("TOPS")
    ax.set_title("stream_fma_v6 compute_iters=16: model vs NCU")
    ax.grid(True, linestyle=":", linewidth=0.7)
    ax.legend()
    fig.tight_layout()
    fig.savefig(MODEL_DIR / "model_vs_ncu_memory_iters.png")
    plt.close(fig)


def plot_occupancy(rows: list[dict[str, object]]) -> None:
    x = [int(r["memory_iters"]) for r in rows]
    fig, ax = plt.subplots(figsize=(8.5, 5.2), dpi=160)
    ax.plot(
        x,
        [float(r["measured_occupancy_pct"]) for r in rows],
        marker="o",
        label="NCU achieved occupancy",
    )
    ax.plot(
        x,
        [float(r["model_occupancy_pct"]) for r in rows],
        marker="s",
        label="model achieved estimate",
    )
    ax.plot(
        x,
        [float(r["theoretical_occupancy_pct"]) for r in rows],
        marker="^",
        linestyle="--",
        label="theoretical occupancy",
    )
    ax.set_xlabel("memory_iters")
    ax.set_ylabel("occupancy (%)")
    ax.set_title("stream_fma_v6 occupancy model")
    ax.grid(True, linestyle=":", linewidth=0.7)
    ax.legend()
    fig.tight_layout()
    fig.savefig(MODEL_DIR / "occupancy_vs_ncu_memory_iters.png")
    plt.close(fig)


def plot_eligible_memory(rows: list[dict[str, object]]) -> None:
    x = [int(r["memory_iters"]) for r in rows]
    fig, ax = plt.subplots(figsize=(8.5, 5.2), dpi=160)
    ax.plot(
        x,
        [float(r["measured_eligible_warps_pct"]) for r in rows],
        marker="o",
        label="NCU eligible warps",
    )
    ax.plot(
        x,
        [float(r["model_eligible_warps_pct"]) for r in rows],
        marker="s",
        label="model eligible estimate",
    )
    ax.set_xlabel("memory_iters")
    ax.set_ylabel("eligible warps (% of peak active warps)")
    ax.set_title("stream_fma_v6 eligible-warp model, compute_iters=16")
    ax.grid(True, linestyle=":", linewidth=0.7)
    ax.legend()
    fig.tight_layout()
    fig.savefig(MODEL_DIR / "eligible_warps_vs_ncu_memory_iters.png")
    plt.close(fig)


def plot_eligible_compute(rows: list[dict[str, object]]) -> None:
    x = [int(r["compute_iters"]) for r in rows]
    fig, ax = plt.subplots(figsize=(8.5, 5.2), dpi=160)
    ax.plot(
        x,
        [float(r["measured_eligible_warps_pct"]) for r in rows],
        marker="o",
        label="NCU eligible warps",
    )
    ax.plot(
        x,
        [float(r["model_eligible_warps_pct"]) for r in rows],
        marker="s",
        label="model eligible estimate",
    )
    ax.set_xscale("symlog", linthresh=1)
    ax.set_xlabel("compute_iters")
    ax.set_ylabel("eligible warps (% of peak active warps)")
    ax.set_title("stream_fma_v6 eligible-warp model, memory_iters=1")
    ax.grid(True, linestyle=":", linewidth=0.7)
    ax.legend()
    fig.tight_layout()
    fig.savefig(MODEL_DIR / "eligible_warps_vs_ncu_compute_iters.png")
    plt.close(fig)


def plot_dram_utilization(
    compute_rows: list[dict[str, object]], memory_rows: list[dict[str, object]]
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6), dpi=160)

    compute_x = [int(r["compute_iters"]) for r in compute_rows]
    axes[0].plot(
        compute_x,
        [float(r["measured_dram_utilization_pct"]) for r in compute_rows],
        marker="o",
        label="NCU DRAM peak %",
    )
    axes[0].plot(
        compute_x,
        [float(r["model_dram_utilization_pct"]) for r in compute_rows],
        marker="s",
        label="model DRAM peak %",
    )
    axes[0].set_xscale("symlog", linthresh=1)
    axes[0].set_xlabel("compute_iters")
    axes[0].set_ylabel("DRAM BW utilization (%)")
    axes[0].set_title("compute sweep, memory_iters=1")
    axes[0].grid(True, linestyle=":", linewidth=0.7)
    axes[0].legend(fontsize=8)

    memory_x = [int(r["memory_iters"]) for r in memory_rows]
    axes[1].plot(
        memory_x,
        [float(r["measured_dram_utilization_pct"]) for r in memory_rows],
        marker="o",
        label="NCU DRAM peak %",
    )
    axes[1].plot(
        memory_x,
        [float(r["model_dram_utilization_pct"]) for r in memory_rows],
        marker="s",
        label="model DRAM peak %",
    )
    axes[1].set_xlabel("memory_iters")
    axes[1].set_ylabel("DRAM BW utilization (%)")
    axes[1].set_title("memory sweep, compute_iters=16")
    axes[1].grid(True, linestyle=":", linewidth=0.7)
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(MODEL_DIR / "dram_bw_utilization_vs_ncu.png")
    plt.close(fig)


def write_summary(
    compute_rows: list[dict[str, object]], memory_rows: list[dict[str, object]]
) -> None:
    compute_time_mape = mape(
        [float(r["measured_time_ns"]) for r in compute_rows],
        [float(r["model_time_ns"]) for r in compute_rows],
    )
    compute_roof_time_mape = mape(
        [float(r["measured_time_ns"]) for r in compute_rows],
        [float(r["roofline_time_ns"]) for r in compute_rows],
    )
    memory_time_mape = mape(
        [float(r["measured_time_ns"]) for r in memory_rows],
        [float(r["model_time_ns"]) for r in memory_rows],
    )
    memory_roof_time_mape = mape(
        [float(r["measured_time_ns"]) for r in memory_rows],
        [float(r["roofline_time_ns"]) for r in memory_rows],
    )
    occupancy_mape = mape(
        [float(r["measured_occupancy_pct"]) for r in memory_rows],
        [float(r["model_occupancy_pct"]) for r in memory_rows],
    )
    compute_dram_util_mape = mape(
        [float(r["measured_dram_utilization_pct"]) for r in compute_rows],
        [float(r["model_dram_utilization_pct"]) for r in compute_rows],
    )
    memory_dram_util_mape = mape(
        [float(r["measured_dram_utilization_pct"]) for r in memory_rows],
        [float(r["model_dram_utilization_pct"]) for r in memory_rows],
    )
    compute_eligible_mape = mape(
        [float(r["measured_eligible_warps_pct"]) for r in compute_rows],
        [float(r["model_eligible_warps_pct"]) for r in compute_rows],
    )
    memory_eligible_mape = mape(
        [float(r["measured_eligible_warps_pct"]) for r in memory_rows],
        [float(r["model_eligible_warps_pct"]) for r in memory_rows],
    )

    compute_table = "\n".join(
        "| {ci} | {meas:.3f} | {model:.3f} | {roof:.3f} | {dram:.2f} | {mdram:.2f} | {elig:.2f} | {melig:.2f} | {terr:.2f} | {bneck} |".format(
            ci=int(r["compute_iters"]),
            meas=float(r["measured_tops"]),
            model=float(r["model_tops"]),
            roof=float(r["roofline_tops"]),
            dram=float(r["measured_dram_utilization_pct"]),
            mdram=float(r["model_dram_utilization_pct"]),
            elig=float(r["measured_eligible_warps_pct"]),
            melig=float(r["model_eligible_warps_pct"]),
            terr=float(r["model_time_error_pct"]),
            bneck=r["bottleneck"],
        )
        for r in compute_rows
    )
    memory_table = "\n".join(
        "| {mi} | {ctas} | {meas:.3f} | {model:.3f} | {roof:.3f} | {dram:.2f} | {mdram:.2f} | {occ:.2f} | {mocc:.2f} | {elig:.2f} | {melig:.2f} | {terr:.2f} |".format(
            mi=int(r["memory_iters"]),
            ctas=int(r["ctas"]),
            meas=float(r["measured_tops"]),
            model=float(r["model_tops"]),
            roof=float(r["roofline_tops"]),
            dram=float(r["measured_dram_utilization_pct"]),
            mdram=float(r["model_dram_utilization_pct"]),
            occ=float(r["measured_occupancy_pct"]),
            mocc=float(r["model_occupancy_pct"]),
            elig=float(r["measured_eligible_warps_pct"]),
            melig=float(r["model_eligible_warps_pct"]),
            terr=float(r["model_time_error_pct"]),
        )
        for r in memory_rows
    )

    (MODEL_DIR / "model_validation_summary.md").write_text(
        f"""# stream_fma_v6 Model Validation

## Model

Inputs are `n`, `compute_iters`, `memory_iters`, `threads_per_cta`, and hardware
parameters. The model computes work, traffic, CTA mapping, warp occupancy, and
the bottleneck time:

```text
ops = n * (6 * compute_iters + 2)
algorithm_bytes = n * 16
ctas = n / (threads_per_cta * memory_iters)
predicted_time = max(compute_time, dram_time)
TOPS = ops / predicted_time / 1e12
```

The calibrated v6 model uses 12 modeled DRAM bytes per element because NCU
measures this kernel as read-dominated DRAM traffic, and it uses exposed
efficiency terms for scalar FMA issue and streaming DRAM bandwidth.

## Accuracy

| sweep | v6 model time MAPE | peak roofline time MAPE |
| --- | ---: | ---: |
| compute_iters, memory_iters=1 | {compute_time_mape:.2f}% | {compute_roof_time_mape:.2f}% |
| memory_iters, compute_iters=16 | {memory_time_mape:.2f}% | {memory_roof_time_mape:.2f}% |

Memory-sweep occupancy MAPE is `{occupancy_mape:.2f}%`.
DRAM BW utilization MAPE is `{compute_dram_util_mape:.2f}%` for the compute sweep
and `{memory_dram_util_mape:.2f}%` for the memory sweep.
Eligible-warp occupancy MAPE is `{compute_eligible_mape:.2f}%` for the compute
sweep and `{memory_eligible_mape:.2f}%` for the memory sweep.

## Compute-Iters Sweep

| compute_iters | measured TOPS | model TOPS | peak roofline TOPS | NCU DRAM peak % | model DRAM peak % | NCU eligible % | model eligible % | model time error % | bottleneck |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
{compute_table}

## Memory-Iters Sweep

| memory_iters | CTAs | measured TOPS | model TOPS | peak roofline TOPS | NCU DRAM peak % | model DRAM peak % | measured occ % | model occ % | NCU eligible % | model eligible % | model time error % |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{memory_table}

## Interpretation

The peak roofline baseline mispredicts this kernel because it combines ideal
hardware ceilings with algorithm-level DRAM bytes. In the memory-bound region it
is pessimistic relative to NCU, which reports this kernel as mostly read DRAM
traffic; in the high-compute region it becomes optimistic because scalar FMA
issue efficiency is below the CUDA-core peak. The v6 model adds kernel-specific
scalar-FMA efficiency, NCU-like DRAM traffic, and a warp occupancy estimate
based on CTA count and register pressure. This keeps the parameters physical
while matching the measured sweep more closely.
"""
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--refresh-ncu",
        action="store_true",
        help="rerun the v6 NCU analysis before validating",
    )
    parser.add_argument("--fpu-tops", type=float, default=55.9104)
    parser.add_argument("--dram-gbps", type=float, default=941.44)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.refresh_ncu:
        run(["python3", str(RUN_V6_ANALYSIS)])

    hw = HardwareParams(fpu_tops=args.fpu_tops, dram_bw_gbps=args.dram_gbps)
    compute_rows = validate_rows(read_csv(COMPUTE_CSV), hw)
    memory_rows = validate_rows(read_csv(MEMORY_CSV), hw)

    write_csv(compute_rows, MODEL_DIR / "model_validation_compute_iters.csv")
    write_csv(memory_rows, MODEL_DIR / "model_validation_memory_iters.csv")
    plot_compute(compute_rows)
    plot_memory(memory_rows)
    plot_occupancy(memory_rows)
    plot_eligible_compute(compute_rows)
    plot_eligible_memory(memory_rows)
    plot_dram_utilization(compute_rows, memory_rows)
    write_summary(compute_rows, memory_rows)


if __name__ == "__main__":
    main()
