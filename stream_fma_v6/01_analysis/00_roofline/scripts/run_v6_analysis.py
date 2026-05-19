#!/usr/bin/env python3
"""Collect stream_fma_v6 roofline, compute sweep, and memory sweep metrics."""

from __future__ import annotations

import csv
import json
import math
import re
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent
ROOF_DIR = SCRIPT_DIR.parent
V6_DIR = ROOF_DIR.parents[1]
ROOT = V6_DIR.parents[1]
V2_DIR = ROOT / "01_kernel" / "stream_fma_v2"
SWEEP_DIR = V6_DIR / "01_analysis" / "01_sweep_test"
RAW_DIR = ROOF_DIR / "ncu_raw"
BIN = V6_DIR / "stream_fma_v6"

N = 1_032_192
THREADS = 256
COMPUTE_SWEEP = [0, 1, 2, 4, 8, 16, 32, 64, 128]
MEMORY_SWEEP = [1, 2, 4, 8]
MEMORY_SWEEP_ITERS = 16

SPEC = {
    "gpu": "NVIDIA GeForce RTX 5080",
    "fp32_non_tensor_tflops": 56.349,
    "dram_gbps": 960.0,
}

METRICS = ",".join(
    [
        "gpu__time_duration",
        "dram__bytes",
        "dram__bytes.sum.per_second",
        "dram__bytes.sum.pct_of_peak_sustained_elapsed",
        "dram__bytes_op_read",
        "dram__bytes_op_write",
        "sm__sass_thread_inst_executed_op_ffma_pred_on",
        "sm__sass_thread_inst_executed_op_fadd_pred_on",
        "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct",
        "sm__warps_active.avg.per_cycle_active",
        "sm__warps_active.avg.pct_of_peak_sustained_active",
        "smsp__warps_eligible.avg.per_cycle_active",
        "smsp__warps_eligible.avg.pct_of_peak_sustained_active",
        "launch__occupancy_limit_blocks",
        "launch__occupancy_limit_registers",
        "launch__occupancy_limit_warps",
        "launch__occupancy_limit_shared_mem",
    ]
)

METRIC_RE = re.compile(
    r"^\s*([A-Za-z0-9_]+(?:\.[A-Za-z0-9_]+)*)\s+"
    r"([A-Za-z0-9_/%]+)\s+"
    r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s*$"
)


def run_logged(cmd: list[str], log_path: Path, expect_success: bool = True) -> str:
    print(" ".join(cmd), flush=True)
    result = subprocess.run(
        cmd,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    log_path.write_text("$ " + " ".join(cmd) + "\n\n" + result.stdout)
    if expect_success and result.returncode != 0:
        raise SystemExit(f"command failed ({result.returncode}); see {log_path}")
    if not expect_success and result.returncode == 0:
        raise SystemExit(f"command unexpectedly passed; see {log_path}")
    return result.stdout


def parse_metrics(text: str) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for line in text.splitlines():
        match = METRIC_RE.match(line)
        if match:
            name, _unit, value = match.groups()
            metrics[name] = float(value)
    return metrics


def get_metric(metrics: dict[str, float], name: str) -> float:
    for key in (name, name + ".sum", name + ".avg"):
        if key in metrics:
            return metrics[key]
    return 0.0


def ctas_for(memory_iters: int) -> int:
    return N // (THREADS * memory_iters)


def app_cmd(memory_iters: int, compute_iters: int) -> list[str]:
    return [
        str(BIN),
        "--n",
        str(N),
        "--iters",
        str(compute_iters),
        "--memory-iters",
        str(memory_iters),
        "--warmup",
        "0",
        "--repeats",
        "1",
    ]


def ncu_cmd(memory_iters: int, compute_iters: int) -> list[str]:
    return [
        "/usr/local/cuda-13.2/bin/ncu",
        "--target-processes",
        "all",
        "--kernel-name",
        f"regex:stream_fma_v6_m{memory_iters}_kernel",
        "--launch-count",
        "1",
        "--metrics",
        METRICS,
        "--print-units",
        "base",
        "--print-metric-name",
        "name",
        *app_cmd(memory_iters, compute_iters),
    ]


def collect_point(memory_iters: int, compute_iters: int) -> dict[str, float | int | str]:
    kernel = f"stream_fma_v6_m{memory_iters}_kernel"
    text = run_logged(
        ncu_cmd(memory_iters, compute_iters),
        RAW_DIR / f"{kernel}_i{compute_iters}.txt",
    )
    metrics = parse_metrics(text)
    ffma = get_metric(metrics, "sm__sass_thread_inst_executed_op_ffma_pred_on")
    fadd = get_metric(metrics, "sm__sass_thread_inst_executed_op_fadd_pred_on")
    flops = 2.0 * ffma + fadd
    duration_ns = get_metric(metrics, "gpu__time_duration")
    dram_bytes = get_metric(metrics, "dram__bytes")
    elements_per_cta = THREADS * memory_iters
    ctas = ctas_for(memory_iters)
    algorithm_bytes = float(N) * 16.0

    return {
        "kernel": kernel,
        "n": N,
        "memory_iters": memory_iters,
        "threads_per_cta": THREADS,
        "ctas": ctas,
        "elements_per_cta": elements_per_cta,
        "elements_per_thread": memory_iters,
        "compute_iters": compute_iters,
        "ai_algorithm": (6.0 * compute_iters + 2.0) / 16.0,
        "duration_ns": duration_ns,
        "dram_bytes": dram_bytes,
        "dram_read_bytes": get_metric(metrics, "dram__bytes_op_read"),
        "dram_write_bytes": get_metric(metrics, "dram__bytes_op_write"),
        "dram_GBps": get_metric(metrics, "dram__bytes.sum.per_second") / 1.0e9,
        "dram_peak_pct": get_metric(
            metrics, "dram__bytes.sum.pct_of_peak_sustained_elapsed"
        ),
        "ffma_inst": ffma,
        "fadd_inst": fadd,
        "flops": flops,
        "algorithm_bytes": algorithm_bytes,
        "ai_ncu": flops / dram_bytes if dram_bytes else math.nan,
        "gflops": flops / (duration_ns * 1.0e-9) / 1.0e9
        if duration_ns
        else math.nan,
        "long_scoreboard_pct": get_metric(
            metrics, "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct"
        ),
        "active_warps_per_sm_cycle": get_metric(
            metrics, "sm__warps_active.avg.per_cycle_active"
        ),
        "achieved_occupancy_pct": get_metric(
            metrics, "sm__warps_active.avg.pct_of_peak_sustained_active"
        ),
        "eligible_warps_per_scheduler_cycle": get_metric(
            metrics, "smsp__warps_eligible.avg.per_cycle_active"
        ),
        "eligible_warps_pct": get_metric(
            metrics, "smsp__warps_eligible.avg.pct_of_peak_sustained_active"
        ),
        "limit_blocks": get_metric(metrics, "launch__occupancy_limit_blocks"),
        "limit_registers": get_metric(metrics, "launch__occupancy_limit_registers"),
        "limit_warps": get_metric(metrics, "launch__occupancy_limit_warps"),
        "limit_shared_mem": get_metric(metrics, "launch__occupancy_limit_shared_mem"),
    }


def write_csv(rows: list[dict[str, float | int | str]], path: Path) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open() as f:
        return list(csv.DictReader(f))


def row_by_key(rows: list[dict[str, str]], key_name: str) -> dict[int, dict[str, str]]:
    return {int(row[key_name]): row for row in rows}


def v2_memory_gflops(row: dict[str, str]) -> float:
    flops = float(row["n"]) * (6.0 * float(row["compute_iters"]) + 2.0)
    return flops / (float(row["duration_ns"]) * 1.0e-9) / 1.0e9


def roof_values(xs: list[float], compute_tflops: float, bw_gbps: float) -> list[float]:
    return [min(compute_tflops, bw_gbps * x / 1000.0) for x in xs]


def plot_roofline(
    rows: list[dict[str, float | int | str]],
    summary: dict[str, float | int | list[int]],
    path: Path,
) -> None:
    xs = [10 ** (-2 + i * (6 / 400)) for i in range(401)]
    spec_y = roof_values(
        xs, float(summary["spec_fp32_tflops"]), float(summary["spec_dram_gbps"])
    )
    ncu_y = roof_values(
        xs,
        float(summary["ncu_clock_fp32_tflops"]),
        float(summary["ncu_clock_dram_gbps"]),
    )

    fig, ax = plt.subplots(figsize=(9.5, 6.2), dpi=160)
    ax.loglog(xs, spec_y, color="#1f77b4", linewidth=2.0, label="Spec roofline")
    ax.loglog(
        xs,
        ncu_y,
        color="#d62728",
        linewidth=2.0,
        linestyle="--",
        label="NCU clock-adjusted roofline",
    )
    ax.scatter(
        [float(r["ai_ncu"]) for r in rows],
        [float(r["gflops"]) / 1000.0 for r in rows],
        marker="o",
        s=44,
        color="#2ca02c",
        label="v6 m1, NCU DRAM bytes",
    )
    ax.scatter(
        [float(r["ai_algorithm"]) for r in rows],
        [float(r["gflops"]) / 1000.0 for r in rows],
        marker="x",
        s=50,
        color="#111111",
        label="v6 m1, algorithm bytes",
    )
    for row in rows:
        ax.annotate(
            str(row["compute_iters"]),
            (float(row["ai_ncu"]), float(row["gflops"]) / 1000.0),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
        )
    ax.set_xlabel("Arithmetic intensity (FLOP/DRAM byte)")
    ax.set_ylabel("FP32 throughput (TFLOP/s)")
    ax.set_title("RTX 5080 CUDA-Core FP32 Roofline for stream_fma_v6_m1")
    ax.grid(True, which="both", linestyle=":", linewidth=0.6, alpha=0.7)
    ax.legend(loc="lower right", fontsize=8)
    ax.set_xlim(1e-2, 1e4)
    ymax = max(
        float(summary["spec_fp32_tflops"]),
        float(summary["ncu_clock_fp32_tflops"]),
        *(float(r["gflops"]) / 1000.0 for r in rows),
    ) * 1.5
    ax.set_ylim(1e-3, ymax)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_active_eligible_scoreboard(
    rows: list[dict[str, float | int | str]], path: Path
) -> None:
    plot_scheduler_readiness(
        rows=rows,
        path=path,
        x_key="compute_iters",
        xlabel="compute_iters",
        title="stream_fma_v6 m1: Active vs Eligible Warps",
        symlog=True,
    )


def plot_memory_active_eligible_scoreboard(
    rows: list[dict[str, float | int | str]], path: Path
) -> None:
    plot_scheduler_readiness(
        rows=rows,
        path=path,
        x_key="memory_iters",
        xlabel="memory_iters",
        title="stream_fma_v6 compute_iters=16: Active vs Eligible Warps",
        symlog=False,
    )


def plot_scheduler_readiness(
    rows: list[dict[str, float | int | str]],
    path: Path,
    x_key: str,
    xlabel: str,
    title: str,
    symlog: bool,
) -> None:
    x = [int(row[x_key]) for row in rows]
    active = [float(row["achieved_occupancy_pct"]) for row in rows]
    eligible = [float(row["eligible_warps_pct"]) for row in rows]
    scoreboard = [float(row["long_scoreboard_pct"]) for row in rows]

    fig, ax = plt.subplots(figsize=(9.0, 5.2), dpi=160)
    ax.plot(
        x,
        active,
        marker="o",
        linewidth=2.0,
        color="#1f77b4",
        label="active warp occupancy (%)",
    )
    ax.plot(
        x,
        eligible,
        marker="s",
        linewidth=2.0,
        color="#2ca02c",
        label="eligible warps (%)",
    )
    if symlog:
        ax.set_xscale("symlog", linthresh=1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("warp metric (% of peak sustained active)")
    ax.set_ylim(0, max(100.0, max(active + eligible) * 1.1))
    ax.grid(True, linestyle=":", linewidth=0.7)

    ax2 = ax.twinx()
    ax2.plot(
        x,
        scoreboard,
        marker="^",
        linewidth=2.0,
        linestyle="--",
        color="#d62728",
        label="long scoreboard stall (%)",
    )
    ax2.set_ylabel("long scoreboard stall (%)")
    ax2.set_ylim(0, max(100.0, max(scoreboard) * 1.1))

    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="best", fontsize=8)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def load_summary_defaults() -> dict[str, float]:
    v2_summary_path = V2_DIR / "01_analysis" / "00_roofline" / "roofline_summary.json"
    if not v2_summary_path.exists():
        return {
            "ncu_clock_fp32_tflops": SPEC["fp32_non_tensor_tflops"],
            "ncu_clock_dram_gbps": SPEC["dram_gbps"],
        }
    v2_summary = json.loads(v2_summary_path.read_text())
    return {
        "ncu_clock_fp32_tflops": float(v2_summary["ncu_clock_fp32_tflops"]),
        "ncu_clock_dram_gbps": float(v2_summary["ncu_clock_dram_gbps"]),
    }


def write_markdown(
    compute_rows: list[dict[str, float | int | str]],
    memory_rows: list[dict[str, float | int | str]],
    summary: dict[str, float | int | list[int]],
) -> None:
    v2_compute_csv = V2_DIR / "01_analysis" / "00_roofline" / "roofline_v2_m1_data.csv"
    v2_memory_csv = (
        V2_DIR / "01_analysis" / "01_sweep_test" / "memory_iters_sweep_compute_iters_16.csv"
    )
    v2_compute = row_by_key(read_csv(v2_compute_csv), "iters") if v2_compute_csv.exists() else {}
    v2_memory = row_by_key(read_csv(v2_memory_csv), "memory_iters") if v2_memory_csv.exists() else {}

    compute_table = "\n".join(
        "| {iters} | {ai:.3f} | {gflops:.1f} | {v2} | {dram:.1f} | {dram_pct:.2f} | {long:.2f} | {occ:.2f} | {eligible:.2f} |".format(
            iters=int(row["compute_iters"]),
            ai=float(row["ai_algorithm"]),
            gflops=float(row["gflops"]),
            v2="{:.1f}".format(float(v2_compute[int(row["compute_iters"])]["gflops"]))
            if int(row["compute_iters"]) in v2_compute
            else "n/a",
            dram=float(row["dram_GBps"]),
            dram_pct=float(row["dram_peak_pct"]),
            long=float(row["long_scoreboard_pct"]),
            occ=float(row["achieved_occupancy_pct"]),
            eligible=float(row["eligible_warps_pct"]),
        )
        for row in compute_rows
    )
    sweep_compute_table = "\n".join(
        "| {iters} | {ai:.3f} | {gflops:.1f} | {dram:.1f} | {dram_pct:.2f} | {long:.2f} | {occ:.2f} | {eligible:.2f} | {eligible_raw:.2f} |".format(
            iters=int(row["compute_iters"]),
            ai=float(row["ai_algorithm"]),
            gflops=float(row["gflops"]),
            dram=float(row["dram_GBps"]),
            dram_pct=float(row["dram_peak_pct"]),
            long=float(row["long_scoreboard_pct"]),
            occ=float(row["achieved_occupancy_pct"]),
            eligible=float(row["eligible_warps_pct"]),
            eligible_raw=float(row["eligible_warps_per_scheduler_cycle"]),
        )
        for row in compute_rows
    )
    memory_table = "\n".join(
        "| {m} | {ctas} | {gflops:.1f} | {v2gflops} | {dram:.1f} | {v2dram} | {long:.2f} | {v2long} | {occ:.2f} | {v2occ} | {eligible:.2f} | {eligible_raw:.2f} |".format(
            m=int(row["memory_iters"]),
            ctas=int(row["ctas"]),
            gflops=float(row["gflops"]),
            v2gflops="{:.1f}".format(v2_memory_gflops(v2_memory[int(row["memory_iters"])]))
            if int(row["memory_iters"]) in v2_memory
            else "n/a",
            dram=float(row["dram_GBps"]),
            v2dram="{:.1f}".format(float(v2_memory[int(row["memory_iters"])]["dram_GBps"]))
            if int(row["memory_iters"]) in v2_memory
            else "n/a",
            long=float(row["long_scoreboard_pct"]),
            v2long="{:.2f}".format(
                float(v2_memory[int(row["memory_iters"])]["long_scoreboard_pct"])
            )
            if int(row["memory_iters"]) in v2_memory
            else "n/a",
            occ=float(row["achieved_occupancy_pct"]),
            v2occ="{:.2f}".format(
                float(v2_memory[int(row["memory_iters"])]["achieved_occupancy_pct"])
            )
            if int(row["memory_iters"]) in v2_memory
            else "n/a",
            eligible=float(row["eligible_warps_pct"]),
            eligible_raw=float(row["eligible_warps_per_scheduler_cycle"]),
        )
        for row in memory_rows
    )

    (ROOF_DIR / "stream_fma_v6_roofline_summary.md").write_text(
        f"""# stream_fma_v6 Roofline Summary

## Scope

This profiles `stream_fma_v6_m1_kernel` with `n={N}` for the roofline. v6 uses
a conventional grid-size-from-work launch like v5, but each thread processes
`memory_iters` elements in one CTA tile. There is no outer `round` loop. Tensor
Core throughput is excluded. FLOPs use NCU counters: `2 * FFMA + FADD`.

## Hardware Ceilings

- GPU: {SPEC["gpu"]}
- Spec CUDA-core FP32 peak: {summary["spec_fp32_tflops"]:.3f} TFLOP/s
- Spec DRAM bandwidth: {summary["spec_dram_gbps"]:.1f} GB/s
- NCU clock-adjusted CUDA-core FP32 ceiling: {summary["ncu_clock_fp32_tflops"]:.3f} TFLOP/s
- NCU clock-adjusted DRAM ceiling: {summary["ncu_clock_dram_gbps"]:.1f} GB/s

## Compute-Iters Sweep

| compute_iters | AI algorithm | v6 m1 GFLOP/s | v2 m1 GFLOP/s | DRAM GB/s | DRAM peak % | long score % | active occ % | eligible % |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{compute_table}

## Interpretation

For `memory_iters=1`, v6 and v5 have the same one-thread-per-element mapping.
The v6 memory sweep changes CTAs instead of outer rounds: larger
`memory_iters` means fewer CTAs and more elements per thread in the same launch.
"""
    )

    (SWEEP_DIR / "compute_iters_sweep_memory_iters_1.md").write_text(
        f"""# stream_fma_v6 Compute-Iters Sweep

## Scope

This is the NCU compute sweep for `stream_fma_v6_m1_kernel`.

| compute_iters | AI algorithm | GFLOP/s | DRAM GB/s | DRAM peak % | long score % | active occ % | eligible % | eligible warps/SMSP cycle |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{sweep_compute_table}

![Active vs eligible warps and long scoreboard](compute_iters_active_eligible_scoreboard.png)
"""
    )

    (SWEEP_DIR / "memory_iters_sweep_compute_iters_16.md").write_text(
        f"""# stream_fma_v6 Memory-Iters Sweep

## Scope

This profiles `stream_fma_v6` with fixed `n={N}` and `compute_iters=16`, sweeping
`memory_iters=1,2,4,8`. Unlike v2, v6 has no outer rounds; increasing
`memory_iters` reduces CTA count because each thread handles more elements once.

## Results and v2 Comparison

| memory_iters | v6 CTAs | v6 GFLOP/s | v2 GFLOP/s | v6 DRAM GB/s | v2 DRAM GB/s | v6 long score % | v2 long score % | v6 active occ % | v2 occ % | v6 eligible % | eligible warps/SMSP cycle |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{memory_table}

![Active vs eligible warps and long scoreboard](memory_iters_active_eligible_scoreboard.png)

## Interpretation

v2 keeps CTAs fixed at 504 and changes outer rounds. v6 keeps total work fixed
and reduces CTAs from 4032 to 504 as `memory_iters` grows from 1 to 8. This
separates per-thread batching from round-based thread reuse.
"""
    )


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    SWEEP_DIR.mkdir(parents=True, exist_ok=True)

    run_logged(["make", "-C", str(V6_DIR), "all", "instructions"], RAW_DIR / "build.txt")
    for memory_iters in MEMORY_SWEEP:
        run_logged(
            [
                str(BIN),
                "--n",
                str(N),
                "--iters",
                "16",
                "--memory-iters",
                str(memory_iters),
                "--warmup",
                "2",
                "--repeats",
                "5",
            ],
            RAW_DIR / f"verify_m{memory_iters}_n{N}_i16.txt",
        )
    run_logged(
        [
            str(BIN),
            "--n",
            "1032301",
            "--iters",
            "16",
            "--memory-iters",
            "4",
            "--warmup",
            "0",
            "--repeats",
            "1",
        ],
        RAW_DIR / "verify_reject_unsafe_n_m4.txt",
        expect_success=False,
    )

    compute_rows = [collect_point(1, iters) for iters in COMPUTE_SWEEP]
    memory_rows = [collect_point(memory_iters, MEMORY_SWEEP_ITERS) for memory_iters in MEMORY_SWEEP]
    write_csv(compute_rows, ROOF_DIR / "roofline_v6_m1_data.csv")
    write_csv(compute_rows, SWEEP_DIR / "compute_iters_sweep_memory_iters_1.csv")
    write_csv(memory_rows, SWEEP_DIR / "memory_iters_sweep_compute_iters_16.csv")

    ncu_defaults = load_summary_defaults()
    summary = {
        "spec_fp32_tflops": SPEC["fp32_non_tensor_tflops"],
        "spec_dram_gbps": SPEC["dram_gbps"],
        "ncu_clock_fp32_tflops": ncu_defaults["ncu_clock_fp32_tflops"],
        "ncu_clock_dram_gbps": ncu_defaults["ncu_clock_dram_gbps"],
        "n": N,
        "threads_per_cta": THREADS,
        "memory_iters_roofline": 1,
        "compute_sweep": COMPUTE_SWEEP,
        "memory_sweep_compute_iters": MEMORY_SWEEP_ITERS,
        "memory_sweep": MEMORY_SWEEP,
    }
    (ROOF_DIR / "roofline_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    plot_roofline(compute_rows, summary, ROOF_DIR / "roofline_stream_fma_v6_m1.png")
    plot_active_eligible_scoreboard(
        compute_rows, SWEEP_DIR / "compute_iters_active_eligible_scoreboard.png"
    )
    plot_memory_active_eligible_scoreboard(
        memory_rows, SWEEP_DIR / "memory_iters_active_eligible_scoreboard.png"
    )
    write_markdown(compute_rows, memory_rows, summary)


if __name__ == "__main__":
    main()
