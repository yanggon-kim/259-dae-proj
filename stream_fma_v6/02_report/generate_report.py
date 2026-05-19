#!/usr/bin/env python3
"""Generate Markdown, HTML, and PDF report for the stream_fma_v6 model."""

from __future__ import annotations

import argparse
import csv
import html
import math
import re
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt


REPORT_DIR = Path(__file__).resolve().parent
V6_DIR = REPORT_DIR.parent
ROOT = V6_DIR.parent.parent
MODEL_DIR = V6_DIR / "01_analysis" / "02_model"
SWEEP_DIR = V6_DIR / "01_analysis" / "01_sweep_test"
MODEL_SCRIPT_DIR = str(MODEL_DIR)
if MODEL_SCRIPT_DIR not in sys.path:
    sys.path.insert(0, MODEL_SCRIPT_DIR)

from stream_fma_v6_model import (  # noqa: E402
    HardwareParams,
    KernelParams,
    KernelResources,
    make_peak_roofline_hardware,
    predict,
)


ASSET_DIR = REPORT_DIR / "assets"
RAW_DIR = REPORT_DIR / "ncu_raw"
BIN = V6_DIR / "stream_fma_v6"
NCU = "/usr/local/cuda-13.2/bin/ncu"
CHROME = "/usr/bin/google-chrome"

PROBLEM_SIZES = [258_048, 516_096, 1_032_192, 2_064_384, 4_128_768]
PROBLEM_COMPUTE_ITERS = 16
PROBLEM_MEMORY_ITERS = 4
THREADS = 256
METRICS = ",".join(
    [
        "gpu__time_duration",
        "dram__bytes",
        "dram__bytes.sum.per_second",
        "dram__bytes.sum.pct_of_peak_sustained_elapsed",
        "sm__sass_thread_inst_executed_op_ffma_pred_on",
        "sm__sass_thread_inst_executed_op_fadd_pred_on",
        "sm__warps_active.avg.pct_of_peak_sustained_active",
    ]
)
METRIC_RE = re.compile(
    r"^\s*([A-Za-z0-9_]+(?:\.[A-Za-z0-9_]+)*)\s+"
    r"([A-Za-z0-9_/%]+)\s+"
    r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s*$"
)


def run_logged(cmd: list[str], log_path: Path | None = None) -> str:
    print(" ".join(cmd), flush=True)
    result = subprocess.run(
        cmd,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if log_path is not None:
        log_path.write_text("$ " + " ".join(cmd) + "\n\n" + result.stdout)
    if result.returncode != 0:
        raise SystemExit(f"command failed ({result.returncode}); see {log_path}")
    return result.stdout


def parse_metrics(text: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for line in text.splitlines():
        match = METRIC_RE.match(line)
        if match:
            name, _unit, value = match.groups()
            out[name] = float(value)
    return out


def get_metric(metrics: dict[str, float], name: str) -> float:
    for key in (name, name + ".sum", name + ".avg"):
        if key in metrics:
            return metrics[key]
    return 0.0


def mape(actual: list[float], predicted: list[float]) -> float:
    values = [
        abs((p - a) / a)
        for a, p in zip(actual, predicted)
        if a and not math.isnan(a) and not math.isnan(p)
    ]
    return sum(values) / len(values) * 100.0 if values else math.nan


def pct_error(actual: float, predicted: float) -> float:
    return (predicted - actual) / actual * 100.0 if actual else math.nan


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open() as f:
        return list(csv.DictReader(f))


def measured_from_ncu(n: int, refresh: bool) -> dict[str, float]:
    log_path = RAW_DIR / f"problem_n{n}_m{PROBLEM_MEMORY_ITERS}_i{PROBLEM_COMPUTE_ITERS}.txt"
    if refresh or not log_path.exists():
        cmd = [
            NCU,
            "--target-processes",
            "all",
            "--kernel-name",
            f"regex:stream_fma_v6_m{PROBLEM_MEMORY_ITERS}_kernel",
            "--launch-count",
            "1",
            "--metrics",
            METRICS,
            "--print-units",
            "base",
            "--print-metric-name",
            "name",
            str(BIN),
            "--n",
            str(n),
            "--iters",
            str(PROBLEM_COMPUTE_ITERS),
            "--memory-iters",
            str(PROBLEM_MEMORY_ITERS),
            "--warmup",
            "0",
            "--repeats",
            "1",
        ]
        text = run_logged(cmd, log_path)
    else:
        text = log_path.read_text()

    metrics = parse_metrics(text)
    ffma = get_metric(metrics, "sm__sass_thread_inst_executed_op_ffma_pred_on")
    fadd = get_metric(metrics, "sm__sass_thread_inst_executed_op_fadd_pred_on")
    flops = 2.0 * ffma + fadd
    duration_ns = get_metric(metrics, "gpu__time_duration")
    dram_bytes = get_metric(metrics, "dram__bytes")
    return {
        "duration_ns": duration_ns,
        "flops": flops,
        "tops": flops / (duration_ns * 1.0e-9) / 1.0e12,
        "dram_bytes": dram_bytes,
        "dram_gbps": get_metric(metrics, "dram__bytes.sum.per_second") / 1.0e9,
        "dram_peak_pct": get_metric(metrics, "dram__bytes.sum.pct_of_peak_sustained_elapsed"),
        "occupancy_pct": get_metric(metrics, "sm__warps_active.avg.pct_of_peak_sustained_active"),
    }


def predict_row(n: int, compute_iters: int, memory_iters: int, hw: HardwareParams) -> dict[str, float]:
    kernel = KernelParams(
        n=n,
        compute_iters=compute_iters,
        memory_iters=memory_iters,
        threads_per_cta=THREADS,
    )
    resources = KernelResources.for_memory_iters(memory_iters)
    model = predict(kernel, hw, resources)
    roofline = predict(kernel, make_peak_roofline_hardware(hw), resources)
    return {
        "ctas": float(model.mapping.ctas),
        "model_time_ns": model.predicted_time_s * 1.0e9,
        "roofline_time_ns": roofline.predicted_time_s * 1.0e9,
        "model_tops": model.predicted_tops,
        "roofline_tops": roofline.predicted_tops,
        "oi_algorithm": model.operational_intensity_algorithm,
        "model_predicted_dram_gbps": model.predicted_dram_gbps,
        "model_dram_utilization_pct": model.predicted_dram_utilization_pct,
        "model_effective_dram_utilization_pct": model.effective_dram_utilization_pct,
        "model_occupancy_pct": model.occupancy.estimated_achieved_occupancy * 100.0,
    }


def collect_problem_size_data(refresh_ncu: bool) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    hw = HardwareParams()
    for n in PROBLEM_SIZES:
        measured = measured_from_ncu(n, refresh_ncu)
        pred = predict_row(n, PROBLEM_COMPUTE_ITERS, PROBLEM_MEMORY_ITERS, hw)
        rows.append(
            {
                "n": n,
                "compute_iters": PROBLEM_COMPUTE_ITERS,
                "memory_iters": PROBLEM_MEMORY_ITERS,
                "ctas": int(pred["ctas"]),
                "measured_time_ns": measured["duration_ns"],
                "model_time_ns": pred["model_time_ns"],
                "roofline_time_ns": pred["roofline_time_ns"],
                "model_time_error_pct": pct_error(measured["duration_ns"], pred["model_time_ns"]),
                "roofline_time_error_pct": pct_error(measured["duration_ns"], pred["roofline_time_ns"]),
                "measured_tops": measured["tops"],
                "model_tops": pred["model_tops"],
                "roofline_tops": pred["roofline_tops"],
                "measured_dram_gbps": measured["dram_gbps"],
                "model_predicted_dram_gbps": pred["model_predicted_dram_gbps"],
                "measured_dram_utilization_pct": measured["dram_peak_pct"],
                "model_dram_utilization_pct": pred["model_dram_utilization_pct"],
                "model_effective_dram_utilization_pct": pred[
                    "model_effective_dram_utilization_pct"
                ],
                "occupancy_pct": measured["occupancy_pct"],
            }
        )
    write_csv(rows, REPORT_DIR / "problem_size_validation.csv")
    return rows


def load_tile_data() -> list[dict[str, object]]:
    rows = read_csv(MODEL_DIR / "model_validation_memory_iters.csv")
    return [
        {
            "memory_iters": int(r["memory_iters"]),
            "ctas": int(r["ctas"]),
            "model_time_error_pct": float(r["model_time_error_pct"]),
            "roofline_time_error_pct": float(r["roofline_time_error_pct"]),
            "measured_tops": float(r["measured_tops"]),
            "model_tops": float(r["model_tops"]),
            "roofline_tops": float(r["roofline_tops"]),
            "measured_dram_gbps": float(r.get("measured_dram_gbps", r.get("dram_gbps", "nan"))),
            "model_predicted_dram_gbps": float(r.get("model_predicted_dram_gbps", "nan")),
            "measured_dram_utilization_pct": float(r.get("measured_dram_utilization_pct", "nan")),
            "model_dram_utilization_pct": float(r.get("model_dram_utilization_pct", "nan")),
            "measured_occupancy_pct": float(r["measured_occupancy_pct"]),
            "model_occupancy_pct": float(r["model_occupancy_pct"]),
            "measured_eligible_warps_pct": float(r["measured_eligible_warps_pct"]),
            "model_eligible_warps_pct": float(r["model_eligible_warps_pct"]),
            "measured_eligible_warps_per_scheduler_cycle": float(
                r["measured_eligible_warps_per_scheduler_cycle"]
            ),
            "model_eligible_warps_per_scheduler_cycle": float(
                r["model_eligible_warps_per_scheduler_cycle"]
            ),
            "model_compute_exposure_fraction": float(
                r["model_compute_exposure_fraction"]
            ),
        }
        for r in rows
    ]


def load_compute_model_data() -> list[dict[str, object]]:
    rows = read_csv(MODEL_DIR / "model_validation_compute_iters.csv")
    return [
        {
            "compute_iters": int(r["compute_iters"]),
            "measured_eligible_warps_pct": float(r["measured_eligible_warps_pct"]),
            "model_eligible_warps_pct": float(r["model_eligible_warps_pct"]),
            "measured_eligible_warps_per_scheduler_cycle": float(
                r["measured_eligible_warps_per_scheduler_cycle"]
            ),
            "model_eligible_warps_per_scheduler_cycle": float(
                r["model_eligible_warps_per_scheduler_cycle"]
            ),
            "model_compute_exposure_fraction": float(
                r["model_compute_exposure_fraction"]
            ),
        }
        for r in rows
    ]


def load_scheduler_sweep(path: Path, key: str) -> list[dict[str, object]]:
    rows = read_csv(path)
    return [
        {
            key: int(r[key]),
            "gflops": float(r["gflops"]),
            "active_occupancy_pct": float(r["achieved_occupancy_pct"]),
            "eligible_warps_pct": float(r["eligible_warps_pct"]),
            "eligible_warps_per_scheduler_cycle": float(
                r["eligible_warps_per_scheduler_cycle"]
            ),
            "long_scoreboard_pct": float(r["long_scoreboard_pct"]),
        }
        for r in rows
    ]


def plot_problem_error(rows: list[dict[str, object]]) -> None:
    x = [int(r["n"]) for r in rows]
    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=160)
    ax.plot(x, [abs(float(r["model_time_error_pct"])) for r in rows], marker="o", label="model")
    ax.plot(
        x,
        [abs(float(r["roofline_time_error_pct"])) for r in rows],
        marker="s",
        label="roofline",
    )
    ax.set_xscale("log", base=2)
    ax.set_xlabel("n elements")
    ax.set_ylabel("absolute runtime error (%)")
    ax.set_title("Problem-size error, compute_iters=16, memory_iters=4")
    ax.grid(True, linestyle=":", linewidth=0.7)
    ax.legend()
    fig.tight_layout()
    fig.savefig(ASSET_DIR / "error_vs_problem_size.png")
    plt.close(fig)


def plot_tile_error(rows: list[dict[str, object]]) -> None:
    x = [int(r["memory_iters"]) for r in rows]
    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=160)
    ax.plot(x, [abs(float(r["model_time_error_pct"])) for r in rows], marker="o", label="model")
    ax.plot(
        x,
        [abs(float(r["roofline_time_error_pct"])) for r in rows],
        marker="s",
        label="roofline",
    )
    ax.set_xlabel("memory_iters")
    ax.set_ylabel("absolute runtime error (%)")
    ax.set_title("Tile-size error, n=1032192, compute_iters=16")
    ax.grid(True, linestyle=":", linewidth=0.7)
    ax.legend()
    fig.tight_layout()
    fig.savefig(ASSET_DIR / "error_vs_tile_size.png")
    plt.close(fig)


def plot_problem_dram_utilization(rows: list[dict[str, object]]) -> None:
    x = [int(r["n"]) for r in rows]
    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=160)
    ax.plot(
        x,
        [float(r["measured_dram_utilization_pct"]) for r in rows],
        marker="o",
        label="NCU DRAM peak %",
    )
    ax.plot(
        x,
        [float(r["model_dram_utilization_pct"]) for r in rows],
        marker="s",
        label="model DRAM peak %",
    )
    ax.set_xscale("log", base=2)
    ax.set_xlabel("n elements")
    ax.set_ylabel("DRAM BW utilization (%)")
    ax.set_title("Problem-size DRAM utilization, compute_iters=16, memory_iters=4")
    ax.grid(True, linestyle=":", linewidth=0.7)
    ax.legend()
    fig.tight_layout()
    fig.savefig(ASSET_DIR / "dram_utilization_problem_size.png")
    plt.close(fig)


def plot_sensitivity() -> None:
    base_kernel = KernelParams(n=1_032_192, compute_iters=16, memory_iters=4)
    base_hw = HardwareParams()
    factors = [0.5, 0.75, 1.0, 1.25, 1.5]
    params = {
        "DRAM BW": "dram_bw_gbps",
        "FPU TOPS": "fpu_tops",
        "SM count": "sm_count",
        "registers/SM": "registers_per_sm",
        "DAE/WASP memory service": "dram_efficiency",
    }
    fig, ax = plt.subplots(figsize=(7.6, 4.8), dpi=160)
    base_tops = predict(base_kernel, base_hw).predicted_tops
    for label, field in params.items():
        ys = []
        for factor in factors:
            kwargs = base_hw.__dict__.copy()
            value = kwargs[field]
            kwargs[field] = max(1, int(value * factor)) if isinstance(value, int) else value * factor
            hw = HardwareParams(**kwargs)
            ys.append(predict(base_kernel, hw).predicted_tops / base_tops)
        ax.plot(factors, ys, marker="o", label=label)
    ax.axhline(1.0, color="#444444", linewidth=0.8)
    ax.set_xlabel("hardware parameter scale")
    ax.set_ylabel("predicted TOPS / baseline")
    ax.set_title("Sensitivity, n=1032192, compute_iters=16, memory_iters=4")
    ax.grid(True, linestyle=":", linewidth=0.7)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(ASSET_DIR / "sensitivity_analysis.png")
    plt.close(fig)


def copy_model_plots() -> None:
    for name in [
        "model_vs_ncu_compute_iters.png",
        "model_vs_ncu_memory_iters.png",
        "occupancy_vs_ncu_memory_iters.png",
        "dram_bw_utilization_vs_ncu.png",
        "eligible_warps_vs_ncu_compute_iters.png",
        "eligible_warps_vs_ncu_memory_iters.png",
    ]:
        shutil.copy2(MODEL_DIR / name, ASSET_DIR / name)


def copy_scheduler_plots() -> None:
    for name in [
        "compute_iters_active_eligible_scoreboard.png",
        "memory_iters_active_eligible_scoreboard.png",
    ]:
        shutil.copy2(SWEEP_DIR / name, ASSET_DIR / name)


def format_problem_table(rows: list[dict[str, object]]) -> str:
    lines = [
        "| n | CTAs | measured TOPS | model TOPS | NCU DRAM peak % | model DRAM peak % | model error % | roofline error % |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        lines.append(
            "| {n} | {ctas} | {meas:.3f} | {model:.3f} | {dram:.2f} | {mdram:.2f} | {me:.2f} | {re:.2f} |".format(
                n=int(r["n"]),
                ctas=int(r["ctas"]),
                meas=float(r["measured_tops"]),
                model=float(r["model_tops"]),
                dram=float(r["measured_dram_utilization_pct"]),
                mdram=float(r["model_dram_utilization_pct"]),
                me=float(r["model_time_error_pct"]),
                re=float(r["roofline_time_error_pct"]),
            )
        )
    return "\n".join(lines)


def format_scheduler_table(rows: list[dict[str, object]], key: str, label: str) -> str:
    lines = [
        f"| {label} | GFLOP/s | active occ % | eligible % | eligible warps/SMSP cycle | long scoreboard % |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        lines.append(
            "| {x} | {gflops:.1f} | {active:.2f} | {eligible:.2f} | {eligible_raw:.2f} | {long:.2f} |".format(
                x=int(r[key]),
                gflops=float(r["gflops"]),
                active=float(r["active_occupancy_pct"]),
                eligible=float(r["eligible_warps_pct"]),
                eligible_raw=float(r["eligible_warps_per_scheduler_cycle"]),
                long=float(r["long_scoreboard_pct"]),
            )
        )
    return "\n".join(lines)


def format_eligible_table(rows: list[dict[str, object]], key: str, label: str) -> str:
    lines = [
        f"| {label} | NCU eligible % | model eligible % | NCU eligible warps/SMSP cycle | model eligible warps/SMSP cycle | compute exposure |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        lines.append(
            "| {x} | {eligible:.2f} | {meligible:.2f} | {eligible_raw:.2f} | {meligible_raw:.2f} | {exposure:.3f} |".format(
                x=int(r[key]),
                eligible=float(r["measured_eligible_warps_pct"]),
                meligible=float(r["model_eligible_warps_pct"]),
                eligible_raw=float(r["measured_eligible_warps_per_scheduler_cycle"]),
                meligible_raw=float(r["model_eligible_warps_per_scheduler_cycle"]),
                exposure=float(r["model_compute_exposure_fraction"]),
            )
        )
    return "\n".join(lines)


def format_tile_table(rows: list[dict[str, object]]) -> str:
    lines = [
        "| memory_iters | CTAs | measured TOPS | model TOPS | NCU DRAM peak % | model DRAM peak % | measured occ % | model occ % |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        lines.append(
            "| {m} | {ctas} | {meas:.3f} | {model:.3f} | {dram:.2f} | {mdram:.2f} | {occ:.2f} | {mocc:.2f} |".format(
                m=int(r["memory_iters"]),
                ctas=int(r["ctas"]),
                meas=float(r["measured_tops"]),
                model=float(r["model_tops"]),
                dram=float(r["measured_dram_utilization_pct"]),
                mdram=float(r["model_dram_utilization_pct"]),
                occ=float(r["measured_occupancy_pct"]),
                mocc=float(r["model_occupancy_pct"]),
            )
        )
    return "\n".join(lines)


def is_plain_markdown_text(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    return not (
        stripped.startswith("#")
        or stripped.startswith("- ")
        or stripped.startswith("|")
        or stripped.startswith("!")
        or stripped.startswith("```")
    )


def normalize_markdown_paragraphs(md: str) -> str:
    lines = md.splitlines()
    out: list[str] = []
    paragraph: list[str] = []
    in_code = False

    def flush_paragraph() -> None:
        if paragraph:
            out.append(" ".join(part.strip() for part in paragraph))
            paragraph.clear()

    for line in lines:
        if line.startswith("```"):
            flush_paragraph()
            out.append(line)
            in_code = not in_code
            continue
        if in_code:
            out.append(line)
            continue
        if is_plain_markdown_text(line):
            paragraph.append(line)
            continue
        flush_paragraph()
        out.append(line)

    flush_paragraph()
    return "\n".join(out).strip() + "\n"


def markdown_report(
    problem_rows: list[dict[str, object]],
    tile_rows: list[dict[str, object]],
    compute_model_rows: list[dict[str, object]],
    compute_scheduler_rows: list[dict[str, object]],
    memory_scheduler_rows: list[dict[str, object]],
) -> str:
    problem_mape_model = mape(
        [float(r["measured_time_ns"]) for r in problem_rows],
        [float(r["model_time_ns"]) for r in problem_rows],
    )
    problem_mape_roof = mape(
        [float(r["measured_time_ns"]) for r in problem_rows],
        [float(r["roofline_time_ns"]) for r in problem_rows],
    )
    tile_mape_model = mape(
        [float(r["model_time_error_pct"]) + 100.0 for r in tile_rows],
        [100.0 for _ in tile_rows],
    )
    tile_mape_roof = mape(
        [float(r["roofline_time_error_pct"]) + 100.0 for r in tile_rows],
        [100.0 for _ in tile_rows],
    )
    problem_dram_mape = mape(
        [float(r["measured_dram_utilization_pct"]) for r in problem_rows],
        [float(r["model_dram_utilization_pct"]) for r in problem_rows],
    )
    tile_dram_mape = mape(
        [float(r["measured_dram_utilization_pct"]) for r in tile_rows],
        [float(r["model_dram_utilization_pct"]) for r in tile_rows],
    )
    compute_eligible_mape = mape(
        [float(r["measured_eligible_warps_pct"]) for r in compute_model_rows],
        [float(r["model_eligible_warps_pct"]) for r in compute_model_rows],
    )
    tile_eligible_mape = mape(
        [float(r["measured_eligible_warps_pct"]) for r in tile_rows],
        [float(r["model_eligible_warps_pct"]) for r in tile_rows],
    )
    return normalize_markdown_paragraphs(f"""# GPU Kernel Performance Model Report

## 1. Kernel Explanation

### Streaming FMA Kernel

The kernel is a streaming FMA microbenchmark. FMA means fused multiply-add: `fmaf(a, b, c)` computes `a * b + c` as one fused FP32 instruction. In this report, one FMA instruction is counted as 2 floating-point operations because it performs one multiply and one add.

Each CTA owns a contiguous tile of `blockDim.x * memory_iters` elements. Each thread loads `memory_iters` elements from three input arrays, applies a repeated FMA chain, and stores `memory_iters` output values. This creates a simple load-compute-store structure that is useful for studying how streaming DRAM traffic overlaps with scalar FP32 execution.

The core indexing is:

```cpp
block_base = blockIdx.x * blockDim.x * MemoryIters;
idx[m] = block_base + m * blockDim.x + threadIdx.x;
```

### Load Phase

The load phase computes one global index per element handled by the thread. For each `m`, all lanes in a warp access consecutive locations, so the loads from `a[idx[m]]`, `b[idx[m]]`, and `c[idx[m]]` are coalesced. The loaded values are kept in per-thread registers as `acc[m]`, `x[m]`, and `y[m]`.

```cpp
for (int m = 0; m < MemoryIters; ++m) {{
  idx[m] = block_base + m * blockDim.x + threadIdx.x;
  acc[m] = a[idx[m]];
  x[m]   = b[idx[m]];
  y[m]   = c[idx[m]];
}}
```

### Compute Phase

The compute phase repeatedly updates the loaded register values. For each element and each `compute_iters` step, the kernel executes 3 FMA instructions, or 6 FP32 operations. This phase does not load more global data, so increasing `compute_iters` increases arithmetic intensity while keeping the same logical element stream.

```cpp
for (int r = 0; r < compute_iters; ++r) {{
  for (int m = 0; m < MemoryIters; ++m) {{
    acc[m] = fmaf(acc[m], 1.0001f, x[m]);
    x[m]   = fmaf(x[m],   0.9999f, y[m]);
    y[m]   = fmaf(y[m],   1.0003f, acc[m]);
  }}
}}
```

### Store Phase

The store phase writes one output value for every element loaded by the thread. The output combines the final accumulator streams, so each logical element has 3 input loads and 1 output store at the algorithm level.

```cpp
for (int m = 0; m < MemoryIters; ++m) {{
  out[idx[m]] = acc[m] + x[m] + y[m];
}}
```

### Parameter Meaning

`compute_iters` is the main arithmetic-intensity knob. Larger values add more FMA work per loaded element, push the kernel toward the compute-bound region, and reduce the fraction of runtime explained by DRAM bandwidth.

`memory_iters` is the per-thread batching knob. Larger values make each thread load, compute, and store more elements, which reduces CTA count for a fixed `n`, increases register pressure, and changes achieved occupancy and latency hiding. The full-tile rule is `n % (threads_per_cta * memory_iters) == 0`, so every launched thread has valid work.

## 2. Model

Inputs are `n`, `compute_iters`, `memory_iters`, `threads_per_cta`, FP32 TOPS,
DRAM bandwidth, SM count, register-file size, and warp/CTA limits.

### Basic Outputs Required by Mini-Project

The required model outputs are performance in TOPS and achieved operational intensity. TOPS means tera-operations per second. Operational intensity means operations per byte.

```text
elements_per_cta = threads_per_cta * memory_iters
ctas = n / elements_per_cta
ops = n * (6 * compute_iters + 2)
algorithm_bytes = n * 16
OI_algorithm = ops / algorithm_bytes
modeled_dram_bytes = n * dram_bytes_per_element
OI_dram = ops / modeled_dram_bytes
compute_time = ops / effective_fpu_ops_per_s
dram_time = modeled_dram_bytes / effective_dram_bytes_per_s
time = max(compute_time, dram_time)
TOPS = ops / time / 1e12
```

`ops` is the total FP32 operation count. The term `6 * compute_iters` comes from 3 FMA instructions per element per compute iteration, with 2 operations per FMA. The `+ 2` term is the final output add chain, `acc + x + y`.

`algorithm_bytes` is the logical traffic from the CUDA source: 3 FP32 loads and 1 FP32 store per element, or 16 B/element. `modeled_dram_bytes` is the NCU-calibrated DRAM traffic used by the bottleneck model. It is lower here because NCU reports this kernel as mostly read-dominated DRAM traffic.

Example for `n=1032192`, `compute_iters=16`, `memory_iters=4`:

```text
ctas = 1032192 / (256 * 4) = 1008
OI_algorithm = (6 * 16 + 2) / 16 = 6.125 ops/byte
```

### Extended Outputs

The extended outputs explain why the kernel reaches a particular TOPS value. They are not required by the mini-project statement, but they make the model useful for architecture analysis.

```text
resident_ctas_per_sm = min(CTA limit, warp limit, register limit, shared-memory limit)
resident_warps_per_sm = resident_ctas_per_sm * warps_per_cta
estimated_achieved_occupancy = estimated_active_warps_per_sm / max_warps_per_sm
predicted_dram_gbps = modeled_dram_bytes / time / 1e9
dram_utilization_pct = predicted_dram_gbps / peak_dram_gbps * 100
bottleneck = compute if compute_time >= dram_time else dram
```

Warp occupancy estimates how many warps are active on each SM relative to the hardware maximum. The model first computes the theoretical resident warps from CTA, warp, register, and shared-memory limits, then applies a wave-drain factor for small grids. Register counts come from `cuobjdump`: 12, 20, 36, and 40 registers/thread for `memory_iters=1,2,4,8`.

DRAM utilization predicts how much of peak DRAM bandwidth the kernel uses. It is computed from the modeled DRAM bytes divided by the predicted runtime, then normalized by the hardware peak DRAM bandwidth.

### Definitions and Calibrated Terms

Tuned constants are small and physical. Scalar-FMA efficiency captures that this
kernel does not reach the ideal CUDA-core peak. The model uses 12 DRAM bytes per
element because NCU reports mostly read DRAM traffic for this kernel. DRAM
efficiency captures sustained streaming bandwidth. A wave-drain factor accounts
for reduced average occupancy when the grid exposes fewer CTA waves.

`effective_fpu_ops_per_s` is `fpu_tops * compute_efficiency * 1e12`. It is the scalar FP32 compute rate available to this kernel after accounting for instruction mix and issue efficiency. `effective_dram_bytes_per_s` is `dram_bw_gbps * dram_efficiency * occupancy_bw_scale * 1e9`. It is the sustained DRAM service rate available to this streaming access pattern.

`dram_bytes_per_element` is the calibrated DRAM traffic per element. `dram_efficiency` captures the gap between peak DRAM bandwidth and the sustained bandwidth reached by this coalesced streaming kernel. `occupancy_bw_scale` reduces effective bandwidth when too few active warps are available to hide memory latency.

The main challenge is that algorithm bytes are 16 B/element, but NCU DRAM bytes
are closer to 12 B/element. I address this by reporting algorithm OI separately
from the NCU-like modeled DRAM traffic. The DRAM utilization submodel uses the
same predicted runtime as the bottleneck model, so utilization falls naturally
when the kernel becomes compute-bound.

## 3. Validation

Validation used actual NCU measurements on the RTX 5080. The model remains
parameterized, so TitanV values can be substituted later.

Parameter sweeps:

- Problem size sweep: `n={PROBLEM_SIZES}`, `compute_iters=16`, `memory_iters=4`.
- Tile size sweep: `n=1032192`, `compute_iters=16`, `memory_iters=1,2,4,8`.
- Compute sweep: `n=1032192`, `memory_iters=1`, `compute_iters=0..128`.

MAPE means Mean Absolute Percentage Error. It is computed as `(100 / N) * sum(abs((predicted_i - measured_i) / measured_i))`; lower is better. Rows with zero or invalid measured values are excluded.

Problem-size runtime MAPE:

- Model: `{problem_mape_model:.2f}%`
- Peak roofline: `{problem_mape_roof:.2f}%`

Tile-size runtime MAPE:

- Model: `{tile_mape_model:.2f}%`
- Peak roofline: `{tile_mape_roof:.2f}%`

DRAM BW utilization MAPE:

- Problem-size sweep: `{problem_dram_mape:.2f}%`
- Tile-size sweep: `{tile_dram_mape:.2f}%`

Eligible-warp occupancy MAPE:

- Compute sweep: `{compute_eligible_mape:.2f}%`
- Tile-size sweep: `{tile_eligible_mape:.2f}%`

![Error vs problem size](assets/error_vs_problem_size.png)

![Error vs tile size](assets/error_vs_tile_size.png)

![Problem-size DRAM utilization](assets/dram_utilization_problem_size.png)

![Model vs NCU over compute_iters](assets/model_vs_ncu_compute_iters.png)

![Model vs NCU over memory_iters](assets/model_vs_ncu_memory_iters.png)

![Occupancy model](assets/occupancy_vs_ncu_memory_iters.png)

![DRAM utilization model](assets/dram_bw_utilization_vs_ncu.png)

### Problem Size Results

{format_problem_table(problem_rows)}

### Tile Size Results

{format_tile_table(tile_rows)}

### Scheduler Readiness

Active warp occupancy is the percentage of resident active warps on the SM, including stalled warps. Eligible warps are active warps that are ready to issue an instruction at the scheduler. Long scoreboard stall is the fraction of warp issue stall attributed to long-latency dependencies, commonly global-memory dependencies.

In the compute sweep, active occupancy stays high while eligible warps rise as `compute_iters` increases. This means the SM has many resident warps even at low `compute_iters`, but most of them are not ready to issue because long scoreboard stalls dominate. In the memory sweep, changing `memory_iters` changes CTA count and register pressure, so active occupancy and eligible readiness move together with the amount of per-thread batching.

![Compute sweep active vs eligible warps](assets/compute_iters_active_eligible_scoreboard.png)

{format_scheduler_table(compute_scheduler_rows, "compute_iters", "compute_iters")}

![Memory sweep active vs eligible warps](assets/memory_iters_active_eligible_scoreboard.png)

{format_scheduler_table(memory_scheduler_rows, "memory_iters", "memory_iters")}

### Eligible-Warp Model Extension

The eligible-warp extension estimates which active warps are actually ready to issue. The model first separates `compute_time` and `dram_time` into overlapped time and exposed bottleneck time, then predicts a readiness fraction from compute exposure plus a small batching term for larger `memory_iters`. This is not a cycle-accurate memory-latency model, but it lets the roofline model distinguish resident warps from schedulable warps.

![Eligible-warp model over compute_iters](assets/eligible_warps_vs_ncu_compute_iters.png)

{format_eligible_table(compute_model_rows, "compute_iters", "compute_iters")}

![Eligible-warp model over memory_iters](assets/eligible_warps_vs_ncu_memory_iters.png)

{format_eligible_table(tile_rows, "memory_iters", "memory_iters")}

The model performs well across memory-bound settings because it captures the
effective DRAM traffic and streaming bandwidth. The peak roofline is less
accurate: it is pessimistic in the memory-bound region because it uses algorithm
bytes, and optimistic at high `compute_iters` because it assumes ideal scalar
FMA throughput. The weakest model points are the smallest problem sizes, where
launch and wave-drain effects dominate, and the largest problem sizes, where
measured DRAM bandwidth rises above the single calibrated bandwidth.

## 4. Architecture Insight

The sensitivity analysis scales hardware parameters around the measured
configuration for `n=1032192`, `compute_iters=16`, `memory_iters=4`.

![Sensitivity analysis](assets/sensitivity_analysis.png)

This kernel is primarily a streaming memory-service workload until
`compute_iters` becomes very large. Future GPUs or DAE/WASP-like architectures
should improve:

- effective DRAM bandwidth and memory-service efficiency;
- load scheduling that reduces long scoreboard stalls;
- enough resident warps to hide memory latency;
- register capacity only when larger per-thread batching reduces occupancy.

Increasing FPU TOPS helps mostly after the kernel reaches the compute-bound
region. For the DAE/WASP direction, the most useful feature is better overlap of
loads with scalar FMA work, because the workload has simple coalesced streams
and little algorithmic L1/L2 reuse.
""")


def markdown_to_html(md: str) -> str:
    lines = md.splitlines()
    body: list[str] = []
    in_code = False
    code_lines: list[str] = []
    in_table = False
    table_lines: list[str] = []
    paragraph_lines: list[str] = []

    def flush_paragraph() -> None:
        nonlocal paragraph_lines
        if not paragraph_lines:
            return
        text = html.escape(" ".join(line.strip() for line in paragraph_lines))
        text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
        body.append(f"<p>{text}</p>")
        paragraph_lines = []

    def flush_table() -> None:
        nonlocal in_table, table_lines
        if not in_table:
            return
        header = [c.strip() for c in table_lines[0].strip("|").split("|")]
        aligns = [c.strip() for c in table_lines[1].strip("|").split("|")]
        rows = table_lines[2:]
        body.append("<table>")
        body.append("<thead><tr>" + "".join(f"<th>{html.escape(h)}</th>" for h in header) + "</tr></thead>")
        body.append("<tbody>")
        for row in rows:
            cells = [c.strip() for c in row.strip("|").split("|")]
            body.append("<tr>" + "".join(f"<td>{html.escape(c)}</td>" for c in cells) + "</tr>")
        body.append("</tbody></table>")
        in_table = False
        table_lines = []
        _ = aligns

    for line in lines:
        if line.startswith("```"):
            flush_paragraph()
            flush_table()
            if in_code:
                body.append("<pre><code>" + html.escape("\n".join(code_lines)) + "</code></pre>")
                code_lines = []
                in_code = False
            else:
                in_code = True
            continue
        if in_code:
            code_lines.append(line)
            continue
        if line.startswith("|") and line.endswith("|"):
            flush_paragraph()
            in_table = True
            table_lines.append(line)
            continue
        flush_table()
        if not line.strip():
            flush_paragraph()
            continue
        if line.startswith("# "):
            flush_paragraph()
            body.append(f"<h1>{html.escape(line[2:])}</h1>")
        elif line.startswith("## "):
            flush_paragraph()
            body.append(f"<h2>{html.escape(line[3:])}</h2>")
        elif line.startswith("### "):
            flush_paragraph()
            body.append(f"<h3>{html.escape(line[4:])}</h3>")
        elif line.startswith("- "):
            flush_paragraph()
            body.append(f"<ul><li>{html.escape(line[2:])}</li></ul>")
        elif line.startswith("!["):
            flush_paragraph()
            match = re.match(r"!\[(.*?)\]\((.*?)\)", line)
            if match:
                alt, src = match.groups()
                body.append(f'<figure><img src="{html.escape(src)}" alt="{html.escape(alt)}"><figcaption>{html.escape(alt)}</figcaption></figure>')
        else:
            paragraph_lines.append(line)
    flush_paragraph()
    flush_table()
    css = """
body { font-family: Arial, sans-serif; max-width: 980px; margin: 32px auto; color: #111; line-height: 1.45; }
h1, h2, h3 { color: #111; }
h1 { border-bottom: 2px solid #222; padding-bottom: 8px; }
pre { background: #f4f4f4; padding: 12px; overflow-x: auto; border-radius: 4px; }
code { background: #f4f4f4; padding: 1px 3px; border-radius: 3px; }
table { border-collapse: collapse; width: 100%; margin: 12px 0 20px; font-size: 13px; }
th, td { border: 1px solid #ccc; padding: 6px 8px; text-align: right; }
th:first-child, td:first-child { text-align: left; }
th { background: #eee; }
figure { margin: 18px 0; page-break-inside: avoid; }
img { max-width: 100%; border: 1px solid #ddd; }
figcaption { font-size: 12px; color: #555; text-align: center; }
ul { margin-top: 0; }
@media print { body { margin: 18mm; } h2 { page-break-after: avoid; } }
"""
    return "<!doctype html><html><head><meta charset='utf-8'><title>GPU Kernel Performance Model Report</title><style>" + css + "</style></head><body>" + "\n".join(body) + "</body></html>\n"


def write_pdf_from_html() -> None:
    html_path = REPORT_DIR / "report.html"
    pdf_path = REPORT_DIR / "report.pdf"
    run_logged(
        [
            CHROME,
            "--headless",
            "--disable-gpu",
            "--no-sandbox",
            "--no-pdf-header-footer",
            "--print-to-pdf-no-header",
            f"--print-to-pdf={pdf_path}",
            str(html_path),
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh-ncu", action="store_true")
    args = parser.parse_args()

    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    run_logged(["make", "-C", str(V6_DIR), "all"])
    problem_rows = collect_problem_size_data(args.refresh_ncu)
    tile_rows = load_tile_data()
    compute_model_rows = load_compute_model_data()
    compute_scheduler_rows = load_scheduler_sweep(
        SWEEP_DIR / "compute_iters_sweep_memory_iters_1.csv", "compute_iters"
    )
    memory_scheduler_rows = load_scheduler_sweep(
        SWEEP_DIR / "memory_iters_sweep_compute_iters_16.csv", "memory_iters"
    )

    plot_problem_error(problem_rows)
    plot_tile_error(tile_rows)
    plot_problem_dram_utilization(problem_rows)
    plot_sensitivity()
    copy_model_plots()
    copy_scheduler_plots()

    md = markdown_report(
        problem_rows,
        tile_rows,
        compute_model_rows,
        compute_scheduler_rows,
        memory_scheduler_rows,
    )
    (REPORT_DIR / "report.md").write_text(md)
    (REPORT_DIR / "report.html").write_text(markdown_to_html(md))
    write_pdf_from_html()


if __name__ == "__main__":
    main()
