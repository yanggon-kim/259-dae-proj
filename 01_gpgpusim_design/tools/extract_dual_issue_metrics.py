#!/usr/bin/env python3
"""Extract headline WS_BASE / DUAL_ISSUE_PC metrics from GPGPU-Sim logs."""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path


PATTERNS = {
    "cycles": re.compile(r"(?:gpu_sim_cycle|gpu_tot_sim_cycle) = (\d+)"),
    "instructions": re.compile(r"gpu_sim_insn = (\d+)"),
    "issued_cta": re.compile(r"gpu_tot_issued_cta = (\d+)"),
    "occupancy_pct": re.compile(r"gpu_tot_occupancy = ([0-9.]+)%"),
    "pair_attempts": re.compile(r"pc_pair_attempts = (\d+)"),
    "pair_successes": re.compile(r"pc_pair_success = (\d+)"),
    "scoreboard_failures": re.compile(r"pc_pair_fail_scoreboard = (\d+)"),
    "no_producer_windows": re.compile(r"pc_pair_fail_no_producer = (\d+)"),
    "no_consumer_windows": re.compile(r"pc_pair_fail_no_consumer = (\d+)"),
    "regset_busy_failures": re.compile(r"pc_pair_fail_regset_busy = (\d+)"),
    "pipeline_busy_failures": re.compile(r"pc_pair_fail_pipeline_busy = (\d+)"),
    "both_ready_cycles": re.compile(r"pc_pair_diag_both_ready_cycles = (\d+)"),
    "verification": re.compile(r"verification=([A-Z]+)"),
}


def parse_log(path: Path) -> dict[str, str]:
    text = path.read_text(errors="replace")
    row = {"log": str(path)}
    for key, pattern in PATTERNS.items():
        match = pattern.search(text)
        if match:
            row[key] = match.group(1)
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("logs", nargs="+", type=Path)
    args = parser.parse_args()

    rows = [parse_log(path) for path in args.logs]
    keys = ["log"] + sorted({key for row in rows for key in row if key != "log"})
    writer = csv.DictWriter(sys.stdout, fieldnames=keys, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow(row)


if __name__ == "__main__":
    main()
