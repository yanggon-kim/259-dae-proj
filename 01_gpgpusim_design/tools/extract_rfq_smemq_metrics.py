#!/usr/bin/env python3
"""Extract RFQ/SMEMQ headline metrics from GPGPU-Sim full_run.log files."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


PATTERNS = {
    "cycles": re.compile(r"gpu_tot_sim_cycle = (\d+)"),
    "eligible_occ_pct": re.compile(r"eligible_warp_occupancy = ([0-9.]+)%"),
    "avg_eligible_warps_per_scheduler": re.compile(
        r"avg_eligible_warps_per_scheduler = ([0-9.]+)"
    ),
    "verification": re.compile(r"verification=([A-Z]+)"),
    "rfq_cta_per_sm": re.compile(r"wasp_rfq_adjusted_max_cta = (\d+)"),
    "smemq_cta_per_sm": re.compile(r"smemq_adjusted_max_cta = (\d+)"),
    "rfq_active_occ_pct": re.compile(
        r"active_warp_occupancy_after_rfq = ([0-9.]+)%"
    ),
    "smemq_active_occ_pct": re.compile(
        r"active_warp_occupancy_after_smemq = ([0-9.]+)%"
    ),
    "rfq_register_slots_per_cta": re.compile(
        r"wasp_rfq_register_slots_per_cta = (\d+)"
    ),
    "smemq_shared_bytes_per_cta": re.compile(r"smemq_shared_bytes_per_cta = (\d+)"),
}


def parse_log(path: Path) -> dict[str, str]:
    text = path.read_text(errors="replace")
    row: dict[str, str] = {"log": str(path)}
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
    print(",".join(keys))
    for row in rows:
        print(",".join(row.get(key, "") for key in keys))


if __name__ == "__main__":
    main()
