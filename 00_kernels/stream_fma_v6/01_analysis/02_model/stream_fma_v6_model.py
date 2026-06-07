#!/usr/bin/env python3
"""Mechanistic performance model for stream_fma_v6.

Usage examples:
  python3 stream_fma_v6_model.py --n 1032192 --iters 16 --memory-iters 4
  python3 stream_fma_v6_model.py --n 1032192 --iters 128 --memory-iters 1
  python3 stream_fma_v6_model.py --n 1032192 --iters 16 --memory-iters 4 \
      --fpu-tops 55.9104 --dram-gbps 941.44

Inputs:
  --n is the total number of elements processed by the kernel.
  --iters is compute_iters, the number of repeated FMA compute iterations.
  --memory-iters is the number of elements handled by each thread.
  --threads is threads_per_cta and defaults to 256.

Required input rule:
  memory_iters must be 1, 2, 4, or 8.
  n must be divisible by threads_per_cta * memory_iters.
  For the default 256-thread CTA, n must be divisible by 256*memory_iters.

Output:
  The script prints JSON with operation count, arithmetic intensity,
  predicted TOPS, predicted bottleneck, DRAM utilization, CTA mapping,
  occupancy estimates, compute/memory overlap estimates, and eligible-warp
  scheduler-readiness estimates.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass


SUPPORTED_MEMORY_ITERS = (1, 2, 4, 8)
DEFAULT_REGISTERS_PER_THREAD = {
    1: 12,
    2: 20,
    4: 36,
    8: 40,
}


@dataclass(frozen=True)
class KernelParams:
    n: int
    compute_iters: int
    memory_iters: int
    threads_per_cta: int = 256

    def validate(self) -> None:
        if self.n <= 0:
            raise ValueError("n must be positive")
        if self.compute_iters < 0:
            raise ValueError("compute_iters must be non-negative")
        if self.memory_iters not in SUPPORTED_MEMORY_ITERS:
            raise ValueError("memory_iters must be one of 1, 2, 4, 8")
        if self.threads_per_cta <= 0 or self.threads_per_cta % 32 != 0:
            raise ValueError("threads_per_cta must be a positive multiple of 32")
        elements_per_cta = self.threads_per_cta * self.memory_iters
        if self.n % elements_per_cta != 0:
            raise ValueError(
                "stream_fma_v6 uses full CTA tiles: require "
                f"n % (threads_per_cta * memory_iters) == 0, got "
                f"{self.n} % {elements_per_cta}"
            )


@dataclass(frozen=True)
class HardwareParams:
    sm_count: int = 84
    fpu_tops: float = 55.9104
    dram_bw_gbps: float = 941.44
    max_warps_per_sm: int = 48
    max_ctas_per_sm: int = 24
    registers_per_sm: int = 65536
    register_alloc_unit_per_thread: int = 8
    shared_mem_bytes_per_sm: int = 228 * 1024
    warp_size: int = 32
    smsps_per_sm: int = 4

    # Calibrated physical-efficiency terms for this streaming scalar-FMA kernel.
    compute_efficiency: float = 0.69
    dram_efficiency: float = 0.855
    dram_bytes_per_element: float = 12.0
    occupancy_efficiency_cap: float = 0.87
    wave_drain_waves: float = 0.25
    warps_for_full_dram_bw: float = 32.0
    eligible_floor: float = 0.014
    eligible_compute_gain: float = 0.54
    eligible_compute_alpha: float = 1.75
    eligible_memory_batch_gain: float = 0.015


@dataclass(frozen=True)
class KernelResources:
    registers_per_thread: int
    shared_mem_bytes_per_cta: int = 0

    @staticmethod
    def for_memory_iters(memory_iters: int) -> "KernelResources":
        if memory_iters not in DEFAULT_REGISTERS_PER_THREAD:
            raise ValueError("memory_iters must be one of 1, 2, 4, 8")
        return KernelResources(
            registers_per_thread=DEFAULT_REGISTERS_PER_THREAD[memory_iters],
            shared_mem_bytes_per_cta=0,
        )


@dataclass(frozen=True)
class Mapping:
    elements_per_cta: int
    ctas: int
    warps_per_cta: int
    total_warps: int
    full_resident_waves: float


@dataclass(frozen=True)
class OccupancyPrediction:
    rounded_registers_per_thread: int
    registers_per_cta: int
    block_limit_ctas_per_sm: int
    warp_limit_ctas_per_sm: int
    register_limit_ctas_per_sm: int
    shared_mem_limit_ctas_per_sm: int
    resident_ctas_per_sm: int
    resident_warps_per_sm: int
    theoretical_occupancy: float
    estimated_active_warps_per_sm: float
    estimated_achieved_occupancy: float


@dataclass(frozen=True)
class SchedulerPrediction:
    overlapped_time_s: float
    exposed_memory_time_s: float
    exposed_compute_time_s: float
    compute_exposure_fraction: float
    eligible_ready_fraction: float
    estimated_eligible_warps_per_sm: float
    estimated_eligible_warps_per_scheduler_cycle: float
    estimated_eligible_warps_pct: float


@dataclass(frozen=True)
class Prediction:
    kernel: KernelParams
    hardware: HardwareParams
    resources: KernelResources
    mapping: Mapping
    occupancy: OccupancyPrediction
    scheduler: SchedulerPrediction
    ops: float
    algorithm_bytes: float
    model_dram_bytes: float
    operational_intensity_algorithm: float
    operational_intensity_dram: float
    compute_time_s: float
    dram_time_s: float
    predicted_time_s: float
    predicted_tops: float
    bottleneck: str
    effective_fpu_tops: float
    effective_dram_bw_gbps: float
    predicted_dram_gbps: float
    predicted_dram_utilization_pct: float
    effective_dram_utilization_pct: float


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def round_up(value: int, unit: int) -> int:
    if unit <= 0:
        return value
    return ceil_div(value, unit) * unit


def make_peak_roofline_hardware(hw: HardwareParams) -> HardwareParams:
    return HardwareParams(
        sm_count=hw.sm_count,
        fpu_tops=hw.fpu_tops,
        dram_bw_gbps=hw.dram_bw_gbps,
        max_warps_per_sm=hw.max_warps_per_sm,
        max_ctas_per_sm=hw.max_ctas_per_sm,
        registers_per_sm=hw.registers_per_sm,
        register_alloc_unit_per_thread=hw.register_alloc_unit_per_thread,
        shared_mem_bytes_per_sm=hw.shared_mem_bytes_per_sm,
        warp_size=hw.warp_size,
        smsps_per_sm=hw.smsps_per_sm,
        compute_efficiency=1.0,
        dram_efficiency=1.0,
        dram_bytes_per_element=16.0,
        occupancy_efficiency_cap=1.0,
        wave_drain_waves=0.0,
        warps_for_full_dram_bw=1.0,
        eligible_floor=hw.eligible_floor,
        eligible_compute_gain=hw.eligible_compute_gain,
        eligible_compute_alpha=hw.eligible_compute_alpha,
        eligible_memory_batch_gain=hw.eligible_memory_batch_gain,
    )


def predict_mapping(kernel: KernelParams, hw: HardwareParams) -> Mapping:
    kernel.validate()
    elements_per_cta = kernel.threads_per_cta * kernel.memory_iters
    ctas = kernel.n // elements_per_cta
    warps_per_cta = kernel.threads_per_cta // hw.warp_size
    total_warps = ctas * warps_per_cta
    # Number of complete resident waves the grid can provide to all SMs.
    resident_ctas_by_warps = max(1, hw.max_warps_per_sm // warps_per_cta)
    full_resident_waves = ctas / max(1, hw.sm_count * resident_ctas_by_warps)
    return Mapping(
        elements_per_cta=elements_per_cta,
        ctas=ctas,
        warps_per_cta=warps_per_cta,
        total_warps=total_warps,
        full_resident_waves=full_resident_waves,
    )


def predict_occupancy(
    kernel: KernelParams, hw: HardwareParams, resources: KernelResources
) -> OccupancyPrediction:
    mapping = predict_mapping(kernel, hw)
    rounded_regs = round_up(
        resources.registers_per_thread, hw.register_alloc_unit_per_thread
    )
    registers_per_cta = rounded_regs * kernel.threads_per_cta
    register_limit = (
        hw.registers_per_sm // registers_per_cta if registers_per_cta else hw.max_ctas_per_sm
    )
    shared_limit = (
        hw.shared_mem_bytes_per_sm // resources.shared_mem_bytes_per_cta
        if resources.shared_mem_bytes_per_cta
        else hw.max_ctas_per_sm
    )
    block_limit = hw.max_ctas_per_sm
    warp_limit = hw.max_warps_per_sm // mapping.warps_per_cta
    resident_ctas = max(
        1, min(block_limit, warp_limit, register_limit, shared_limit)
    )
    resident_warps = resident_ctas * mapping.warps_per_cta
    theoretical = resident_warps / hw.max_warps_per_sm

    full_waves = mapping.ctas / max(1, hw.sm_count * resident_ctas)
    drain_factor = (
        full_waves / (full_waves + hw.wave_drain_waves)
        if full_waves > 0.0
        else 0.0
    )
    achieved = min(
        theoretical,
        theoretical * hw.occupancy_efficiency_cap * min(1.0, drain_factor),
    )
    active_warps = achieved * hw.max_warps_per_sm

    return OccupancyPrediction(
        rounded_registers_per_thread=rounded_regs,
        registers_per_cta=registers_per_cta,
        block_limit_ctas_per_sm=block_limit,
        warp_limit_ctas_per_sm=warp_limit,
        register_limit_ctas_per_sm=register_limit,
        shared_mem_limit_ctas_per_sm=shared_limit,
        resident_ctas_per_sm=resident_ctas,
        resident_warps_per_sm=resident_warps,
        theoretical_occupancy=theoretical,
        estimated_active_warps_per_sm=active_warps,
        estimated_achieved_occupancy=achieved,
    )


def predict_dram_bw_utilization(
    model_dram_bytes: float,
    predicted_time_s: float,
    hw: HardwareParams,
    effective_dram_bw_gbps: float,
) -> tuple[float, float, float]:
    if predicted_time_s <= 0.0:
        return 0.0, 0.0, 0.0
    predicted_dram_gbps = model_dram_bytes / predicted_time_s / 1.0e9
    peak_utilization = (
        predicted_dram_gbps / hw.dram_bw_gbps * 100.0 if hw.dram_bw_gbps else 0.0
    )
    effective_utilization = (
        predicted_dram_gbps / effective_dram_bw_gbps * 100.0
        if effective_dram_bw_gbps
        else 0.0
    )
    return predicted_dram_gbps, peak_utilization, effective_utilization


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def predict_scheduler(
    kernel: KernelParams,
    hw: HardwareParams,
    occupancy: OccupancyPrediction,
    compute_time_s: float,
    dram_time_s: float,
    predicted_time_s: float,
) -> SchedulerPrediction:
    overlapped_time_s = min(compute_time_s, dram_time_s)
    exposed_memory_time_s = max(0.0, dram_time_s - compute_time_s)
    exposed_compute_time_s = max(0.0, compute_time_s - dram_time_s)
    compute_exposure_fraction = (
        compute_time_s / predicted_time_s if predicted_time_s > 0.0 else 0.0
    )
    memory_batch_relief = max(0.0, math.log2(float(kernel.memory_iters)) - 2.0)
    eligible_ready_fraction = clamp(
        hw.eligible_floor
        + hw.eligible_compute_gain
        * (compute_exposure_fraction ** hw.eligible_compute_alpha)
        + hw.eligible_memory_batch_gain * memory_batch_relief,
        0.0,
        1.0,
    )
    estimated_eligible_warps_per_sm = (
        occupancy.estimated_active_warps_per_sm * eligible_ready_fraction
    )
    estimated_eligible_warps_per_scheduler_cycle = (
        estimated_eligible_warps_per_sm / hw.smsps_per_sm if hw.smsps_per_sm else 0.0
    )
    estimated_eligible_warps_pct = (
        estimated_eligible_warps_per_sm / hw.max_warps_per_sm * 100.0
        if hw.max_warps_per_sm
        else 0.0
    )
    return SchedulerPrediction(
        overlapped_time_s=overlapped_time_s,
        exposed_memory_time_s=exposed_memory_time_s,
        exposed_compute_time_s=exposed_compute_time_s,
        compute_exposure_fraction=compute_exposure_fraction,
        eligible_ready_fraction=eligible_ready_fraction,
        estimated_eligible_warps_per_sm=estimated_eligible_warps_per_sm,
        estimated_eligible_warps_per_scheduler_cycle=estimated_eligible_warps_per_scheduler_cycle,
        estimated_eligible_warps_pct=estimated_eligible_warps_pct,
    )


def predict(
    kernel: KernelParams,
    hw: HardwareParams,
    resources: KernelResources | None = None,
) -> Prediction:
    kernel.validate()
    if resources is None:
        resources = KernelResources.for_memory_iters(kernel.memory_iters)

    mapping = predict_mapping(kernel, hw)
    occupancy = predict_occupancy(kernel, hw, resources)
    ops = float(kernel.n) * (6.0 * float(kernel.compute_iters) + 2.0)
    algorithm_bytes = float(kernel.n) * 16.0
    model_dram_bytes = float(kernel.n) * hw.dram_bytes_per_element

    effective_fpu_tops = hw.fpu_tops * hw.compute_efficiency
    occupancy_bw_scale = min(
        1.0, occupancy.estimated_active_warps_per_sm / hw.warps_for_full_dram_bw
    )
    effective_dram_bw_gbps = hw.dram_bw_gbps * hw.dram_efficiency * occupancy_bw_scale

    compute_time_s = ops / (effective_fpu_tops * 1.0e12)
    dram_time_s = model_dram_bytes / (effective_dram_bw_gbps * 1.0e9)
    predicted_time_s = max(compute_time_s, dram_time_s)
    predicted_tops = ops / predicted_time_s / 1.0e12
    bottleneck = "compute" if compute_time_s >= dram_time_s else "dram"
    scheduler = predict_scheduler(
        kernel=kernel,
        hw=hw,
        occupancy=occupancy,
        compute_time_s=compute_time_s,
        dram_time_s=dram_time_s,
        predicted_time_s=predicted_time_s,
    )
    (
        predicted_dram_gbps,
        predicted_dram_utilization_pct,
        effective_dram_utilization_pct,
    ) = predict_dram_bw_utilization(
        model_dram_bytes=model_dram_bytes,
        predicted_time_s=predicted_time_s,
        hw=hw,
        effective_dram_bw_gbps=effective_dram_bw_gbps,
    )

    return Prediction(
        kernel=kernel,
        hardware=hw,
        resources=resources,
        mapping=mapping,
        occupancy=occupancy,
        scheduler=scheduler,
        ops=ops,
        algorithm_bytes=algorithm_bytes,
        model_dram_bytes=model_dram_bytes,
        operational_intensity_algorithm=ops / algorithm_bytes,
        operational_intensity_dram=ops / model_dram_bytes,
        compute_time_s=compute_time_s,
        dram_time_s=dram_time_s,
        predicted_time_s=predicted_time_s,
        predicted_tops=predicted_tops,
        bottleneck=bottleneck,
        effective_fpu_tops=effective_fpu_tops,
        effective_dram_bw_gbps=effective_dram_bw_gbps,
        predicted_dram_gbps=predicted_dram_gbps,
        predicted_dram_utilization_pct=predicted_dram_utilization_pct,
        effective_dram_utilization_pct=effective_dram_utilization_pct,
    )


def prediction_to_dict(prediction: Prediction) -> dict[str, object]:
    out = {
        "kernel": asdict(prediction.kernel),
        "hardware": asdict(prediction.hardware),
        "resources": asdict(prediction.resources),
        "mapping": asdict(prediction.mapping),
        "occupancy": asdict(prediction.occupancy),
        "scheduler": asdict(prediction.scheduler),
        "ops": prediction.ops,
        "algorithm_bytes": prediction.algorithm_bytes,
        "model_dram_bytes": prediction.model_dram_bytes,
        "operational_intensity_algorithm": prediction.operational_intensity_algorithm,
        "operational_intensity_dram": prediction.operational_intensity_dram,
        "compute_time_s": prediction.compute_time_s,
        "dram_time_s": prediction.dram_time_s,
        "predicted_time_s": prediction.predicted_time_s,
        "predicted_tops": prediction.predicted_tops,
        "bottleneck": prediction.bottleneck,
        "effective_fpu_tops": prediction.effective_fpu_tops,
        "effective_dram_bw_gbps": prediction.effective_dram_bw_gbps,
        "predicted_dram_gbps": prediction.predicted_dram_gbps,
        "predicted_dram_utilization_pct": prediction.predicted_dram_utilization_pct,
        "effective_dram_utilization_pct": prediction.effective_dram_utilization_pct,
    }
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=1_032_192)
    parser.add_argument("--iters", type=int, default=16, dest="compute_iters")
    parser.add_argument("--memory-iters", type=int, default=4)
    parser.add_argument("--threads", type=int, default=256, dest="threads_per_cta")
    parser.add_argument("--fpu-tops", type=float, default=55.9104)
    parser.add_argument("--dram-gbps", type=float, default=941.44)
    parser.add_argument("--peak-roofline", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    kernel = KernelParams(
        n=args.n,
        compute_iters=args.compute_iters,
        memory_iters=args.memory_iters,
        threads_per_cta=args.threads_per_cta,
    )
    hw = HardwareParams(fpu_tops=args.fpu_tops, dram_bw_gbps=args.dram_gbps)
    if args.peak_roofline:
        hw = make_peak_roofline_hardware(hw)
    resources = KernelResources.for_memory_iters(args.memory_iters)
    try:
        prediction = predict(kernel, hw, resources)
    except ValueError as exc:
        print(f"model input error: {exc}", file=sys.stderr)
        raise SystemExit(1) from None
    print(json.dumps(prediction_to_dict(prediction), indent=2))


if __name__ == "__main__":
    main()
