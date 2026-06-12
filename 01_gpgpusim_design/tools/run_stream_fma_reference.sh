#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 <rfq|smemq|rfq-dual|smemq-dual>" >&2
    exit 2
fi

CASE=$1
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
DESIGN_DIR=$(realpath "$SCRIPT_DIR/..")
KERNEL_DIR="$DESIGN_DIR/kernels/stream_fma_reference"
RUN_ROOT="$DESIGN_DIR/runs"

case "$CASE" in
    rfq)
        CONFIG_DIR="$DESIGN_DIR/configs/rfq_based_depth32"
        PTX="$DESIGN_DIR/ptx_overrides/rfq_based/stream_fma_v6_rfq_based.ptx"
        PTXINFO="$DESIGN_DIR/ptx_overrides/rfq_based/stream_fma_v6_rfq_based.ptxinfo"
        ;;
    smemq)
        CONFIG_DIR="$DESIGN_DIR/configs/smemq_based_depth32"
        PTX="$DESIGN_DIR/ptx_overrides/smemq_based/stream_fma_v6_smemq_based.ptx"
        PTXINFO="$DESIGN_DIR/ptx_overrides/smemq_based/stream_fma_v6_smemq_based.ptxinfo"
        ;;
    rfq-dual)
        CONFIG_DIR="$DESIGN_DIR/configs/rfq_dual_issue_depth32"
        PTX="$DESIGN_DIR/ptx_overrides/rfq_based/stream_fma_v6_rfq_based.ptx"
        PTXINFO="$DESIGN_DIR/ptx_overrides/rfq_based/stream_fma_v6_rfq_based.ptxinfo"
        ;;
    smemq-dual)
        CONFIG_DIR="$DESIGN_DIR/configs/smemq_dual_issue_depth32"
        PTX="$DESIGN_DIR/ptx_overrides/smemq_based/stream_fma_v6_smemq_based.ptx"
        PTXINFO="$DESIGN_DIR/ptx_overrides/smemq_based/stream_fma_v6_smemq_based.ptxinfo"
        ;;
    *)
        echo "error: unknown case: $CASE" >&2
        exit 2
        ;;
esac

if [[ ! -x "$KERNEL_DIR/stream_fma_v6_v2_m4_c128_cta512" ]]; then
    make -C "$KERNEL_DIR" NVCC="${NVCC:-/usr/local/cuda-11.7/bin/nvcc}" ARCH="${ARCH:-sm_70}"
fi

RUN_DIR="$RUN_ROOT/$CASE"
mkdir -p "$RUN_DIR"
cp -a "$CONFIG_DIR"/. "$RUN_DIR"/
cp "$DESIGN_DIR"/configs/h200_132sm_mshr512/accelwattch*.xml "$RUN_DIR"/

cd "$RUN_DIR"
PTX_SIM_USE_PTX_FILE=1 \
PTX_SIM_KERNELFILE="$PTX" \
PTX_SIM_PTXINFO_FILE="$PTXINFO" \
"$KERNEL_DIR/stream_fma_v6_v2_m4_c128_cta512" \
  --n 524288 --iters 128 --memory-iters 4 --warmup 0 --repeats 1 \
  > full_run.log 2>&1

echo "wrote $RUN_DIR/full_run.log"
grep -E "verification=|gpu_sim_cycle|eligible_warp_occupancy|pc_coissue_success|smemq_shared_bytes_per_cta|wasp_rfq_adjusted_max_cta" full_run.log || true
