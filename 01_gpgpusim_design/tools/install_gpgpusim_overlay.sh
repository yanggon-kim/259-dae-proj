#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 <gpgpu-sim_distribution-root>" >&2
    exit 2
fi

TARGET_ROOT=$(realpath "$1")
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
DESIGN_DIR=$(realpath "$SCRIPT_DIR/..")
OVERLAY_DIR="$DESIGN_DIR/gpgpusim_overlay"

if [[ ! -d "$TARGET_ROOT/src/gpgpu-sim" || ! -d "$TARGET_ROOT/src/cuda-sim" ]]; then
    echo "error: target does not look like a GPGPU-Sim source tree: $TARGET_ROOT" >&2
    exit 1
fi

while IFS= read -r -d '' src; do
    rel=${src#"$OVERLAY_DIR"/}
    dst="$TARGET_ROOT/$rel"
    mkdir -p "$(dirname "$dst")"
    cp "$src" "$dst"
    echo "installed $rel"
done < <(find "$OVERLAY_DIR" -type f -print0 | sort -z)

cat <<EOF

Overlay installed into:
  $TARGET_ROOT

Rebuild with:
  export CUDA_INSTALL_PATH=/usr/local/cuda-11.7
  cd $TARGET_ROOT
  source setup_environment release
  make -j\$(nproc)
EOF
