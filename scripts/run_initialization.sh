#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 3 ]; then
    echo "Usage: $0 <vggt|omega> <scene_dir> <checkpoint>"
    exit 1
fi

METHOD="$1"
SCENE_DIR="$(realpath "$2")"
CHECKPOINT="$(realpath "$3")"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

VGGTPY="${VGGTPY:-python}"
GPU="${GPU:-0}"

export VGGT_REPO="${VGGT_REPO:-$PROJECT_ROOT/third_party/repos/vggt}"
export VGGT_OMEGA_REPO="${VGGT_OMEGA_REPO:-$PROJECT_ROOT/third_party/repos/vggt-omega}"

if [ ! -d "$SCENE_DIR/images" ]; then
    echo "ERROR: Missing image directory: $SCENE_DIR/images"
    exit 1
fi

if [ ! -f "$CHECKPOINT" ]; then
    echo "ERROR: Missing checkpoint: $CHECKPOINT"
    exit 1
fi

case "$METHOD" in
    vggt)
        CUDA_VISIBLE_DEVICES="$GPU" "$VGGTPY" \
            "$PROJECT_ROOT/src/vggt/demo_colmap_offline.py" \
            --scene_dir "$SCENE_DIR" \
            --checkpoint "$CHECKPOINT" \
            --conf_thres_value "${CONF_THRESHOLD:-1.0}"
        ;;

    omega)
        CUDA_VISIBLE_DEVICES="$GPU" "$VGGTPY" \
            "$PROJECT_ROOT/src/vggt_omega/demo_colmap_omega_offline.py" \
            --scene_dir "$SCENE_DIR" \
            --checkpoint "$CHECKPOINT" \
            --conf_percentile "${CONF_PERCENTILE:-50}" \
            --max_points "${MAX_POINTS:-100000}"
        ;;

    *)
        echo "ERROR: Method must be either 'vggt' or 'omega'."
        exit 1
        ;;
esac
