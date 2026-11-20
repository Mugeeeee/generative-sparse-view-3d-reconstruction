#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <scene_dir> <output_dir> [iterations]"
    exit 1
fi

SCENE_DIR="$(realpath "$1")"
OUTPUT_DIR="$2"
ITERATIONS="${3:-30000}"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

GSPY="${GSPY:-python}"
GS_REPO="${GS_REPO:-$PROJECT_ROOT/third_party/repos/gaussian-splatting}"
GPU="${GPU:-0}"

if [ ! -f "$GS_REPO/train.py" ]; then
    echo "ERROR: Gaussian Splatting repository not found at $GS_REPO"
    echo "Set GS_REPO or run scripts/clone_dependencies.sh."
    exit 1
fi

if [ ! -d "$SCENE_DIR/images" ] || [ ! -d "$SCENE_DIR/sparse/0" ]; then
    echo "ERROR: Scene must contain images/ and sparse/0/."
    exit 1
fi

mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR="$(realpath "$OUTPUT_DIR")"

cd "$GS_REPO"

CUDA_VISIBLE_DEVICES="$GPU" "$GSPY" train.py \
    -s "$SCENE_DIR" \
    -m "$OUTPUT_DIR" \
    --eval \
    --iterations "$ITERATIONS" \
    --test_iterations "$ITERATIONS" \
    --save_iterations "$ITERATIONS" \
    --checkpoint_iterations "$ITERATIONS" \
    --disable_viewer
