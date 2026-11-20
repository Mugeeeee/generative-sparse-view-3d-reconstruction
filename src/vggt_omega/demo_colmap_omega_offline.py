#!/usr/bin/env python3
"""Offline VGGT-Omega inference and COLMAP export for 3DGS initialization."""

from __future__ import annotations

import argparse
import glob
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import trimesh
from PIL import Image


# Locate separately cloned upstream repositories.
THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[1]

VGGT_REPO = Path(
    os.environ.get(
        "VGGT_REPO",
        PROJECT_ROOT / "third_party" / "repos" / "vggt",
    )
).resolve()

VGGT_OMEGA_REPO = Path(
    os.environ.get(
        "VGGT_OMEGA_REPO",
        PROJECT_ROOT / "third_party" / "repos" / "vggt-omega",
    )
).resolve()

if not (VGGT_REPO / "vggt").is_dir():
    raise FileNotFoundError(
        f"VGGT repository not found at {VGGT_REPO}. "
        "Set VGGT_REPO to the cloned VGGT repository."
    )

if not (VGGT_OMEGA_REPO / "vggt_omega").is_dir():
    raise FileNotFoundError(
        f"VGGT-Omega repository not found at {VGGT_OMEGA_REPO}. "
        "Set VGGT_OMEGA_REPO to the cloned VGGT-Omega repository."
    )

sys.path.insert(0, str(VGGT_REPO))
sys.path.insert(0, str(VGGT_OMEGA_REPO))

from vggt.dependency.np_to_pycolmap import batch_np_matrix_to_pycolmap_wo_track
from vggt.utils.geometry import unproject_depth_map_to_point_map
from vggt.utils.helper import create_pixel_coordinate_grid
from vggt_omega.models import VGGTOmega
from vggt_omega.utils.load_fn import load_and_preprocess_images
from vggt_omega.utils.pose_enc import encoding_to_camera




def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run VGGT-Omega offline and export a COLMAP sparse model."
    )
    parser.add_argument("--scene_dir", required=True)
    parser.add_argument("--checkpoint", required=True, help="Path to the local VGGT-Omega checkpoint")
    parser.add_argument("--image_resolution", type=int, default=416)
    parser.add_argument(
        "--conf_percentile",
        type=float,
        default=50.0,
        help="Keep points at or above this depth-confidence percentile.",
    )
    parser.add_argument("--max_points", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def image_sizes(image_paths: list[str]) -> np.ndarray:
    sizes = []
    for path in image_paths:
        with Image.open(path) as image:
            width, height = image.size
        aspect = height / max(width, 1)
        if not 0.5 <= aspect <= 2.0:
            raise ValueError(
                f"Unsupported extreme aspect ratio for {path}: {width}x{height}. "
                "This exporter does not yet undo Omega's center crop."
            )
        sizes.append((width, height))
    return np.asarray(sizes, dtype=np.int64)


def rescale_reconstruction_to_original_images(
    reconstruction,
    image_names: list[str],
    original_sizes: np.ndarray,
    processed_width: int,
    processed_height: int,
):
    for image_id in reconstruction.images:
        image = reconstruction.images[image_id]
        index = image_id - 1
        image.name = image_names[index]

        camera = reconstruction.cameras[image.camera_id]
        original_width, original_height = original_sizes[index]
        scale_x = original_width / processed_width
        scale_y = original_height / processed_height

        params = np.asarray(camera.params).copy()
        if len(params) != 4:
            raise ValueError("Expected a four-parameter PINHOLE camera")
        params[0] *= scale_x  # fx
        params[1] *= scale_y  # fy
        params[2] *= scale_x  # cx
        params[3] *= scale_y  # cy
        camera.params = params
        camera.width = int(original_width)
        camera.height = int(original_height)

        for point2d in image.points2D:
            point2d.xy = np.asarray(point2d.xy) * np.array([scale_x, scale_y])

    return reconstruction


def main() -> None:
    args = parse_args()
    if not 0 <= args.conf_percentile <= 100:
        raise ValueError("--conf_percentile must be between 0 and 100")
    if args.max_points <= 0:
        raise ValueError("--max_points must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    scene_dir = Path(args.scene_dir).resolve()
    image_dir = scene_dir / "images"
    image_paths = sorted(
        path for path in glob.glob(str(image_dir / "*")) if os.path.isfile(path)
    )
    if not image_paths:
        raise ValueError(f"No images found in {image_dir}")
    image_names = [Path(path).name for path in image_paths]
    original_sizes = image_sizes(image_paths)

    print(f"Loading {len(image_paths)} images with Omega resolution {args.image_resolution}...")
    images = load_and_preprocess_images(
        image_paths,
        image_resolution=args.image_resolution,
    ).cuda()
    processed_height, processed_width = images.shape[-2:]
    print(f"Processed tensor shape: {tuple(images.shape)}")

    print(f"Loading checkpoint: {args.checkpoint}")
    model = VGGTOmega()
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    del state
    model = model.cuda().eval()

    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    with torch.inference_mode():
        predictions = model(images)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    extrinsics, intrinsics = encoding_to_camera(
        predictions["pose_enc"], predictions["images"].shape[-2:]
    )
    extrinsics = extrinsics.squeeze(0).detach().float().cpu().numpy()
    intrinsics = intrinsics.squeeze(0).detach().float().cpu().numpy()
    depth = predictions["depth"].squeeze(0).detach().float().cpu().numpy()
    confidence = (
        predictions["depth_conf"].squeeze(0).detach().float().cpu().numpy()
    )
    rgb = (
        predictions["images"]
        .squeeze(0)
        .detach()
        .float()
        .cpu()
        .numpy()
        .transpose(0, 2, 3, 1)
    )
    rgb = (rgb * 255.0).clip(0, 255).astype(np.uint8)

    points = unproject_depth_map_to_point_map(depth, extrinsics, intrinsics)
    finite = np.isfinite(points).all(axis=-1) & np.isfinite(confidence)
    if not np.any(finite):
        raise RuntimeError("Omega produced no finite points")
    threshold = float(np.percentile(confidence[finite], args.conf_percentile))
    mask = finite & (confidence >= threshold) & (confidence > 1e-5)

    selected = np.flatnonzero(mask.reshape(-1))
    rng = np.random.default_rng(args.seed)
    if len(selected) > args.max_points:
        selected = np.sort(rng.choice(selected, args.max_points, replace=False))
    limited_mask = np.zeros(mask.size, dtype=bool)
    limited_mask[selected] = True
    limited_mask = limited_mask.reshape(mask.shape)

    num_frames, height, width = confidence.shape
    points_xyf = create_pixel_coordinate_grid(num_frames, height, width)
    selected_points = points[limited_mask]
    selected_xyf = points_xyf[limited_mask]
    selected_rgb = rgb[limited_mask]

    print(f"Inference seconds: {elapsed:.3f}")
    print(
        "Peak VRAM GB: "
        f"{torch.cuda.max_memory_allocated() / 1024**3:.3f}"
    )
    print(
        f"Confidence percentile {args.conf_percentile:.1f}: {threshold:.6f}; "
        f"exporting {len(selected_points)} points"
    )

    reconstruction = batch_np_matrix_to_pycolmap_wo_track(
        selected_points,
        selected_xyf,
        selected_rgb,
        extrinsics,
        intrinsics,
        np.asarray([processed_width, processed_height]),
        shared_camera=False,
        camera_type="PINHOLE",
    )
    reconstruction = rescale_reconstruction_to_original_images(
        reconstruction,
        image_names,
        original_sizes,
        processed_width,
        processed_height,
    )

    sparse_dir = scene_dir / "sparse" / "0"
    sparse_dir.mkdir(parents=True, exist_ok=True)
    reconstruction.write(str(sparse_dir))
    trimesh.PointCloud(selected_points, colors=selected_rgb).export(
        str(sparse_dir / "points.ply")
    )

    print(f"Saved COLMAP reconstruction to {sparse_dir}")
    print(
        f"cameras={reconstruction.num_cameras()}, "
        f"registered_images={reconstruction.num_reg_images()}, "
        f"points3D={reconstruction.num_points3D()}"
    )


if __name__ == "__main__":
    main()
