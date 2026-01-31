"""Depth evaluation and visualization for PointOdyssey inference results."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch import Tensor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from datasets.utils import load_depth_16bit, resize_depth


# =============================================================================
# Data Loading
# =============================================================================


def load_gt_depths(
    scene_dir: Path,
    max_frames: Optional[int] = None,
    depth_scale: float = 1000.0,
) -> np.ndarray:
    """Load GT depth maps from scene directory.

    Args:
        scene_dir: Path to scene directory (containing depths/ subdirectory).
        max_frames: Maximum number of frames to load (None = all).
        depth_scale: Scale factor for 16-bit PNG conversion.

    Returns:
        (T, H, W) depth array in meters.
    """
    depth_dir = scene_dir / "depths"
    if not depth_dir.exists():
        raise FileNotFoundError(f"Depths directory not found: {depth_dir}")

    depth_files = sorted(depth_dir.glob("depth_*.png"))
    if not depth_files:
        raise FileNotFoundError(f"No depth files found in {depth_dir}")

    if max_frames is not None:
        depth_files = depth_files[:max_frames]

    depths = []
    for dp in depth_files:
        depth = load_depth_16bit(dp, scale=depth_scale)
        depths.append(depth)

    return np.stack(depths, axis=0)


def load_pred_depths(pred_path: Path) -> np.ndarray:
    """Load predicted depth maps from .npy file.

    Args:
        pred_path: Path to depth.npy file from inference.

    Returns:
        (T, H, W) depth array.
    """
    if not pred_path.exists():
        raise FileNotFoundError(f"Prediction file not found: {pred_path}")

    pred = np.load(pred_path)
    if pred.ndim != 3:
        raise ValueError(f"Expected 3D array (T, H, W), got shape {pred.shape}")

    return pred


def align_depth_sizes(
    pred: np.ndarray,
    gt: np.ndarray,
    mode: str = "resize_pred",
) -> Tuple[np.ndarray, np.ndarray]:
    """Align predicted and GT depth sizes.

    Args:
        pred: (T, H_pred, W_pred) predicted depths.
        gt: (T, H_gt, W_gt) ground truth depths.
        mode: "resize_pred" or "resize_gt".

    Returns:
        Aligned (pred, gt) arrays with matching spatial dimensions.
    """
    T_pred, H_pred, W_pred = pred.shape
    T_gt, H_gt, W_gt = gt.shape

    # Align temporal dimension
    T = min(T_pred, T_gt)
    pred = pred[:T]
    gt = gt[:T]

    # Align spatial dimensions
    if H_pred != H_gt or W_pred != W_gt:
        if mode == "resize_pred":
            pred_tensor = torch.from_numpy(pred)
            pred_resized = resize_depth(pred_tensor, (W_gt, H_gt))
            pred = pred_resized.numpy()
        elif mode == "resize_gt":
            gt_tensor = torch.from_numpy(gt)
            gt_resized = resize_depth(gt_tensor, (W_pred, H_pred))
            gt = gt_resized.numpy()
        else:
            raise ValueError(f"Unknown resize mode: {mode}")

    return pred, gt


def create_valid_mask(
    gt: np.ndarray,
    pred: np.ndarray,
    min_depth: float = 0.1,
    max_depth: float = 100.0,
) -> np.ndarray:
    """Create valid mask for evaluation.

    Args:
        gt: Ground truth depth.
        pred: Predicted depth.
        min_depth: Minimum valid depth.
        max_depth: Maximum valid depth.

    Returns:
        Boolean mask of valid pixels.
    """
    valid = (gt > min_depth) & (gt < max_depth) & (pred > 0) & np.isfinite(pred)
    return valid


# =============================================================================
# Metrics Computation
# =============================================================================


def compute_abs_rel(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> float:
    """Absolute relative error: |pred - gt| / gt."""
    valid_pred = pred[mask]
    valid_gt = gt[mask]
    if len(valid_gt) == 0:
        return float("nan")
    return float(np.mean(np.abs(valid_pred - valid_gt) / valid_gt))


def compute_sq_rel(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> float:
    """Squared relative error: (pred - gt)^2 / gt."""
    valid_pred = pred[mask]
    valid_gt = gt[mask]
    if len(valid_gt) == 0:
        return float("nan")
    return float(np.mean(((valid_pred - valid_gt) ** 2) / valid_gt))


def compute_rmse(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> float:
    """Root mean squared error."""
    valid_pred = pred[mask]
    valid_gt = gt[mask]
    if len(valid_gt) == 0:
        return float("nan")
    return float(np.sqrt(np.mean((valid_pred - valid_gt) ** 2)))


def compute_rmse_log(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> float:
    """RMSE in log space."""
    valid_pred = pred[mask]
    valid_gt = gt[mask]
    if len(valid_gt) == 0:
        return float("nan")
    valid_pred = np.maximum(valid_pred, 1e-6)
    valid_gt = np.maximum(valid_gt, 1e-6)
    return float(np.sqrt(np.mean((np.log(valid_pred) - np.log(valid_gt)) ** 2)))


def compute_threshold_accuracy(
    pred: np.ndarray,
    gt: np.ndarray,
    mask: np.ndarray,
    threshold: float,
) -> float:
    """Threshold accuracy: % of max(pred/gt, gt/pred) < threshold."""
    valid_pred = pred[mask]
    valid_gt = gt[mask]
    if len(valid_gt) == 0:
        return float("nan")
    ratio = np.maximum(valid_pred / valid_gt, valid_gt / valid_pred)
    return float(np.mean(ratio < threshold))


def compute_all_metrics(
    pred: np.ndarray,
    gt: np.ndarray,
    mask: np.ndarray,
) -> Dict[str, float]:
    """Compute all standard depth metrics.

    Returns:
        Dictionary with abs_rel, sq_rel, rmse, rmse_log, delta_1/2/3, num_valid_pixels.
    """
    return {
        "abs_rel": compute_abs_rel(pred, gt, mask),
        "sq_rel": compute_sq_rel(pred, gt, mask),
        "rmse": compute_rmse(pred, gt, mask),
        "rmse_log": compute_rmse_log(pred, gt, mask),
        "delta_1": compute_threshold_accuracy(pred, gt, mask, 1.25),
        "delta_2": compute_threshold_accuracy(pred, gt, mask, 1.25**2),
        "delta_3": compute_threshold_accuracy(pred, gt, mask, 1.25**3),
        "num_valid_pixels": int(np.sum(mask)),
    }


def compute_per_frame_metrics(
    pred: np.ndarray,
    gt: np.ndarray,
    min_depth: float = 0.1,
    max_depth: float = 100.0,
) -> Dict[str, list]:
    """Compute metrics for each frame.

    Returns:
        Dictionary with lists of per-frame metrics.
    """
    T = len(pred)
    metrics_keys = ["abs_rel", "sq_rel", "rmse", "rmse_log", "delta_1", "delta_2", "delta_3"]
    per_frame = {k: [] for k in metrics_keys}

    for t in range(T):
        mask_t = create_valid_mask(gt[t : t + 1], pred[t : t + 1], min_depth, max_depth)
        m = compute_all_metrics(pred[t : t + 1], gt[t : t + 1], mask_t)
        for k in metrics_keys:
            per_frame[k].append(m[k])

    return per_frame


# =============================================================================
# Visualization (PIL-based, no matplotlib)
# =============================================================================


def apply_viridis_colormap(depth: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    """Apply viridis-like colormap to depth map.

    Args:
        depth: (H, W) depth array.
        vmin: Minimum depth for normalization.
        vmax: Maximum depth for normalization.

    Returns:
        (H, W, 3) RGB image in uint8.
    """
    # Normalize to [0, 1]
    norm = np.clip((depth - vmin) / (vmax - vmin + 1e-8), 0, 1)

    # Viridis-like colormap (simplified version)
    # dark purple -> blue -> green -> yellow
    r = np.clip(0.267004 + norm * (0.993248 - 0.267004) * norm, 0, 1)
    g = np.clip(0.004874 + norm * (0.906157 - 0.004874), 0, 1)
    b = np.clip(0.329415 + norm * (0.143936 - 0.329415) * (1 - norm) + (1 - norm) * 0.5, 0, 1)

    # More accurate viridis approximation
    t = norm
    r = 0.267004 + t * (0.282327 + t * (0.293307 + t * (-0.716830 + t * 0.874230)))
    g = 0.004874 + t * (1.015861 + t * (-0.044795 + t * (-0.291366 + t * 0.227626)))
    b = 0.329415 + t * (1.405770 + t * (-2.234848 + t * (1.259500 + t * (-0.259500))))

    r = np.clip(r, 0, 1)
    g = np.clip(g, 0, 1)
    b = np.clip(b, 0, 1)

    rgb = np.stack([r, g, b], axis=-1)
    return (rgb * 255).astype(np.uint8)


def apply_hot_colormap(error: np.ndarray, vmax: float) -> np.ndarray:
    """Apply hot colormap to error map.

    Args:
        error: (H, W) error array.
        vmax: Maximum error for normalization.

    Returns:
        (H, W, 3) RGB image in uint8.
    """
    # Normalize to [0, 1]
    norm = np.clip(error / (vmax + 1e-8), 0, 1)

    # Hot colormap: black -> red -> yellow -> white
    r = np.clip(norm * 3, 0, 1)
    g = np.clip((norm - 0.33) * 3, 0, 1)
    b = np.clip((norm - 0.67) * 3, 0, 1)

    rgb = np.stack([r, g, b], axis=-1)
    return (rgb * 255).astype(np.uint8)


def create_comparison_image(
    pred_depth: np.ndarray,
    gt_depth: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """Create side-by-side comparison image (pred | gt | error).

    Args:
        pred_depth: (H, W) predicted depth.
        gt_depth: (H, W) ground truth depth.
        mask: (H, W) valid mask.

    Returns:
        (H, W*3, 3) RGB comparison image.
    """
    # Compute depth range from valid pixels (filter inf/nan)
    valid_gt = gt_depth[mask]
    valid_pred = pred_depth[mask]
    valid_gt = valid_gt[np.isfinite(valid_gt)]
    valid_pred = valid_pred[np.isfinite(valid_pred)]

    if len(valid_gt) > 0 and len(valid_pred) > 0:
        # Clamp extreme values before percentile to avoid overflow
        pred_clamped = np.clip(valid_pred, -1e6, 1e6)
        gt_clamped = np.clip(valid_gt, -1e6, 1e6)
        vmin = min(np.percentile(gt_clamped, 5), np.percentile(pred_clamped, 5))
        vmax = max(np.percentile(gt_clamped, 95), np.percentile(pred_clamped, 95))
    else:
        vmin, vmax = 0, 1

    # Clamp depths for visualization
    pred_vis_input = np.clip(pred_depth, vmin, vmax)
    gt_vis_input = np.clip(gt_depth, vmin, vmax)

    # Create depth visualizations
    pred_vis = apply_viridis_colormap(pred_vis_input, vmin, vmax)
    gt_vis = apply_viridis_colormap(gt_vis_input, vmin, vmax)

    # Create error visualization
    error = np.abs(pred_depth - gt_depth)
    error = np.clip(error, 0, 1e6)  # Clamp to avoid overflow
    error[~mask] = 0
    valid_error = error[mask]
    valid_error = valid_error[np.isfinite(valid_error)]
    error_max = np.percentile(valid_error, 95) if len(valid_error) > 0 else 1
    error_vis = apply_hot_colormap(error, error_max)

    # Concatenate horizontally: pred | gt | error
    comparison = np.concatenate([pred_vis, gt_vis, error_vis], axis=1)

    return comparison


def visualize_depth_sequence(
    pred_depths: np.ndarray,
    gt_depths: np.ndarray,
    output_dir: Path,
    max_vis_frames: int = 10,
    min_depth: float = 0.1,
    max_depth: float = 100.0,
) -> None:
    """Generate and save visualization images using PIL."""
    from PIL import Image

    T = len(pred_depths)
    step = max(1, T // max_vis_frames)
    frame_indices = list(range(0, T, step))[:max_vis_frames]

    output_dir.mkdir(parents=True, exist_ok=True)

    for frame_idx in frame_indices:
        pred = pred_depths[frame_idx]
        gt = gt_depths[frame_idx]

        mask = create_valid_mask(gt[np.newaxis], pred[np.newaxis], min_depth, max_depth)[0]

        # Create comparison image
        comparison = create_comparison_image(pred, gt, mask)

        # Save using PIL
        img = Image.fromarray(comparison)
        img.save(output_dir / f"frame_{frame_idx:05d}.png")

    print(f"Saved {len(frame_indices)} visualization images to {output_dir}")


# =============================================================================
# Main Entry Point
# =============================================================================


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate and visualize depth predictions")
    parser.add_argument(
        "--pred-path",
        type=str,
        required=True,
        help="Path to predicted depth.npy file",
    )
    parser.add_argument(
        "--scene-dir",
        type=str,
        required=True,
        help="Path to scene directory (containing depths/ subdirectory)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for metrics and visualizations",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum frames to evaluate",
    )
    parser.add_argument(
        "--depth-scale",
        type=float,
        default=1000.0,
        help="Depth scale factor for GT 16-bit PNG conversion",
    )
    parser.add_argument(
        "--min-depth",
        type=float,
        default=0.1,
        help="Minimum valid depth for evaluation (meters)",
    )
    parser.add_argument(
        "--max-depth",
        type=float,
        default=100.0,
        help="Maximum valid depth for evaluation (meters)",
    )
    parser.add_argument(
        "--max-vis-frames",
        type=int,
        default=10,
        help="Maximum frames to visualize",
    )
    parser.add_argument(
        "--resize-mode",
        type=str,
        default="resize_pred",
        choices=["resize_pred", "resize_gt"],
        help="How to handle size mismatch",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    pred_path = Path(args.pred_path)
    scene_dir = Path(args.scene_dir)
    output_dir = Path(args.output_dir)

    print(f"Loading predictions from: {pred_path}")
    pred_depths = load_pred_depths(pred_path)
    print(f"  Prediction shape: {pred_depths.shape}")

    print(f"Loading GT depths from: {scene_dir / 'depths'}")
    max_frames = args.max_frames or len(pred_depths)
    gt_depths = load_gt_depths(scene_dir, max_frames=max_frames, depth_scale=args.depth_scale)
    print(f"  GT shape: {gt_depths.shape}")

    print(f"Aligning sizes (mode: {args.resize_mode})...")
    pred_depths, gt_depths = align_depth_sizes(pred_depths, gt_depths, mode=args.resize_mode)
    print(f"  Aligned shape: pred={pred_depths.shape}, gt={gt_depths.shape}")

    print("Computing metrics...")
    mask = create_valid_mask(gt_depths, pred_depths, args.min_depth, args.max_depth)
    overall_metrics = compute_all_metrics(pred_depths, gt_depths, mask)
    per_frame = compute_per_frame_metrics(
        pred_depths, gt_depths, min_depth=args.min_depth, max_depth=args.max_depth
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_result = {
        "overall": overall_metrics,
        "per_frame": per_frame,
        "config": {
            "pred_path": str(pred_path.resolve()),
            "scene_dir": str(scene_dir.resolve()),
            "depth_scale": args.depth_scale,
            "min_depth": args.min_depth,
            "max_depth": args.max_depth,
            "resize_mode": args.resize_mode,
            "num_frames": len(pred_depths),
        },
    }

    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_result, f, indent=2)
    print(f"Saved metrics to: {metrics_path}")

    print("Generating visualizations...")
    vis_dir = output_dir / "vis"
    visualize_depth_sequence(
        pred_depths,
        gt_depths,
        vis_dir,
        max_vis_frames=args.max_vis_frames,
        min_depth=args.min_depth,
        max_depth=args.max_depth,
    )

    print("\n" + "=" * 50)
    print("Depth Evaluation Results")
    print("=" * 50)
    print(f"  Abs Rel:      {overall_metrics['abs_rel']:.4f}")
    print(f"  Sq Rel:       {overall_metrics['sq_rel']:.4f}")
    print(f"  RMSE:         {overall_metrics['rmse']:.4f}")
    print(f"  RMSE log:     {overall_metrics['rmse_log']:.4f}")
    print(f"  delta < 1.25:   {overall_metrics['delta_1']:.2%}")
    print(f"  delta < 1.25^2: {overall_metrics['delta_2']:.2%}")
    print(f"  delta < 1.25^3: {overall_metrics['delta_3']:.2%}")
    print(f"  Valid pixels: {overall_metrics['num_valid_pixels']:,}")
    print("=" * 50)


if __name__ == "__main__":
    main()
