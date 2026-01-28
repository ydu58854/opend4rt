"""Utility functions for D4RT dataset processing.

This module provides common utilities for loading, transforming,
and processing data for D4RT-compatible datasets.
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple, List, Optional, Union

import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F

try:
    from PIL import Image
except ImportError as exc:
    raise ImportError("PIL is required. Install pillow.") from exc


# =============================================================================
# Image Loading
# =============================================================================

def load_image(path: Union[str, Path]) -> np.ndarray:
    """Load an image as a numpy array.

    Args:
        path: Path to the image file.

    Returns:
        Image array with shape (H, W, 3) and dtype float32 in [0, 1].
    """
    img = Image.open(path).convert("RGB")
    arr = np.array(img, dtype=np.float32) / 255.0
    return arr


def load_image_tensor(
    path: Union[str, Path],
    target_size: Optional[Tuple[int, int]] = None,
    normalize: bool = True,
) -> Tensor:
    """Load an image as a PyTorch tensor.

    Args:
        path: Path to the image file.
        target_size: Optional target size (W, H) for resizing.
        normalize: Whether to normalize to [0, 1].

    Returns:
        Image tensor with shape (C, H, W).
    """
    img = Image.open(path).convert("RGB")

    if target_size is not None:
        img = img.resize(target_size, Image.BILINEAR)

    arr = np.array(img, dtype=np.uint8)
    tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous()

    if normalize:
        tensor = tensor.float() / 255.0
    else:
        tensor = tensor.float()

    return tensor


def load_depth_16bit(
    path: Union[str, Path],
    scale: float = 1000.0,
) -> np.ndarray:
    """Load a 16-bit PNG depth map.

    Args:
        path: Path to the depth image (16-bit PNG).
        scale: Scale factor to convert to metric units.

    Returns:
        Depth array with shape (H, W) in metric units.
    """
    img = Image.open(path)
    depth = np.array(img, dtype=np.float32)
    depth = depth / 65535.0 * scale
    return depth


def load_depth_tensor(
    path: Union[str, Path],
    target_size: Optional[Tuple[int, int]] = None,
    scale: float = 1000.0,
) -> Tensor:
    """Load a depth map as a PyTorch tensor.

    Args:
        path: Path to the depth image.
        target_size: Optional target size (W, H) for resizing.
        scale: Scale factor for 16-bit PNG.

    Returns:
        Depth tensor with shape (H, W).
    """
    depth = load_depth_16bit(path, scale)
    tensor = torch.from_numpy(depth)

    if target_size is not None:
        tensor = resize_depth(tensor, target_size)

    return tensor


# =============================================================================
# Resize Functions
# =============================================================================

def resize_image(
    image: Tensor,
    target_size: Tuple[int, int],
    mode: str = "bilinear",
) -> Tensor:
    """Resize an image tensor.

    Args:
        image: Image tensor (C, H, W) or (B, C, H, W).
        target_size: Target size (W, H).
        mode: Interpolation mode.

    Returns:
        Resized image tensor.
    """
    W, H = target_size
    if image.dim() == 3:
        image = image.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False

    resized = F.interpolate(image, size=(H, W), mode=mode, align_corners=False)

    if squeeze:
        resized = resized.squeeze(0)

    return resized


def resize_depth(
    depth: Tensor,
    target_size: Tuple[int, int],
) -> Tensor:
    """Resize a depth map using nearest neighbor interpolation.

    Args:
        depth: Depth tensor (H, W) or (B, H, W).
        target_size: Target size (W, H).

    Returns:
        Resized depth tensor.
    """
    W, H = target_size
    if depth.dim() == 2:
        depth = depth.unsqueeze(0).unsqueeze(0)
        squeeze = 2
    elif depth.dim() == 3:
        depth = depth.unsqueeze(1)
        squeeze = 1
    else:
        squeeze = 0

    resized = F.interpolate(depth, size=(H, W), mode="nearest")

    if squeeze == 2:
        resized = resized.squeeze(0).squeeze(0)
    elif squeeze == 1:
        resized = resized.squeeze(1)

    return resized


# =============================================================================
# Coordinate Transformations
# =============================================================================

def normalize_uv(
    px: np.ndarray,
    py: np.ndarray,
    W: int,
    H: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Normalize pixel coordinates to [0, 1] range.

    Uses align_corners=True convention:
    - px=0 -> u=0
    - px=W-1 -> u=1
    - px=(W-1)/2 -> u=0.5 (center pixel)

    This is consistent with D4RT paper which assumes principal point
    at (0.5, 0.5) in normalized coordinates.

    Args:
        px: X pixel coordinates.
        py: Y pixel coordinates.
        W: Image width.
        H: Image height.

    Returns:
        Tuple of normalized (u, v) coordinates in [0, 1].
    """
    u = px / (W - 1) if W > 1 else px
    v = py / (H - 1) if H > 1 else py
    return u, v


def denormalize_uv(
    u: np.ndarray,
    v: np.ndarray,
    W: int,
    H: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert normalized [0, 1] coordinates back to pixel coordinates.

    Uses align_corners=True convention (inverse of normalize_uv):
    - u=0 -> px=0
    - u=1 -> px=W-1
    - u=0.5 -> px=(W-1)/2

    Args:
        u: Normalized X coordinates in [0, 1].
        v: Normalized Y coordinates in [0, 1].
        W: Image width.
        H: Image height.

    Returns:
        Tuple of pixel (px, py) coordinates.
    """
    px = u * (W - 1)
    py = v * (H - 1)
    return px, py


def world_to_camera(
    xyz_world: np.ndarray,
    w2c: np.ndarray,
) -> np.ndarray:
    """Transform points from world coordinates to camera coordinates.

    Args:
        xyz_world: World coordinates (N, 3) or (3,).
        w2c: World-to-camera matrix (4, 4).

    Returns:
        Camera coordinates (N, 3) or (3,).
    """
    return transform_points(xyz_world, w2c)


def camera_to_world(
    xyz_cam: np.ndarray,
    w2c: np.ndarray,
) -> np.ndarray:
    """Transform points from camera coordinates to world coordinates.

    Args:
        xyz_cam: Camera coordinates (N, 3) or (3,).
        w2c: World-to-camera matrix (4, 4).

    Returns:
        World coordinates (N, 3) or (3,).
    """
    c2w = np.linalg.inv(w2c)
    return transform_points(xyz_cam, c2w)


def rotate_vectors(
    vectors: np.ndarray,
    rotation: np.ndarray,
) -> np.ndarray:
    """Rotate direction vectors (normals, etc.) using a rotation matrix.

    Unlike transform_points, this applies only rotation without translation.

    Args:
        vectors: Direction vectors (N, 3) or (3,).
        rotation: 3x3 rotation matrix OR 4x4 transformation matrix
                  (only the 3x3 rotation part is used).

    Returns:
        Rotated vectors (N, 3) or (3,).
    """
    single = vectors.ndim == 1
    if single:
        vectors = vectors.reshape(1, 3)

    R = rotation[:3, :3] if rotation.shape[0] == 4 else rotation
    result = (R @ vectors.T).T

    if single:
        result = result.squeeze(0)
    return result


def transform_points(
    points: np.ndarray,
    transform: np.ndarray,
) -> np.ndarray:
    """Apply a 4x4 transformation matrix to 3D points.

    Args:
        points: 3D points (N, 3) or (3,).
        transform: 4x4 transformation matrix.

    Returns:
        Transformed points (N, 3) or (3,).
    """
    single = points.ndim == 1
    if single:
        points = points.reshape(1, 3)

    # Convert to homogeneous coordinates
    ones = np.ones((points.shape[0], 1), dtype=points.dtype)
    points_homo = np.concatenate([points, ones], axis=1)

    # Apply transformation
    transformed = (transform @ points_homo.T).T
    result = transformed[:, :3]

    if single:
        result = result.squeeze(0)

    return result


def project_to_image(
    xyz_cam: np.ndarray,
    intrinsics: np.ndarray,
) -> np.ndarray:
    """Project 3D camera coordinates to 2D image coordinates.

    Args:
        xyz_cam: Camera coordinates (N, 3) or (3,).
        intrinsics: Camera intrinsic matrix (3, 3).

    Returns:
        Image coordinates (N, 2) or (2,).
    """
    single = xyz_cam.ndim == 1
    if single:
        xyz_cam = xyz_cam.reshape(1, 3)

    # Extract intrinsic parameters
    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    cx = intrinsics[0, 2]
    cy = intrinsics[1, 2]

    # Project
    x = xyz_cam[:, 0]
    y = xyz_cam[:, 1]
    z = xyz_cam[:, 2]

    # Avoid division by zero
    z = np.where(np.abs(z) < 1e-8, 1e-8, z)

    px = fx * x / z + cx
    py = fy * y / z + cy

    result = np.stack([px, py], axis=-1)

    if single:
        result = result.squeeze(0)

    return result


# =============================================================================
# Frame Sampling
# =============================================================================

def uniform_sample_frames(
    total_frames: int,
    target_frames: int = 48,
) -> List[int]:
    """Uniformly sample frame indices.

    Args:
        total_frames: Total number of available frames.
        target_frames: Number of frames to sample.

    Returns:
        List of frame indices.
    """
    if total_frames <= target_frames:
        # Not enough frames, pad with last frame
        indices = list(range(total_frames))
        indices += [total_frames - 1] * (target_frames - total_frames)
    else:
        # Uniform sampling
        step = total_frames / target_frames
        indices = [int(i * step) for i in range(target_frames)]

    return indices


def create_frame_index_mapping(
    original_indices: List[int],
) -> dict:
    """Create a mapping from original frame indices to sampled indices.

    Args:
        original_indices: List of original frame indices.

    Returns:
        Dictionary mapping original index to sampled index.
    """
    return {orig: i for i, orig in enumerate(original_indices)}


# =============================================================================
# Intrinsic Matrix Operations
# =============================================================================

def build_intrinsics_matrix(
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> np.ndarray:
    """Build a 3x3 camera intrinsic matrix.

    Args:
        fx: Focal length in x direction.
        fy: Focal length in y direction.
        cx: Principal point x coordinate.
        cy: Principal point y coordinate.

    Returns:
        3x3 intrinsic matrix.
    """
    return np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1],
    ], dtype=np.float32)


def scale_intrinsics(
    intrinsics: np.ndarray,
    scale_x: float,
    scale_y: float,
) -> np.ndarray:
    """Scale intrinsic parameters for resized images.

    Args:
        intrinsics: 3x3 intrinsic matrix.
        scale_x: Scale factor in x direction (new_W / old_W).
        scale_y: Scale factor in y direction (new_H / old_H).

    Returns:
        Scaled intrinsic matrix.
    """
    K = intrinsics.copy()
    K[0, 0] *= scale_x  # fx
    K[1, 1] *= scale_y  # fy
    K[0, 2] *= scale_x  # cx
    K[1, 2] *= scale_y  # cy
    return K


# =============================================================================
# Disparity and Flow
# =============================================================================

def compute_disparity_2d(
    uv_src: np.ndarray,
    uv_tgt: np.ndarray,
) -> np.ndarray:
    """Compute 2D disparity (optical flow) between source and target.

    Args:
        uv_src: Source coordinates (N, 2) in normalized [0, 1].
        uv_tgt: Target coordinates (N, 2) in normalized [0, 1].

    Returns:
        Disparity (N, 3) with [du, dv, 0].
    """
    disp_2d = uv_tgt - uv_src
    zeros = np.zeros((disp_2d.shape[0], 1), dtype=disp_2d.dtype)
    return np.concatenate([disp_2d, zeros], axis=-1)


def compute_disparity_3d(
    xyz_src: np.ndarray,
    xyz_tgt: np.ndarray,
) -> np.ndarray:
    """Compute 3D displacement between source and target.

    Args:
        xyz_src: Source 3D coordinates (N, 3).
        xyz_tgt: Target 3D coordinates (N, 3).

    Returns:
        Displacement (N, 3).
    """
    return xyz_tgt - xyz_src


# =============================================================================
# Normal Estimation
# =============================================================================

def estimate_normals_from_depth(
    depth: np.ndarray,
    intrinsics: np.ndarray,
) -> np.ndarray:
    """Estimate surface normals from a depth map.

    Args:
        depth: Depth map (H, W).
        intrinsics: Camera intrinsic matrix (3, 3).

    Returns:
        Normal map (H, W, 3) with unit normals.
    """
    H, W = depth.shape
    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    cx = intrinsics[0, 2]
    cy = intrinsics[1, 2]

    # Create pixel coordinate grids
    u = np.arange(W, dtype=np.float32)
    v = np.arange(H, dtype=np.float32)
    u, v = np.meshgrid(u, v)

    # Back-project to 3D
    z = depth
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    # Compute gradients
    dz_dx = np.gradient(z, axis=1)
    dz_dy = np.gradient(z, axis=0)

    # Compute normals
    nx = -dz_dx / fx
    ny = -dz_dy / fy
    nz = np.ones_like(z)

    # Normalize
    norm = np.sqrt(nx**2 + ny**2 + nz**2) + 1e-8
    normals = np.stack([nx / norm, ny / norm, nz / norm], axis=-1)

    return normals.astype(np.float32)


def sample_normals_at_points(
    normals: np.ndarray,
    uv: np.ndarray,
    W: int,
    H: int,
) -> np.ndarray:
    """Sample normals at given UV coordinates.

    Args:
        normals: Normal map (H, W, 3).
        uv: Normalized UV coordinates (N, 2) in [0, 1].
        W: Image width.
        H: Image height.

    Returns:
        Sampled normals (N, 3).
    """
    # Convert normalized UV to pixel coordinates
    px = (uv[:, 0] * (W - 1)).astype(np.int32)
    py = (uv[:, 1] * (H - 1)).astype(np.int32)

    # Clamp to valid range
    px = np.clip(px, 0, W - 1)
    py = np.clip(py, 0, H - 1)

    return normals[py, px]


# =============================================================================
# Sobel Edge Detection
# =============================================================================

def sobel_edge_detection(
    depth: Tensor,
    threshold: float = 0.1,
) -> Tensor:
    """Detect edges in a depth map using Sobel filters.

    Args:
        depth: Depth tensor (T, H, W) or (H, W).
        threshold: Edge threshold relative to max gradient.

    Returns:
        Binary edge mask with same shape as input.
    """
    device = depth.device
    dtype = depth.dtype

    # Sobel kernels
    sobel_x = torch.tensor([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ], dtype=dtype, device=device).view(1, 1, 3, 3)

    sobel_y = torch.tensor([
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]
    ], dtype=dtype, device=device).view(1, 1, 3, 3)

    # Handle different input shapes
    original_shape = depth.shape
    if depth.dim() == 2:
        depth = depth.unsqueeze(0).unsqueeze(0)
    elif depth.dim() == 3:
        depth = depth.unsqueeze(1)

    # Apply Sobel filters
    grad_x = F.conv2d(depth, sobel_x, padding=1)
    grad_y = F.conv2d(depth, sobel_y, padding=1)

    # Compute gradient magnitude
    grad_mag = torch.sqrt(grad_x ** 2 + grad_y ** 2)

    # Normalize and threshold
    max_grad = grad_mag.max()
    if max_grad > 0:
        edges = (grad_mag / max_grad) > threshold
    else:
        edges = grad_mag > 0

    edges = edges.float()

    # Restore original shape
    if len(original_shape) == 2:
        edges = edges.squeeze(0).squeeze(0)
    elif len(original_shape) == 3:
        edges = edges.squeeze(1)

    return edges
