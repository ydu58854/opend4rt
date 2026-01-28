"""PointOdyssey dataset implementation for D4RT.

PointOdyssey is a large-scale synthetic dataset for long-term point tracking.
This module provides a D4RT-compatible dataset implementation.

Dataset structure:
    pointodyssey/
    ├── train/
    │   ├── scene_name/
    │   │   ├── rgbs/
    │   │   │   ├── rgb_00000.jpg
    │   │   │   └── ...
    │   │   ├── depths/
    │   │   │   ├── depth_00000.png
    │   │   │   └── ...
    │   │   ├── normals/  (optional, .jpg format)
    │   │   ├── anno.npz
    │   │   ├── info.npz
    │   │   └── scene_info.json
    │   └── ...
    └── test/
        └── ...

anno.npz contents:
    - trajs_2d: (T, N, 2) 2D trajectories in pixel coordinates
    - trajs_3d: (T, N, 3) 3D trajectories in world coordinates
    - valids: (T, N) validity mask (point in scene)
    - visibs: (T, N) visibility mask (point not occluded)
    - intrinsics: (T, 3, 3) camera intrinsic matrices
    - extrinsics: (T, 4, 4) camera extrinsic matrices (w2c, world-to-camera)

Coordinate convention (verified against official PointOdyssey reprojection.py):
    - extrinsics are w2c (world-to-camera), NOT c2w
    - Camera uses OpenCV convention: +X right, +Y down, +Z forward
    - Visible points have positive Z in camera coordinates
    - Official reprojection: XYZ_cam = extrinsics @ [xyz_world, 1]^T
    - Normal maps are rendered in OpenGL camera space by Blender
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch import Tensor

from .base_config import PointOdysseyConfig
from .base_dataset import Base4DTrajectoryDataset
from .utils import (
    load_image,
    load_image_tensor,
    load_depth_16bit,
    resize_depth,
    estimate_normals_from_depth,
)


class PointOdysseyDataset(Base4DTrajectoryDataset):
    """PointOdyssey dataset for D4RT training.

    This dataset loads dense point trajectories from PointOdyssey and
    provides D4RT-compatible samples with queries and targets.

    Attributes:
        config: PointOdysseyConfig instance.
        scenes: List of available scenes.
        scene_cache: Cache for loaded scene data.
    """

    def __init__(self, config: PointOdysseyConfig):
        self.scene_cache: Dict[str, Dict[str, Any]] = {}
        super().__init__(config)

    def _load_scene_list(self) -> List[Dict[str, Any]]:
        """Scan the dataset directory for valid scenes.

        Returns:
            List of scene info dictionaries.
        """
        scenes = []
        split_dir = self.config.root / self.config.split

        if not split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {split_dir}")

        for scene_dir in sorted(split_dir.iterdir()):
            if not scene_dir.is_dir():
                continue

            # Check for required files
            anno_file = scene_dir / "anno.npz"
            rgbs_dir = scene_dir / "rgbs"

            if not anno_file.exists() or not rgbs_dir.exists():
                continue

            scene_id = scene_dir.name

            # Apply scene filters
            if not self.config.should_include_scene(scene_id):
                continue

            # Count frames
            rgb_files = sorted(rgbs_dir.glob("rgb_*.jpg"))
            num_frames = len(rgb_files)

            if num_frames == 0:
                continue

            # Get original resolution from first image
            # This is more reliable than inferring from intrinsics
            from PIL import Image
            first_img = Image.open(rgb_files[0])
            orig_W, orig_H = first_img.size
            first_img.close()

            scenes.append({
                "scene_id": scene_id,
                "path": scene_dir,
                "num_frames": num_frames,
                "orig_size": (orig_W, orig_H),
            })

        if len(scenes) == 0:
            raise ValueError(f"No valid scenes found in {split_dir}")

        return scenes

    def _get_scene_path(self, scene_id: str) -> Path:
        """Get the path to a scene directory.

        Args:
            scene_id: Scene identifier.

        Returns:
            Path to scene directory.
        """
        return self.config.root / self.config.split / scene_id

    def _load_anno(self, scene_id: str) -> Dict[str, np.ndarray]:
        """Load annotation data with caching.

        Args:
            scene_id: Scene identifier.

        Returns:
            Dictionary with annotation arrays.
        """
        if scene_id not in self.scene_cache:
            scene_path = self._get_scene_path(scene_id)
            anno_path = scene_path / "anno.npz"

            with np.load(anno_path) as data:
                self.scene_cache[scene_id] = {
                    "trajs_2d": data["trajs_2d"].astype(np.float32),
                    "trajs_3d": data["trajs_3d"].astype(np.float32),
                    "valids": data["valids"].astype(bool),
                    "visibs": data["visibs"].astype(bool),
                    "intrinsics": data["intrinsics"].astype(np.float32),
                    "extrinsics": data["extrinsics"].astype(np.float32),
                }

        return self.scene_cache[scene_id]

    def _load_trajectories(self, scene_id: str) -> Dict[str, np.ndarray]:
        """Load trajectory data for a scene.

        Args:
            scene_id: Scene identifier.

        Returns:
            Dictionary with trajectory arrays.
        """
        anno = self._load_anno(scene_id)
        return {
            "trajs_2d": anno["trajs_2d"],
            "trajs_3d": anno["trajs_3d"],
            "valids": anno["valids"],
            "visibs": anno["visibs"],
        }

    def _load_camera_params(self, scene_id: str) -> Dict[str, np.ndarray]:
        """Load camera parameters for a scene.

        Args:
            scene_id: Scene identifier.

        Returns:
            Dictionary with camera intrinsics and extrinsics.
        """
        anno = self._load_anno(scene_id)
        return {
            "intrinsics": anno["intrinsics"],
            "extrinsics": anno["extrinsics"],
        }

    def _load_frames(self, scene_id: str, frame_indices: List[int]) -> Tensor:
        """Load RGB frames for a scene.

        Args:
            scene_id: Scene identifier.
            frame_indices: List of frame indices to load.

        Returns:
            (C, T, H, W) tensor with RGB frames.
        """
        scene_path = self._get_scene_path(scene_id)
        rgb_dir = scene_path / "rgbs"

        frames = []
        for idx in frame_indices:
            img_path = rgb_dir / f"rgb_{idx:05d}.jpg"

            if not img_path.exists():
                raise FileNotFoundError(f"RGB image not found: {img_path}")

            img = load_image_tensor(
                img_path,
                target_size=self.config.target_resolution,
                normalize=self.config.normalize,
            )
            frames.append(img)

        # Stack as (T, C, H, W) then permute to (C, T, H, W)
        stacked = torch.stack(frames, dim=0)  # (T, C, H, W)
        return stacked.permute(1, 0, 2, 3).contiguous()  # (C, T, H, W)

    def _load_depths(
        self,
        scene_id: str,
        frame_indices: List[int],
    ) -> Optional[np.ndarray]:
        """Load depth maps for a scene.

        Args:
            scene_id: Scene identifier.
            frame_indices: List of frame indices to load.

        Returns:
            (T, H, W) depth array or None if not available.
        """
        if not self.config.use_depth:
            return None

        scene_path = self._get_scene_path(scene_id)
        depth_dir = scene_path / "depths"

        if not depth_dir.exists():
            return None

        depths = []
        W, H = self.config.target_resolution

        for idx in frame_indices:
            depth_path = depth_dir / f"depth_{idx:05d}.png"

            if not depth_path.exists():
                return None

            depth = load_depth_16bit(depth_path, scale=self.config.depth_scale)

            # Resize if needed
            depth_tensor = torch.from_numpy(depth)
            depth_resized = resize_depth(depth_tensor, (W, H))
            depths.append(depth_resized.numpy())

        return np.stack(depths, axis=0)  # (T, H, W)

    def _load_normals(
        self,
        scene_id: str,
        frame_indices: List[int],
    ) -> Optional[np.ndarray]:
        """Load or compute normal maps for a scene.

        Args:
            scene_id: Scene identifier.
            frame_indices: List of frame indices.

        Returns:
            (T, H, W, 3) normal array or None if not available.
        """
        if not self.config.use_normals:
            return None

        scene_path = self._get_scene_path(scene_id)
        normal_dir = scene_path / "normals"

        # First try to load precomputed normals
        if normal_dir.exists():
            normals = []
            W, H = self.config.target_resolution

            for idx in frame_indices:
                normal_path = normal_dir / f"normal_{idx:05d}.jpg"
                if normal_path.exists():
                    # Load normal image (stored as RGB [0,255] -> [-1,1])
                    normal_img = load_image(normal_path)  # (H, W, 3) in [0, 1]
                    normal = normal_img * 2.0 - 1.0  # Convert to [-1, 1]
                    # Resize if needed
                    if normal.shape[1] != W or normal.shape[0] != H:
                        from PIL import Image
                        normal_pil = Image.fromarray(((normal + 1.0) * 127.5).astype(np.uint8))
                        normal_pil = normal_pil.resize((W, H), Image.BILINEAR)
                        normal = np.array(normal_pil, dtype=np.float32) / 127.5 - 1.0
                    normals.append(normal)
                else:
                    break

            if len(normals) == len(frame_indices):
                return np.stack(normals, axis=0)  # (T, H, W, 3)

        # Compute normals from depth if precomputed not available
        depths = self._load_depths(scene_id, frame_indices)
        if depths is None:
            return None

        # Get intrinsics for the sampled frames
        anno = self._load_anno(scene_id)
        intrinsics = anno["intrinsics"][frame_indices]

        # Scale intrinsics for target resolution
        orig_H, orig_W = depths.shape[1], depths.shape[2]
        W, H = self.config.target_resolution

        normals = []
        for t in range(len(frame_indices)):
            K = intrinsics[t].copy()
            # Intrinsics are for original resolution, scale for target
            scale_x = W / orig_W if orig_W != W else 1.0
            scale_y = H / orig_H if orig_H != H else 1.0
            K[0, 0] *= scale_x
            K[1, 1] *= scale_y
            K[0, 2] *= scale_x
            K[1, 2] *= scale_y

            normal = estimate_normals_from_depth(depths[t], K)
            normals.append(normal)

        return np.stack(normals, axis=0)  # (T, H, W, 3)

    def clear_cache(self) -> None:
        """Clear the scene data cache to free memory."""
        self.scene_cache.clear()

class PointOdysseySingleFrameDataset(PointOdysseyDataset):
    """PointOdyssey dataset for single-frame overfitting tests.

    This variant samples all queries from a single frame within each scene.
    """

    def __init__(self, config: PointOdysseyConfig, frame_index: int = 0):
        self.frame_index = frame_index
        super().__init__(config)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single-frame sample.

        All queries have t_src = t_tgt = t_cam = frame_index.
        """
        from .trajectory_sampler import SingleFrameTrajectoryQuerySampler

        scene_info = self.scenes[idx]
        scene_id = scene_info["scene_id"]

        # Load data
        traj_data = self._load_trajectories(scene_id)
        cam_data = self._load_camera_params(scene_id)

        # Use single frame repeated to meet num_frames requirement
        total_frames = traj_data["trajs_2d"].shape[0]
        t = min(self.frame_index, total_frames - 1)

        # For model compatibility, we still need 48 frames (same frame repeated)
        frame_indices = [t] * self.config.num_frames

        # Load frames (same frame repeated)
        images = self._load_frames(scene_id, [t])  # (C, 1, H, W)
        images = images.expand(-1, self.config.num_frames, -1, -1).contiguous()

        C, T, H, W = images.shape

        # Get original resolution from scene_info (stored during _load_scene_list)
        orig_W, orig_H = scene_info["orig_size"]

        # Load depth and normal for this frame
        depths = self._load_depths(scene_id, [t])
        depth = depths[0] if depths is not None else None

        normals = self._load_normals(scene_id, [t])
        normal = normals[0] if normals is not None else None

        # Sample queries for single frame
        sampler = SingleFrameTrajectoryQuerySampler(
            num_queries=self.config.num_queries,
            edge_ratio=self.config.query_edge_ratio,
        )

        queries, targets = sampler.sample(
            trajs_2d=traj_data["trajs_2d"],
            trajs_3d=traj_data["trajs_3d"],
            valids=traj_data["valids"],
            visibs=traj_data["visibs"],
            intrinsics=cam_data["intrinsics"],
            extrinsics=cam_data["extrinsics"],
            frame_idx=t,
            image_size=(W, H),
            orig_size=(orig_W, orig_H),
            depth=depth,
            normal=normal,
        )

        meta = {
            "aspect_ratio": float(W) / float(H),
            "img_patch_size": self.config.img_patch_size,
            "align_corners": self.config.align_corners,
        }

        return {
            "meta": meta,
            "images": images,
            "query": queries,
            "targets": targets,
        }
