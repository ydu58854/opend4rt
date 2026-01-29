"""Base dataset classes for D4RT-compatible datasets.

This module provides abstract base classes that define the interface
for D4RT-compatible datasets.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
import random
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from .base_config import Base4DDatasetConfig
from .trajectory_sampler import TrajectoryQuerySampler
from .utils import (
    uniform_sample_frames,
    sample_random_stride_frames,
    apply_temporal_color_jitter,
    apply_temporal_color_drop,
    apply_gaussian_blur,
    random_resized_crop_params,
    apply_random_resized_crop,
)


class Base4DTrajectoryDataset(Dataset, ABC):
    """Abstract base class for 4D trajectory-based datasets.

    This class provides the common interface and logic for datasets that
    use dense point trajectories for supervision. Subclasses must implement
    the abstract methods for loading scene-specific data.

    The output format is compatible with D4RT's d4rt_collate_fn.

    Attributes:
        config: Dataset configuration.
        scenes: List of scene data dictionaries.
        sampler: Query sampler instance.
    """

    def __init__(self, config: Base4DDatasetConfig):
        super().__init__()
        self.config = config
        self.scenes: List[Dict[str, Any]] = []
        self.sampler: Optional[TrajectoryQuerySampler] = None

        self._setup()

    def _setup(self) -> None:
        """Initialize the dataset by loading scene list and creating sampler."""
        self.scenes = self._load_scene_list()

        self.sampler = TrajectoryQuerySampler(
            num_queries=self.config.num_queries,
            edge_ratio=self.config.query_edge_ratio,
            min_visible_frames=self.config.min_visible_frames,
            tcam_equals_ttgt_ratio=self.config.tcam_equals_ttgt_ratio,
        )

    def __len__(self) -> int:
        return len(self.scenes)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a sample from the dataset.

        Returns:
            Dictionary with keys:
                - meta: dict with aspect_ratio, img_patch_size, align_corners
                - images: (C, T, H, W) tensor
                - query: (N, 5) tensor
                - targets: dict with L3D, L2D, Lvis, Ldisp, Lconf, Lnormal
        """
        scene_info = self.scenes[idx]
        scene_id = scene_info["scene_id"]

        # Load trajectory and camera data
        traj_data = self._load_trajectories(scene_id)
        cam_data = self._load_camera_params(scene_id)

        # Determine frame count and sample indices
        total_frames = traj_data["trajs_2d"].shape[0]
        if self.config.temporal_random_stride:
            frame_indices = sample_random_stride_frames(
                total_frames,
                self.config.num_frames,
                self.config.temporal_stride_min,
                self.config.temporal_stride_max,
            )
        else:
            frame_indices = uniform_sample_frames(total_frames, self.config.num_frames)

        # Load frames (raw if augmentation enabled)
        if self.config.use_data_augmentation:
            images = self._load_frames_raw(scene_id, frame_indices)
        else:
            images = self._load_frames(scene_id, frame_indices)
        images = self._ensure_num_frames(images)

        # Get original resolution from scene_info if available (preferred)
        # Otherwise fall back to inferring from intrinsics (assumes principal point at center)
        if "orig_size" in scene_info:
            orig_W, orig_H = scene_info["orig_size"]
        else:
            import warnings
            warnings.warn(
                f"Scene {scene_id} missing orig_size in scene_info. "
                "Inferring from intrinsics (assumes principal point at center).",
                UserWarning
            )
            intrinsics = cam_data["intrinsics"]
            orig_cx = intrinsics[0, 0, 2]
            orig_cy = intrinsics[0, 1, 2]
            orig_W = int(orig_cx * 2)
            orig_H = int(orig_cy * 2)
        orig_size = (orig_W, orig_H)

        # Load optional depth and normal maps
        if self.config.use_data_augmentation:
            depths = self._load_depths_raw(scene_id, frame_indices)
            normals = self._load_normals_raw(scene_id, frame_indices)
        else:
            depths = self._load_depths(scene_id, frame_indices)
            normals = self._load_normals(scene_id, frame_indices)

        # Data augmentation (temporal + spatial)
        if self.config.use_data_augmentation:
            # Prepare video as (T, C, H, W)
            video = images.permute(1, 0, 2, 3).contiguous()
            rescale_back = False
            if not self.config.normalize:
                video = video / 255.0
                rescale_back = True

            # Temporal color jitter (consistent across frames)
            b, c, s, h = self.config.color_jitter
            if b > 0 or c > 0 or s > 0 or h > 0:
                brightness = random.uniform(max(0.0, 1.0 - b), 1.0 + b)
                contrast = random.uniform(max(0.0, 1.0 - c), 1.0 + c)
                saturation = random.uniform(max(0.0, 1.0 - s), 1.0 + s)
                hue = random.uniform(-h, h)
                video = apply_temporal_color_jitter(
                    video,
                    brightness=brightness,
                    contrast=contrast,
                    saturation=saturation,
                    hue=hue,
                )

            # Color drop & blur
            video = apply_temporal_color_drop(video, self.config.color_drop_prob)
            video = apply_gaussian_blur(
                video,
                prob=self.config.gaussian_blur_prob,
                sigma_min=self.config.gaussian_blur_sigma[0],
                sigma_max=self.config.gaussian_blur_sigma[1],
            )

            # Random resized crop (sampled/applied in ORIGINAL resolution, then resized)
            crop_params = random_resized_crop_params(
                height=orig_H,
                width=orig_W,
                scale=self.config.random_crop_scale,
                ratio=self.config.random_crop_ratio,
                zoom_prob=self.config.random_zoom_prob,
                zoom_scale=self.config.random_zoom_scale,
            )
            y0_o, x0_o, ch_o, cw_o = crop_params

            depth_tensor = None
            normal_tensor = None
            if depths is not None:
                depth_tensor = torch.from_numpy(depths).float()
            if normals is not None:
                normal_tensor = torch.from_numpy(normals).float()

            video, depth_tensor, normal_tensor = apply_random_resized_crop(
                video,
                depth_tensor,
                normal_tensor,
                crop_params=crop_params,
                output_size=self.config.target_resolution,
            )

            if rescale_back:
                video = (video * 255.0).clamp(0.0, 255.0)
            images = video.permute(1, 0, 2, 3).contiguous()
            if depth_tensor is not None:
                depths = depth_tensor.numpy()
            if normal_tensor is not None:
                normals = normal_tensor.numpy()

            # Adjust trajectories + intrinsics for crop (in ORIGINAL resolution space)
            trajs_2d = traj_data["trajs_2d"].copy()
            intrinsics = cam_data["intrinsics"].copy()

            trajs_2d[..., 0] -= x0_o
            trajs_2d[..., 1] -= y0_o

            in_bounds = (
                (trajs_2d[..., 0] >= 0)
                & (trajs_2d[..., 0] < cw_o)
                & (trajs_2d[..., 1] >= 0)
                & (trajs_2d[..., 1] < ch_o)
            )

            valids = traj_data["valids"].copy()
            visibs = traj_data["visibs"].copy()
            valids &= in_bounds
            visibs &= in_bounds

            intrinsics[:, 0, 2] -= x0_o
            intrinsics[:, 1, 2] -= y0_o

            traj_data = {
                "trajs_2d": trajs_2d,
                "trajs_3d": traj_data["trajs_3d"],
                "valids": valids,
                "visibs": visibs,
            }
            cam_data = {
                "intrinsics": intrinsics,
                "extrinsics": cam_data["extrinsics"],
            }
            orig_size = (cw_o, ch_o)

        # Get image dimensions (after possible augmentation)
        C, T, H, W = images.shape

        # Sample queries and compute targets
        queries, targets = self.sampler.sample(
            trajs_2d=traj_data["trajs_2d"],
            trajs_3d=traj_data["trajs_3d"],
            valids=traj_data["valids"],
            visibs=traj_data["visibs"],
            intrinsics=cam_data["intrinsics"],
            extrinsics=cam_data["extrinsics"],
            frame_indices=frame_indices,
            image_size=(W, H),
            orig_size=orig_size,
            depths=depths,
            normals=normals,
        )

        # Build meta
        meta = {
            "aspect_ratio": float(orig_size[0]) / float(orig_size[1]),
            "img_patch_size": self.config.img_patch_size,
            "align_corners": self.config.align_corners,
        }

        return {
            "meta": meta,
            "images": images,
            "query": queries,
            "targets": targets,
        }

    def _ensure_num_frames(self, images: Tensor) -> Tensor:
        """Ensure the images tensor has exactly num_frames frames.

        Args:
            images: (C, T, H, W) tensor.

        Returns:
            (C, num_frames, H, W) tensor.
        """
        C, T, H, W = images.shape

        if T == self.config.num_frames:
            return images
        elif T > self.config.num_frames:
            return images[:, :self.config.num_frames]
        else:
            # Pad by repeating last frame
            pad_count = self.config.num_frames - T
            last_frame = images[:, -1:].expand(C, pad_count, H, W)
            return torch.cat([images, last_frame], dim=1)

    @abstractmethod
    def _load_scene_list(self) -> List[Dict[str, Any]]:
        """Load list of available scenes.

        Returns:
            List of scene info dictionaries with at least 'scene_id' key.
        """
        pass

    @abstractmethod
    def _load_trajectories(self, scene_id: str) -> Dict[str, np.ndarray]:
        """Load trajectory data for a scene.

        Args:
            scene_id: Scene identifier.

        Returns:
            Dictionary with keys:
                - trajs_2d: (T, N, 2) 2D trajectories in pixel coordinates
                - trajs_3d: (T, N, 3) 3D trajectories in world coordinates
                - valids: (T, N) validity mask
                - visibs: (T, N) visibility mask
        """
        pass

    @abstractmethod
    def _load_camera_params(self, scene_id: str) -> Dict[str, np.ndarray]:
        """Load camera parameters for a scene.

        Args:
            scene_id: Scene identifier.

        Returns:
            Dictionary with keys:
                - intrinsics: (T, 3, 3) camera intrinsic matrices
                - extrinsics: (T, 4, 4) world-to-camera (w2c) matrices
        """
        pass

    @abstractmethod
    def _load_frames(self, scene_id: str, frame_indices: List[int]) -> Tensor:
        """Load image frames for a scene.

        Args:
            scene_id: Scene identifier.
            frame_indices: List of frame indices to load.

        Returns:
            (C, T, H, W) tensor with loaded frames.
        """
        pass

    def _load_frames_raw(self, scene_id: str, frame_indices: List[int]) -> Tensor:
        """Load image frames at original resolution (optional override)."""
        return self._load_frames(scene_id, frame_indices)

    def _load_depths(
        self,
        scene_id: str,
        frame_indices: List[int],
    ) -> Optional[np.ndarray]:
        """Load depth maps for a scene (optional).

        Args:
            scene_id: Scene identifier.
            frame_indices: List of frame indices to load.

        Returns:
            (T, H, W) depth array or None if not available.
        """
        return None

    def _load_depths_raw(
        self,
        scene_id: str,
        frame_indices: List[int],
    ) -> Optional[np.ndarray]:
        """Load depth maps at original resolution (optional override)."""
        return self._load_depths(scene_id, frame_indices)

    def _load_normals(
        self,
        scene_id: str,
        frame_indices: List[int],
    ) -> Optional[np.ndarray]:
        """Load normal maps for a scene (optional).

        Args:
            scene_id: Scene identifier.
            frame_indices: List of frame indices to load.

        Returns:
            (T, H, W, 3) normal array or None if not available.
        """
        return None

    def _load_normals_raw(
        self,
        scene_id: str,
        frame_indices: List[int],
    ) -> Optional[np.ndarray]:
        """Load normal maps at original resolution (optional override)."""
        return self._load_normals(scene_id, frame_indices)

    def get_scene_info(self, idx: int) -> Dict[str, Any]:
        """Get scene information without loading all data.

        Args:
            idx: Dataset index.

        Returns:
            Scene info dictionary.
        """
        return self.scenes[idx]


class Base3DDataset(Dataset, ABC):
    """Abstract base class for 3D datasets using covisibility.

    3D datasets use covisibility matrices for frame sampling instead
    of dense trajectories. This class provides the interface for such
    datasets.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.scenes = []
        self._setup()

    @abstractmethod
    def _setup(self) -> None:
        """Initialize the dataset."""
        pass

    @abstractmethod
    def _load_scene_list(self) -> List[Dict[str, Any]]:
        """Load list of available scenes."""
        pass

    @abstractmethod
    def _load_covisibility(self, scene_id: str) -> np.ndarray:
        """Load covisibility matrix for frame sampling.

        Args:
            scene_id: Scene identifier.

        Returns:
            (N_frames, N_frames) covisibility matrix.
        """
        pass

    @abstractmethod
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a sample from the dataset."""
        pass

    def __len__(self) -> int:
        return len(self.scenes)
