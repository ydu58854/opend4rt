"""Trajectory-based query sampling for 4D datasets.

This module provides query sampling strategies based on trajectory data,
designed for 4D datasets like PointOdyssey.
"""
from __future__ import annotations

from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
from torch import Tensor

from .utils import (
    normalize_uv,
    world_to_camera,
    compute_disparity_2d,
    sobel_edge_detection,
    compute_motion_boundaries,
)


class TrajectoryQuerySampler:
    """Query sampler for trajectory-based 4D datasets.

    Samples queries and targets from dense point trajectories following
    the D4RT paper (Appendix A):
    - tsrc, ttgt, tcam are sampled uniformly at random
    - ttgt = tcam with probability tcam_equals_ttgt_ratio (default 0.4)
    - Edge sampling: prioritize 30% of queries near depth edges

    Attributes:
        num_queries: Number of queries to sample per sample.
        edge_ratio: Fraction of queries from depth edges (default 0.3).
        min_visible_frames: Minimum frames a trajectory must be visible.
        tcam_equals_ttgt_ratio: Probability that t_cam = t_tgt (default 0.4, per paper).
    """

    def __init__(
        self,
        num_queries: int = 2048,
        edge_ratio: float = 0.3,
        min_visible_frames: int = 8,
        tcam_equals_ttgt_ratio: float = 0.4,
    ):
        self.num_queries = num_queries
        self.edge_ratio = edge_ratio
        self.min_visible_frames = min_visible_frames
        self.tcam_equals_ttgt_ratio = tcam_equals_ttgt_ratio

    def sample(
        self,
        trajs_2d: np.ndarray,
        trajs_3d: np.ndarray,
        valids: np.ndarray,
        visibs: np.ndarray,
        intrinsics: np.ndarray,
        extrinsics: np.ndarray,
        frame_indices: List[int],
        image_size: Tuple[int, int],
        orig_size: Tuple[int, int],
        depths: Optional[np.ndarray] = None,
        normals: Optional[np.ndarray] = None,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Sample queries and compute targets from trajectory data.

        Args:
            trajs_2d: 2D trajectory points (T_orig, N, 2) in ORIGINAL pixel coordinates.
            trajs_3d: 3D trajectory points (T_orig, N, 3) in world coordinates.
            valids: Validity mask (T_orig, N).
            visibs: Visibility mask (T_orig, N).
            intrinsics: Camera intrinsics (T_orig, 3, 3) - ORIGINAL, not scaled.
            extrinsics: Camera extrinsics w2c (T_orig, 4, 4).
            frame_indices: List of sampled frame indices (length 48).
            image_size: Target image size (W, H).
            orig_size: Original image size (orig_W, orig_H).
            depths: Optional depth maps (T_sampled, H, W) for edge detection.
            normals: Optional normal maps (T_sampled, H, W, 3).

        Returns:
            queries: (N, 5) tensor [u, v, t_src, t_tgt, t_cam].
            targets: dict with L3D, L2D, Lvis, Ldisp, Lconf, Lnormal.
        """
        W, H = image_size
        orig_W, orig_H = orig_size
        T = len(frame_indices)

        # Scale factors from original to target resolution
        scale_x = W / orig_W
        scale_y = H / orig_H

        # Extract data for sampled frames
        trajs_2d_sampled = trajs_2d[frame_indices]  # (T, N, 2) - still in original pixel coords
        trajs_3d_sampled = trajs_3d[frame_indices]  # (T, N, 3)
        valids_sampled = valids[frame_indices]      # (T, N)
        visibs_sampled = visibs[frame_indices]      # (T, N)
        intrinsics_sampled = intrinsics[frame_indices].copy()  # (T, 3, 3)
        extrinsics_sampled = extrinsics[frame_indices]  # (T, 4, 4)

        # Scale intrinsics to target resolution
        intrinsics_sampled[:, 0, 0] *= scale_x  # fx
        intrinsics_sampled[:, 1, 1] *= scale_y  # fy
        intrinsics_sampled[:, 0, 2] *= scale_x  # cx
        intrinsics_sampled[:, 1, 2] *= scale_y  # cy

        # Filter valid trajectories
        valid_traj_mask = self._filter_valid_trajectories(
            valids_sampled, visibs_sampled
        )

        # Sample query pairs following D4RT paper:
        # - 30% from depth/motion edges, 70% random
        # - tsrc, ttgt sampled uniformly at random
        # - tcam = ttgt with probability 0.4, otherwise random
        n_edge = int(self.num_queries * self.edge_ratio)
        n_rand = self.num_queries - n_edge

        all_queries = []

        # Compute motion boundaries from trajectories
        motion_edges = compute_motion_boundaries(
            trajs_2d_sampled,
            valids_sampled,
            visibs_sampled,
            image_size=(W, H),
            threshold=0.1,
        )

        # Edge sampling (with depth edge and motion boundary preference)
        if n_edge > 0:
            edge_queries = self._sample_edge_queries(
                trajs_2d_sampled, valids_sampled, visibs_sampled,
                valid_traj_mask, n_edge, W, H, orig_W, orig_H, depths, motion_edges
            )
            all_queries.extend(edge_queries)

        # Random sampling (tsrc, ttgt uniformly at random)
        if n_rand > 0:
            rand_queries = self._sample_random_queries(
                trajs_2d_sampled, valids_sampled, visibs_sampled,
                valid_traj_mask, n_rand, W, H
            )
            all_queries.extend(rand_queries)

        # Handle empty queries case
        if len(all_queries) == 0:
            return (
                torch.zeros((0, 5), dtype=torch.float32),
                {
                    "L3D": torch.zeros((0, 3), dtype=torch.float32),
                    "L2D": torch.zeros((0, 2), dtype=torch.float32),
                    "Lvis": torch.zeros((0, 1), dtype=torch.float32),
                    "Ldisp": torch.zeros((0, 3), dtype=torch.float32),
                    "Lconf": torch.zeros((0, 1), dtype=torch.float32),
                    "Lnormal": torch.zeros((0, 3), dtype=torch.float32),
                }
            )

        # Convert to arrays
        queries_array = np.array(all_queries)  # (N, 3): [t_src, t_tgt, point_idx]

        # Compute all targets
        queries, targets = self._compute_targets(
            queries_array,
            trajs_2d_sampled,
            trajs_3d_sampled,
            valids_sampled,
            visibs_sampled,
            extrinsics_sampled,
            W, H,
            orig_W, orig_H,
            normals,
        )

        return queries, targets

    def _filter_valid_trajectories(
        self,
        valids: np.ndarray,
        visibs: np.ndarray,
    ) -> np.ndarray:
        """Filter trajectories that are visible in enough frames.

        Args:
            valids: Validity mask (T, N).
            visibs: Visibility mask (T, N).

        Returns:
            Boolean mask (N,) indicating valid trajectories.
        """
        # Count visible frames for each trajectory
        visible_counts = np.sum(valids & visibs, axis=0)  # (N,)
        return visible_counts >= self.min_visible_frames

    def _sample_edge_queries(
        self,
        trajs_2d: np.ndarray,
        valids: np.ndarray,
        visibs: np.ndarray,
        valid_traj_mask: np.ndarray,
        num_samples: int,
        W: int,
        H: int,
        orig_W: int,
        orig_H: int,
        depths: Optional[np.ndarray] = None,
        motion_edges: Optional[np.ndarray] = None,
    ) -> List[Tuple[int, int, int]]:
        """Sample queries from depth discontinuities and motion boundaries.

        Args:
            trajs_2d: 2D trajectories in ORIGINAL pixel coordinates.
            W, H: Target image dimensions.
            orig_W, orig_H: Original image dimensions.
            depths: Depth maps at TARGET resolution.
            motion_edges: Motion boundary maps at TARGET resolution (T, H, W).

        Returns:
            List of (t_src, t_tgt, point_idx) tuples.
        """
        T, N, _ = trajs_2d.shape
        queries = []

        # Scale factors from original to target resolution
        scale_x = W / orig_W
        scale_y = H / orig_H

        # Compute combined edge map from depth and motion
        edges = None

        # Depth edges
        if depths is not None:
            depth_tensor = torch.from_numpy(depths).float()
            depth_edges = sobel_edge_detection(depth_tensor, threshold=0.1)
            edges = depth_edges.numpy()  # (T, H, W)

        # Motion edges
        if motion_edges is not None:
            if edges is not None:
                # Combine depth and motion edges (logical OR)
                edges = np.maximum(edges, motion_edges)
            else:
                edges = motion_edges

        for _ in range(num_samples):
            # Randomly select source frame
            t_src = np.random.randint(0, T)

            # Find valid visible points in this frame
            frame_mask = valids[t_src] & visibs[t_src] & valid_traj_mask
            valid_points = np.where(frame_mask)[0]

            if len(valid_points) == 0:
                continue

            # If we have edges, prefer points near edges
            if edges is not None:
                # Get pixel coordinates and scale to target resolution
                px = (trajs_2d[t_src, valid_points, 0] * scale_x).astype(int)
                py = (trajs_2d[t_src, valid_points, 1] * scale_y).astype(int)

                # Clamp to valid range
                px = np.clip(px, 0, W - 1)
                py = np.clip(py, 0, H - 1)

                # Check which points are near edges (depth OR motion boundaries)
                edge_scores = edges[t_src, py, px]
                edge_mask = edge_scores > 0.5

                if edge_mask.sum() > 0:
                    # Sample from edge points
                    edge_points = valid_points[edge_mask]
                    point_idx = np.random.choice(edge_points)
                else:
                    # Fallback to random valid point
                    point_idx = np.random.choice(valid_points)
            else:
                point_idx = np.random.choice(valid_points)

            # Select target frame uniformly at random
            t_tgt = np.random.randint(0, T)

            queries.append((t_src, t_tgt, point_idx))

        return queries

    def _sample_random_queries(
        self,
        trajs_2d: np.ndarray,
        valids: np.ndarray,
        visibs: np.ndarray,
        valid_traj_mask: np.ndarray,
        num_samples: int,
        W: int,
        H: int,
    ) -> List[Tuple[int, int, int]]:
        """Sample random queries uniformly.

        Returns:
            List of (t_src, t_tgt, point_idx) tuples.
        """
        T, N, _ = trajs_2d.shape
        queries = []

        for _ in range(num_samples):
            # Random source frame
            t_src = np.random.randint(0, T)

            # Find valid visible points at source
            frame_mask = valids[t_src] & visibs[t_src] & valid_traj_mask
            valid_points = np.where(frame_mask)[0]

            if len(valid_points) == 0:
                continue

            point_idx = np.random.choice(valid_points)

            # Random target frame
            t_tgt = np.random.randint(0, T)
            queries.append((t_src, t_tgt, point_idx))

        return queries

    def _compute_targets(
        self,
        queries_array: np.ndarray,
        trajs_2d: np.ndarray,
        trajs_3d: np.ndarray,
        valids: np.ndarray,
        visibs: np.ndarray,
        extrinsics: np.ndarray,
        W: int,
        H: int,
        orig_W: int,
        orig_H: int,
        normals: Optional[np.ndarray] = None,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Compute query vectors and all target values.

        Args:
            queries_array: (N, 3) array with [t_src, t_tgt, point_idx].
            trajs_2d: Sampled 2D trajectories (T, N_points, 2) in ORIGINAL pixel coordinates.
            trajs_3d: Sampled 3D trajectories (T, N_points, 3).
            valids: Validity mask (T, N_points).
            visibs: Visibility mask (T, N_points).
            extrinsics: Camera extrinsics w2c (T, 4, 4).
            W, H: Target image dimensions.
            orig_W, orig_H: Original image dimensions.
            normals: Optional normal maps (T, H, W, 3).

        Returns:
            queries: (N, 5) tensor.
            targets: dict with all target tensors.
        """
        N = len(queries_array)
        T = trajs_2d.shape[0]  # Number of sampled frames

        # Scale factors from original to target resolution
        scale_x = W / orig_W
        scale_y = H / orig_H

        # Initialize outputs
        query_list = []
        l3d_list = []
        l2d_list = []
        lvis_list = []
        ldisp_list = []
        lconf_list = []
        lnormal_list = []

        for i in range(N):
            t_src = int(queries_array[i, 0])
            t_tgt = int(queries_array[i, 1])
            point_idx = int(queries_array[i, 2])

            # D4RT paper Appendix A: "we enforce ttgt = tcam with probability 0.4"
            # Otherwise, t_cam is sampled uniformly at random
            if np.random.random() < self.tcam_equals_ttgt_ratio:
                t_cam = t_tgt
            else:
                t_cam = np.random.randint(0, T)

            # Source UV (normalized using ORIGINAL resolution since trajs_2d is in original pixels)
            px_src, py_src = trajs_2d[t_src, point_idx]
            u_src, v_src = normalize_uv(
                np.array([px_src]), np.array([py_src]), orig_W, orig_H
            )
            u_src, v_src = float(u_src[0]), float(v_src[0])

            # Clamp UV to [0, 1]
            u_src = np.clip(u_src, 0.0, 1.0)
            v_src = np.clip(v_src, 0.0, 1.0)

            # Query vector: [u, v, t_src, t_tgt, t_cam]
            query_list.append([u_src, v_src, float(t_src), float(t_tgt), float(t_cam)])

            # Target UV (L2D) - normalized using ORIGINAL resolution
            px_tgt_orig, py_tgt_orig = trajs_2d[t_tgt, point_idx]
            u_tgt, v_tgt = normalize_uv(
                np.array([px_tgt_orig]), np.array([py_tgt_orig]), orig_W, orig_H
            )
            u_tgt, v_tgt = float(u_tgt[0]), float(v_tgt[0])
            u_tgt = np.clip(u_tgt, 0.0, 1.0)
            v_tgt = np.clip(v_tgt, 0.0, 1.0)
            l2d_list.append([u_tgt, v_tgt])

            # Scale pixel coordinates to target resolution for normal map sampling
            px_tgt = px_tgt_orig * scale_x
            py_tgt = py_tgt_orig * scale_y

            # L3D: World coordinates to camera coordinates
            xyz_world = trajs_3d[t_tgt, point_idx]
            w2c = extrinsics[t_cam]
            xyz_cam = world_to_camera(xyz_world, w2c)
            l3d_list.append(xyz_cam.tolist())

            # Visibility (Lvis)
            vis = float(visibs[t_tgt, point_idx])
            lvis_list.append([vis])

            # Disparity (Ldisp): difference in UV
            disp = compute_disparity_2d(
                np.array([[u_src, v_src]]),
                np.array([[u_tgt, v_tgt]])
            )
            ldisp_list.append(disp[0].tolist())

            # Confidence (Lconf): based on validity and visibility
            valid_src = valids[t_src, point_idx]
            valid_tgt = valids[t_tgt, point_idx]
            vis_src = visibs[t_src, point_idx]
            vis_tgt = visibs[t_tgt, point_idx]
            conf = float(valid_src and valid_tgt and vis_src and vis_tgt)
            lconf_list.append([conf])

            # Normal (Lnormal)
            # Note: PointOdyssey normal maps are in OpenGL camera space (+Z toward camera)
            # D4RT expects OpenCV camera space (+Z forward, facing normals have Z < 0)
            # Conversion: Y_opencv = -Y_opengl, Z_opencv = -Z_opengl
            if normals is not None:
                # Sample normal at target location (in t_tgt's OpenGL camera frame)
                px_int = int(np.clip(px_tgt, 0, W - 1))
                py_int = int(np.clip(py_tgt, 0, H - 1))
                normal = normals[t_tgt, py_int, px_int].copy()
                # Convert from OpenGL to OpenCV camera space
                normal[1] = -normal[1]  # Flip Y
                normal[2] = -normal[2]  # Flip Z
                # Rotate normal from t_tgt camera frame to t_cam camera frame
                # n_world = R_tgt^T @ n_tgt, n_cam = R_cam @ n_world
                if t_cam != t_tgt:
                    R_cam = extrinsics[t_cam, :3, :3]
                    R_tgt = extrinsics[t_tgt, :3, :3]
                    normal = R_cam @ (R_tgt.T @ normal)
                # Normalize to unit vector (normals may not be unit due to JPEG compression or interpolation)
                norm = np.linalg.norm(normal)
                if norm > 1e-6:
                    normal = normal / norm
                else:
                    normal = np.array([0.0, 0.0, 0.0])
                lnormal_list.append(normal.tolist())
            else:
                # Default to zero normal
                lnormal_list.append([0.0, 0.0, 0.0])

        # Convert to tensors
        queries = torch.tensor(query_list, dtype=torch.float32)
        targets = {
            "L3D": torch.tensor(l3d_list, dtype=torch.float32),
            "L2D": torch.tensor(l2d_list, dtype=torch.float32),
            "Lvis": torch.tensor(lvis_list, dtype=torch.float32),
            "Ldisp": torch.tensor(ldisp_list, dtype=torch.float32),
            "Lconf": torch.tensor(lconf_list, dtype=torch.float32),
            "Lnormal": torch.tensor(lnormal_list, dtype=torch.float32),
        }

        return queries, targets


class SingleFrameTrajectoryQuerySampler:
    """Query sampler for single-frame overfitting tests.

    All queries have t_src = t_tgt = t_cam.
    """

    def __init__(
        self,
        num_queries: int = 2048,
        edge_ratio: float = 0.3,
    ):
        self.num_queries = num_queries
        self.edge_ratio = edge_ratio

    def sample(
        self,
        trajs_2d: np.ndarray,
        trajs_3d: np.ndarray,
        valids: np.ndarray,
        visibs: np.ndarray,
        intrinsics: np.ndarray,
        extrinsics: np.ndarray,
        frame_idx: int,
        image_size: Tuple[int, int],
        orig_size: Tuple[int, int],
        depth: Optional[np.ndarray] = None,
        normal: Optional[np.ndarray] = None,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Sample queries from a single frame.

        Args:
            trajs_2d: 2D trajectories (T, N, 2) in ORIGINAL pixel coordinates.
            trajs_3d: 3D trajectories (T, N, 3).
            valids: Validity mask (T, N).
            visibs: Visibility mask (T, N).
            intrinsics: Camera intrinsics (T, 3, 3) - ORIGINAL, not scaled.
            extrinsics: Camera extrinsics (T, 4, 4).
            frame_idx: Frame index to sample from.
            image_size: Target image size (W, H).
            orig_size: Original image size (orig_W, orig_H).
            depth: Optional depth map (H, W) at TARGET resolution.
            normal: Optional normal map (H, W, 3) at TARGET resolution.

        Returns:
            queries: (N, 5) tensor.
            targets: dict with target tensors.
        """
        W, H = image_size
        orig_W, orig_H = orig_size
        t = frame_idx

        # Scale factors from original to target resolution
        scale_x = W / orig_W
        scale_y = H / orig_H

        # Get data for this frame
        trajs_2d_frame = trajs_2d[t]  # (N, 2)
        trajs_3d_frame = trajs_3d[t]  # (N, 3)
        valids_frame = valids[t]      # (N,)
        visibs_frame = visibs[t]      # (N,)

        # Find valid visible points
        valid_mask = valids_frame & visibs_frame
        valid_points = np.where(valid_mask)[0]

        if len(valid_points) == 0:
            raise ValueError(f"No valid points in frame {frame_idx}")

        # Sample points
        n_edge = int(self.num_queries * self.edge_ratio)
        n_rand = self.num_queries - n_edge

        sampled_points = []

        # Edge sampling
        if depth is not None and n_edge > 0:
            depth_tensor = torch.from_numpy(depth).float()
            edges = sobel_edge_detection(depth_tensor, threshold=0.1).numpy()

            # Scale pixel coords from original to target resolution for edge lookup
            px = (trajs_2d_frame[valid_points, 0] * scale_x).astype(int)
            py = (trajs_2d_frame[valid_points, 1] * scale_y).astype(int)
            px = np.clip(px, 0, W - 1)
            py = np.clip(py, 0, H - 1)

            edge_scores = edges[py, px]
            edge_mask = edge_scores > 0.5

            if edge_mask.sum() > 0:
                edge_points = valid_points[edge_mask]
                n_sample = min(n_edge, len(edge_points))
                sampled = np.random.choice(edge_points, n_sample, replace=False)
                sampled_points.extend(sampled.tolist())
                n_edge -= n_sample

        # Fill remaining with random
        remaining = self.num_queries - len(sampled_points)
        if remaining > 0:
            # Sample with replacement if needed
            if len(valid_points) >= remaining:
                sampled = np.random.choice(valid_points, remaining, replace=False)
            else:
                sampled = np.random.choice(valid_points, remaining, replace=True)
            sampled_points.extend(sampled.tolist())

        # Build queries and targets
        query_list = []
        l3d_list = []
        l2d_list = []
        lvis_list = []
        ldisp_list = []
        lconf_list = []
        lnormal_list = []

        for point_idx in sampled_points:
            # Get pixel coords in original resolution
            px_orig, py_orig = trajs_2d_frame[point_idx]
            # Normalize using ORIGINAL resolution (UV is resolution-independent)
            u, v = normalize_uv(np.array([px_orig]), np.array([py_orig]), orig_W, orig_H)
            u, v = float(np.clip(u[0], 0, 1)), float(np.clip(v[0], 0, 1))

            # Query: all same frame
            query_list.append([u, v, float(t), float(t), float(t)])

            # L2D: same as source for same frame
            l2d_list.append([u, v])

            # L3D: world to camera
            xyz_world = trajs_3d_frame[point_idx]
            w2c = extrinsics[t]
            xyz_cam = world_to_camera(xyz_world, w2c)
            l3d_list.append(xyz_cam.tolist())

            # Lvis
            lvis_list.append([float(visibs_frame[point_idx])])

            # Ldisp: zero for same frame
            ldisp_list.append([0.0, 0.0, 0.0])

            # Lconf
            lconf_list.append([float(valids_frame[point_idx] and visibs_frame[point_idx])])

            # Lnormal
            # Convert from OpenGL camera space to OpenCV camera space
            # Scale pixel coords to target resolution for normal map lookup
            if normal is not None:
                px_int = int(np.clip(px_orig * scale_x, 0, W - 1))
                py_int = int(np.clip(py_orig * scale_y, 0, H - 1))
                n = normal[py_int, px_int].copy()
                n[1] = -n[1]  # Flip Y
                n[2] = -n[2]  # Flip Z
                # Normalize to unit vector (normals may not be unit due to JPEG compression or interpolation)
                norm = np.linalg.norm(n)
                if norm > 1e-6:
                    n = n / norm
                else:
                    n = np.array([0.0, 0.0, 0.0])
                lnormal_list.append(n.tolist())
            else:
                lnormal_list.append([0.0, 0.0, 0.0])

        queries = torch.tensor(query_list, dtype=torch.float32)
        targets = {
            "L3D": torch.tensor(l3d_list, dtype=torch.float32),
            "L2D": torch.tensor(l2d_list, dtype=torch.float32),
            "Lvis": torch.tensor(lvis_list, dtype=torch.float32),
            "Ldisp": torch.tensor(ldisp_list, dtype=torch.float32),
            "Lconf": torch.tensor(lconf_list, dtype=torch.float32),
            "Lnormal": torch.tensor(lnormal_list, dtype=torch.float32),
        }

        return queries, targets
