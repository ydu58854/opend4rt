"""Dataset utilities for the D4RT project.

This module provides dataset implementations for D4RT training:

- D4RTDataset: JSON-based dataset for pre-processed annotations
- PointOdysseyDataset: Direct loading from PointOdyssey with trajectory annotations
- Base4DTrajectoryDataset: Base class for 4D trajectory-based datasets

Usage:
    # For PointOdyssey dataset
    from opend4rt.datasets import PointOdysseyDataset, PointOdysseyConfig

    config = PointOdysseyConfig(
        root="/path/to/pointodyssey",
        split="train",
        num_frames=48,
        num_queries=2048,
    )
    dataset = PointOdysseyDataset(config)

    # Use with DataLoader
    from opend4rt.datasets import d4rt_collate_fn
    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset, batch_size=2, collate_fn=d4rt_collate_fn)

    # For adding new datasets, inherit from Base4DTrajectoryDataset
    from opend4rt.datasets import Base4DTrajectoryDataset, Base4DDatasetConfig
"""

# Original D4RT dataset
from .d4rt_dataset import (
    D4RTDataset,
    DatasetConfig,
    build_dataloader,
    d4rt_collate_fn,
)

# Base configuration classes
from .base_config import (
    BaseDatasetConfig,
    Base4DDatasetConfig,
    PointOdysseyConfig,
    DATASET_PATHS,
    get_dataset_path,
)

# Base dataset classes
from .base_dataset import (
    Base4DTrajectoryDataset,
    Base3DDataset,
)

# PointOdyssey dataset
from .pointodyssey import (
    PointOdysseyDataset,
    PointOdysseySingleFrameDataset,
)

# Query samplers
from .trajectory_sampler import (
    TrajectoryQuerySampler,
    SingleFrameTrajectoryQuerySampler,
)

# Utilities
from .utils import (
    # Image loading
    load_image,
    load_image_tensor,
    load_depth_16bit,
    load_depth_tensor,
    # Resize
    resize_image,
    resize_depth,
    # Coordinate transformations
    normalize_uv,
    denormalize_uv,
    world_to_camera,
    camera_to_world,
    transform_points,
    project_to_image,
    # Frame sampling
    uniform_sample_frames,
    create_frame_index_mapping,
    # Intrinsics
    build_intrinsics_matrix,
    scale_intrinsics,
    # Disparity
    compute_disparity_2d,
    compute_disparity_3d,
    # Normals
    estimate_normals_from_depth,
    sample_normals_at_points,
    # Edge detection
    sobel_edge_detection,
)

__all__ = [
    # Original D4RT dataset
    "D4RTDataset",
    "DatasetConfig",
    "build_dataloader",
    "d4rt_collate_fn",
    # Base configuration classes
    "BaseDatasetConfig",
    "Base4DDatasetConfig",
    "PointOdysseyConfig",
    "DATASET_PATHS",
    "get_dataset_path",
    # Base dataset classes
    "Base4DTrajectoryDataset",
    "Base3DDataset",
    # PointOdyssey
    "PointOdysseyDataset",
    "PointOdysseySingleFrameDataset",
    # Query samplers
    "TrajectoryQuerySampler",
    "SingleFrameTrajectoryQuerySampler",
    # Utilities - Image loading
    "load_image",
    "load_image_tensor",
    "load_depth_16bit",
    "load_depth_tensor",
    # Utilities - Resize
    "resize_image",
    "resize_depth",
    # Utilities - Coordinates
    "normalize_uv",
    "denormalize_uv",
    "world_to_camera",
    "camera_to_world",
    "transform_points",
    "project_to_image",
    # Utilities - Frame sampling
    "uniform_sample_frames",
    "create_frame_index_mapping",
    # Utilities - Intrinsics
    "build_intrinsics_matrix",
    "scale_intrinsics",
    # Utilities - Disparity
    "compute_disparity_2d",
    "compute_disparity_3d",
    # Utilities - Normals
    "estimate_normals_from_depth",
    "sample_normals_at_points",
    # Utilities - Edge detection
    "sobel_edge_detection",
]
