"""Base configuration classes for D4RT datasets.

This module provides base configuration classes that can be extended
for different datasets while maintaining compatibility with D4RT format.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Tuple, Literal


# =============================================================================
# 数据集默认路径配置 - 在这里修改默认路径
# =============================================================================

DATASET_PATHS = {
    "pointodyssey": "/inspire/qb-ilm/project/wuliqifa/public/dyh_data/pointodyssey",
    # 后续添加其他数据集路径:
    # "spring": "/path/to/spring",
    # "tapvid": "/path/to/tapvid",
}


def get_dataset_path(dataset_name: str) -> str:
    """获取数据集默认路径。

    Args:
        dataset_name: 数据集名称 (如 "pointodyssey")

    Returns:
        数据集路径字符串

    Raises:
        KeyError: 如果数据集未在 DATASET_PATHS 中定义
    """
    if dataset_name not in DATASET_PATHS:
        raise KeyError(
            f"数据集 '{dataset_name}' 未定义默认路径。"
            f"请在 DATASET_PATHS 中添加，或创建配置时指定 root 参数。"
            f"已定义的数据集: {list(DATASET_PATHS.keys())}"
        )
    return DATASET_PATHS[dataset_name]


# =============================================================================


@dataclass
class BaseDatasetConfig:
    """Base configuration for all D4RT-compatible datasets.

    Attributes:
        root: Root directory of the dataset.
        split: Data split (train/val/test).
        num_frames: Number of frames per sample (D4RT requires 48).
        num_queries: Number of queries to sample per sample.
        target_resolution: Target image resolution (W, H).
        normalize: Whether to normalize images to [0, 1].
        img_patch_size: Patch size for query embedding extraction.
        align_corners: Alignment mode for grid_sample.
    """
    root: str | Path
    split: Literal["train", "val", "test"] = "train"
    num_frames: int = 48
    num_queries: int = 2048
    target_resolution: Tuple[int, int] = (256, 256)  # (W, H)
    normalize: bool = True
    img_patch_size: int = 3
    align_corners: bool = True
    # Temporal sampling
    temporal_random_stride: bool = False
    temporal_stride_min: int = 1
    temporal_stride_max: Optional[int] = None
    # Data augmentation
    use_data_augmentation: bool = False
    color_jitter: Tuple[float, float, float, float] = (0.4, 0.4, 0.4, 0.1)
    color_drop_prob: float = 0.2
    gaussian_blur_prob: float = 0.4
    gaussian_blur_sigma: Tuple[float, float] = (0.1, 2.0)
    random_crop_scale: Tuple[float, float] = (0.3, 1.0)
    random_crop_ratio: Tuple[float, float] = (3.0 / 4.0, 4.0 / 3.0)
    random_zoom_prob: float = 0.05
    random_zoom_scale: Tuple[float, float] = (0.5, 0.9)

    def __post_init__(self):
        self.root = Path(self.root)


@dataclass
class Base4DDatasetConfig(BaseDatasetConfig):
    """Configuration for 4D trajectory-based datasets.

    4D datasets use trajectory data for query sampling instead of
    covisibility matrices.

    Following D4RT paper (Appendix A):
    - tsrc, ttgt, tcam are sampled uniformly at random
    - ttgt = tcam with probability tcam_equals_ttgt_ratio (default 0.4)
    - query_edge_ratio of queries are sampled from depth edges

    Attributes:
        min_visible_frames: Minimum number of frames a trajectory must be visible.
        query_edge_ratio: Fraction of queries sampled from depth edges (default 0.3).
        tcam_equals_ttgt_ratio: Probability that t_cam = t_tgt (default 0.4, per paper).
    """
    min_visible_frames: int = 8
    query_edge_ratio: float = 0.3
    tcam_equals_ttgt_ratio: float = 0.4

    def __post_init__(self):
        super().__post_init__()
        if not 0.0 <= self.query_edge_ratio <= 1.0:
            raise ValueError(f"query_edge_ratio must be in [0, 1], got {self.query_edge_ratio}")
        if not 0.0 <= self.tcam_equals_ttgt_ratio <= 1.0:
            raise ValueError(f"tcam_equals_ttgt_ratio must be in [0, 1], got {self.tcam_equals_ttgt_ratio}")


@dataclass
class PointOdysseyConfig(Base4DDatasetConfig):
    """Configuration for PointOdyssey dataset.

    PointOdyssey is a synthetic 4D dataset with dense point trajectories.

    默认路径可在文件顶部的 DATASET_PATHS["pointodyssey"] 修改。

    Attributes:
        root: 数据集根目录，默认使用 DATASET_PATHS["pointodyssey"]。
        use_depth: Whether to load depth maps.
        use_normals: Whether to load normal maps.
        exclude_scenes: List of scene names to exclude.
        include_scenes: If provided, only include these scenes.
        depth_scale: Scale factor for depth values (16-bit PNG).
            Official: depth_16bit / 65535.0 * 1000.0
    """
    root: str | Path = field(default_factory=lambda: DATASET_PATHS["pointodyssey"])
    use_depth: bool = True
    use_normals: bool = True
    exclude_scenes: List[str] = field(default_factory=list)
    include_scenes: Optional[List[str]] = None
    depth_scale: float = 1000.0  # Official: depth_16bit / 65535.0 * 1000.0
    # Enable augmentation + random stride by default to match D4RT training
    temporal_random_stride: bool = True
    use_data_augmentation: bool = True

    def should_include_scene(self, scene_name: str) -> bool:
        """Check if a scene should be included based on filters."""
        if scene_name in self.exclude_scenes:
            return False
        if self.include_scenes is not None:
            return scene_name in self.include_scenes
        return True
