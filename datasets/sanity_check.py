"""Sanity checks for dataset utilities (no data required)."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import datasets.utils as utils  # type: ignore
else:
    from . import utils


def run() -> None:
    torch.manual_seed(0)

    # Random stride sampling
    indices = utils.sample_random_stride_frames(
        total_frames=120,
        target_frames=48,
        min_stride=1,
        max_stride=4,
    )
    assert len(indices) == 48
    assert all(indices[i] < indices[i + 1] for i in range(len(indices) - 1))

    # Video augmentation pipeline shapes
    t, c, h, w = 48, 3, 360, 640
    video = torch.rand(t, c, h, w)
    depth = torch.rand(t, h, w)
    normal = torch.rand(t, h, w, 3) * 2.0 - 1.0

    video = utils.apply_temporal_color_jitter(
        video, brightness=1.1, contrast=0.9, saturation=1.2, hue=0.05
    )
    video = utils.apply_temporal_color_drop(video, prob=1.0)
    video = utils.apply_gaussian_blur(video, prob=1.0, sigma_min=0.5, sigma_max=0.5)

    crop_params = utils.random_resized_crop_params(
        height=h,
        width=w,
        scale=(0.3, 1.0),
        ratio=(3.0 / 4.0, 4.0 / 3.0),
        zoom_prob=0.0,
        zoom_scale=(0.5, 0.9),
    )
    video, depth, normal = utils.apply_random_resized_crop(
        video,
        depth,
        normal,
        crop_params=crop_params,
        output_size=(512, 288),
    )

    assert video.shape == (t, c, 288, 512)
    assert depth.shape == (t, 288, 512)
    assert normal.shape == (t, 288, 512, 3)


if __name__ == "__main__":
    run()
