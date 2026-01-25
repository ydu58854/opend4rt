"""Dataset utilities for D4RT training.

This module provides a flexible dataset implementation that reads frames/images
and per-query annotations, assembles meta/images/query/targets, and exposes
DataLoader helpers with a padded collate function.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import json

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

try:
    from PIL import Image
except ImportError as exc:  # pragma: no cover - handled by import error
    raise ImportError("PIL is required for image loading. Install pillow.") from exc


@dataclass(frozen=True)
class DatasetConfig:
    """Configuration options for D4RTDataset."""

    img_patch_size: int = 3
    align_corners: bool = True
    num_frames: int = 48
    image_channels: int = 3
    normalize: bool = True


class D4RTDataset(Dataset):
    """Dataset for D4RT training.

    Expected annotation format (JSON):

    {
      "samples": [
        {
          "id": "sample-0001",
          "frames": ["frames/000001.jpg", "frames/000002.jpg", ...],
          "queries": [
            {
              "uv": [0.12, 0.34],
              "t_src": 3,
              "t_tgt": 10,
              "t_cam": 0,
              "xyz": [0.0, 0.1, 0.2],
              "uv_target": [0.13, 0.35],
              "vis": 1.0,
              "disp": [0.01, 0.02, 0.03],
              "conf": 0.9,
              "normal": [0.0, 0.0, 1.0]
            }
          ],
          "meta": {"aspect_ratio": 1.777}
        }
      ]
    }

    Alternate inputs:
    * "image": single image path (will be repeated to num_frames if needed).
    * "video": video path (requires torchvision.io.read_video).
    * "query": list of [u, v, t_src, t_tgt, t_cam] entries.
    * "targets": list of per-query dicts with xyz/uv_target/vis/disp/conf/normal.
    """

    def __init__(
        self,
        root: str | Path,
        annotation_file: str | Path,
        config: DatasetConfig | None = None,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.annotation_file = Path(annotation_file)
        self.config = config or DatasetConfig()
        self.samples = self._load_annotations(self.annotation_file)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample = self.samples[index]
        frames = self._load_frames(sample)
        frames = self._ensure_num_frames(frames)
        images = self._frames_to_tensor(frames)
        aspect_ratio = self._get_aspect_ratio(sample, images)

        queries, targets = self._parse_queries_and_targets(sample)

        meta: Dict[str, Any] = {
            "aspect_ratio": aspect_ratio,
            "img_patch_size": self.config.img_patch_size,
            "align_corners": self.config.align_corners,
        }
        return {
            "meta": meta,
            "images": images,
            "query": queries,
            "targets": targets,
        }

    def _load_annotations(self, path: Path) -> List[Mapping[str, Any]]:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict) and "samples" in payload:
            return payload["samples"]
        raise ValueError("Annotation file must be a list or contain a 'samples' key.")

    def _load_frames(self, sample: Mapping[str, Any]) -> List[Image.Image]:
        if "frames" in sample:
            frame_paths = [self.root / Path(p) for p in sample["frames"]]
            return [self._open_image(path) for path in frame_paths]

        if "image" in sample:
            image = self._open_image(self.root / Path(sample["image"]))
            return [image]

        if "video" in sample:
            return self._load_video_frames(self.root / Path(sample["video"]))

        raise KeyError("Sample must contain 'frames', 'image', or 'video'.")

    def _open_image(self, path: Path) -> Image.Image:
        image = Image.open(path).convert("RGB")
        return image

    def _load_video_frames(self, path: Path) -> List[Image.Image]:
        try:
            from torchvision.io import read_video
        except ImportError as exc:
            raise ImportError(
                "torchvision is required to load video files."
            ) from exc

        frames, _, _ = read_video(str(path), pts_unit="sec")
        if frames.numel() == 0:
            raise ValueError(f"No frames found in video: {path}")
        # frames: (T, H, W, C)
        return [Image.fromarray(frame.numpy()) for frame in frames]

    def _ensure_num_frames(self, frames: List[Image.Image]) -> List[Image.Image]:
        if len(frames) == self.config.num_frames:
            return frames
        if len(frames) == 1:
            return frames * self.config.num_frames
        if len(frames) > self.config.num_frames:
            return frames[: self.config.num_frames]
        # pad by repeating last frame
        pad = [frames[-1]] * (self.config.num_frames - len(frames))
        return frames + pad

    def _frames_to_tensor(self, frames: Sequence[Image.Image]) -> Tensor:
        tensor_frames = []
        for frame in frames:
            frame_array = np.array(frame, dtype=np.uint8)
            frame_tensor = torch.from_numpy(frame_array).permute(2, 0, 1).contiguous()
            if self.config.normalize:
                frame_tensor = frame_tensor.float() / 255.0
            tensor_frames.append(frame_tensor)

        stacked = torch.stack(tensor_frames, dim=0)  # T,C,H,W
        return stacked.permute(1, 0, 2, 3)  # C,T,H,W

    def _get_aspect_ratio(self, sample: Mapping[str, Any], images: Tensor) -> float:
        if "meta" in sample and "aspect_ratio" in sample["meta"]:
            return float(sample["meta"]["aspect_ratio"])
        _, _, height, width = images.shape
        return float(width) / float(height)

    def _parse_queries_and_targets(
        self, sample: Mapping[str, Any]
    ) -> tuple[Tensor, Dict[str, Tensor]]:
        queries: List[List[float]] = []
        l3d: List[List[float]] = []
        l2d: List[List[float]] = []
        lvis: List[List[float]] = []
        ldisp: List[List[float]] = []
        lconf: List[List[float]] = []
        lnormal: List[List[float]] = []

        if "queries" in sample:
            entries = sample["queries"]
            for entry in entries:
                queries.append(self._extract_query(entry))
                l3d.append(self._extract_vector(entry, "xyz", 3))
                l2d.append(self._extract_vector(entry, "uv_target", 2))
                lvis.append([self._extract_scalar(entry, "vis")])
                ldisp.append(self._extract_vector(entry, "disp", 3))
                lconf.append([self._extract_scalar(entry, "conf")])
                lnormal.append(self._extract_vector(entry, "normal", 3))
        elif "query" in sample:
            for entry in sample["query"]:
                if not isinstance(entry, (list, tuple)) or len(entry) != 5:
                    raise ValueError("'query' entries must be length-5 lists.")
                queries.append([float(v) for v in entry])
            targets = sample.get("targets", [])
            if targets and len(targets) != len(queries):
                raise ValueError("'targets' must have the same length as 'query'.")
            for target in targets:
                l3d.append(self._extract_vector(target, "xyz", 3))
                l2d.append(self._extract_vector(target, "uv_target", 2))
                lvis.append([self._extract_scalar(target, "vis")])
                ldisp.append(self._extract_vector(target, "disp", 3))
                lconf.append([self._extract_scalar(target, "conf")])
                lnormal.append(self._extract_vector(target, "normal", 3))
        else:
            raise KeyError("Sample must contain 'queries' or 'query'.")

        query_tensor = torch.tensor(queries, dtype=torch.float32)
        targets = {
            "L3D": torch.tensor(l3d, dtype=torch.float32),
            "L2D": torch.tensor(l2d, dtype=torch.float32),
            "Lvis": torch.tensor(lvis, dtype=torch.float32),
            "Ldisp": torch.tensor(ldisp, dtype=torch.float32),
            "Lconf": torch.tensor(lconf, dtype=torch.float32),
            "Lnormal": torch.tensor(lnormal, dtype=torch.float32),
        }
        return query_tensor, targets

    @staticmethod
    def _extract_query(entry: Mapping[str, Any]) -> List[float]:
        if "uv" in entry:
            u, v = entry["uv"]
        else:
            u = entry.get("u")
            v = entry.get("v")
        if u is None or v is None:
            raise ValueError("Query requires 'uv' or 'u'/'v' entries.")
        return [
            float(u),
            float(v),
            float(entry.get("t_src", 0)),
            float(entry.get("t_tgt", 0)),
            float(entry.get("t_cam", 0)),
        ]

    @staticmethod
    def _extract_scalar(entry: Mapping[str, Any], key: str) -> float:
        value = entry.get(key)
        return float(value) if value is not None else 0.0

    @staticmethod
    def _extract_vector(entry: Mapping[str, Any], key: str, size: int) -> List[float]:
        value = entry.get(key)
        if value is None:
            return [0.0] * size
        if len(value) != size:
            raise ValueError(f"Expected '{key}' to have {size} values.")
        return [float(v) for v in value]


def d4rt_collate_fn(batch: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Collate function that pads variable-length queries."""
    metas = [item["meta"] for item in batch]
    images = torch.stack([item["images"] for item in batch], dim=0)
    queries = [item["query"] for item in batch]
    targets_list = [item["targets"] for item in batch]

    max_queries = max(q.shape[0] for q in queries)
    query_dim = queries[0].shape[-1]
    padded_queries = torch.zeros((len(batch), max_queries, query_dim), dtype=queries[0].dtype)
    query_mask = torch.zeros((len(batch), max_queries), dtype=torch.bool)

    for idx, q in enumerate(queries):
        padded_queries[idx, : q.shape[0]] = q
        query_mask[idx, : q.shape[0]] = True

    def pad_targets(key: str, dim: int) -> Tensor:
        padded = torch.zeros((len(batch), max_queries, dim), dtype=targets_list[0][key].dtype)
        for idx, tgt in enumerate(targets_list):
            padded[idx, : tgt[key].shape[0]] = tgt[key]
        return padded

    targets = {
        "L3D": pad_targets("L3D", 3),
        "L2D": pad_targets("L2D", 2),
        "Lvis": pad_targets("Lvis", 1),
        "Ldisp": pad_targets("Ldisp", 3),
        "Lconf": pad_targets("Lconf", 1),
        "Lnormal": pad_targets("Lnormal", 3),
        "query_mask": query_mask,
    }

    meta_batch = {
        "aspect_ratio": torch.tensor([m["aspect_ratio"] for m in metas], dtype=torch.float32),
        "img_patch_size": torch.tensor([m["img_patch_size"] for m in metas], dtype=torch.int64),
        "align_corners": torch.tensor([m["align_corners"] for m in metas], dtype=torch.bool),
    }

    return {
        "meta": meta_batch,
        "images": images,
        "query": padded_queries,
        "targets": targets,
    }


def build_dataloader(
    dataset: Dataset,
    batch_size: int = 2,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> DataLoader:
    """Build a DataLoader with the default collate function."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=d4rt_collate_fn,
    )
