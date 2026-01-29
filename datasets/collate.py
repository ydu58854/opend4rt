"""Collate utilities for D4RT-compatible datasets."""
from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset


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
