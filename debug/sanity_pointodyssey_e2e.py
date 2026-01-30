"""End-to-end sanity check for PointOdyssey training pipeline.

This script loads one batch from PointOdyssey, runs the model forward,
and computes losses to validate the full data -> model -> loss path.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from datasets import PointOdysseyConfig, PointOdysseyDataset, d4rt_collate_fn  # type: ignore
from model import D4RT, LossHead  # type: ignore
from train import CompositeLoss  # type: ignore


def move_to_device(batch, device):
    meta = {}
    for key, value in batch["meta"].items():
        if torch.is_tensor(value):
            meta[key] = value.to(device)
        else:
            meta[key] = value

    targets = {}
    for key, value in batch["targets"].items():
        targets[key] = value.to(device)

    return {
        "meta": meta,
        "images": batch["images"].to(device),
        "query": batch["query"].to(device),
        "targets": targets,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="E2E sanity check for PointOdyssey")
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-frames", type=int, default=48)
    parser.add_argument("--num-queries", type=int, default=256)
    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--img-patch-size", type=int, default=9)
    parser.add_argument("--encoder-depth", type=int, default=12)
    parser.add_argument("--encoder-embed-dim", type=int, default=768)
    parser.add_argument("--encoder-num-heads", type=int, default=12)
    parser.add_argument("--decoder-depth", type=int, default=8)
    parser.add_argument("--decoder-embed-dim", type=int, default=768)
    parser.add_argument("--decoder-num-heads", type=int, default=8)
    parser.add_argument("--no-aug", action="store_true")
    parser.add_argument("--no-random-stride", action="store_true")
    parser.add_argument("--encoder-pretrained", action="store_true")
    parser.add_argument("--encoder-pretrained-path", type=str, default="")
    parser.add_argument("--encoder-pretrained-variant", type=str, default="vit-b")
    parser.add_argument("--encoder-pretrained-strict", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device(args.device)
    config = PointOdysseyConfig(
        root=args.data_root,
        split=args.split,
        num_frames=args.num_frames,
        num_queries=args.num_queries,
    )
    config.img_patch_size = int(args.img_patch_size)
    if args.no_aug:
        config.use_data_augmentation = False
    if args.no_random_stride:
        config.temporal_random_stride = False

    dataset = PointOdysseyDataset(config)
    if len(dataset) == 0:
        raise RuntimeError("PointOdyssey dataset is empty.")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        collate_fn=d4rt_collate_fn,
    )

    batch = next(iter(loader))

    # Meta sanity
    aspect = batch["meta"]["aspect_ratio"]
    if not torch.is_tensor(aspect) or aspect.dim() != 2:
        raise ValueError(f"aspect_ratio must be (B,1) tensor, got {type(aspect)} {getattr(aspect, 'shape', None)}")
    if not isinstance(batch["meta"]["img_patch_size"], int):
        raise ValueError(f"img_patch_size must be int, got {type(batch['meta']['img_patch_size'])}")
    if not isinstance(batch["meta"]["align_corners"], bool):
        raise ValueError(f"align_corners must be bool, got {type(batch['meta']['align_corners'])}")

    batch = move_to_device(batch, device)

    model = D4RT(
        img_size=args.img_size,
        patch_size=args.patch_size,
        encoder_in_chans=3,
        encoder_embed_dim=args.encoder_embed_dim,
        encoder_depth=args.encoder_depth,
        encoder_num_heads=args.encoder_num_heads,
        decoder_embed_dim=args.decoder_embed_dim,
        decoder_depth=args.decoder_depth,
        decoder_num_heads=args.decoder_num_heads,
        all_frames=args.num_frames,
        encoder_pretrained=args.encoder_pretrained,
        encoder_pretrained_path=args.encoder_pretrained_path,
        encoder_pretrained_variant=args.encoder_pretrained_variant,
        encoder_pretrained_strict=args.encoder_pretrained_strict,
        encoder_pretrained_verbose=True,
    ).to(device)
    model.eval()

    loss_head = LossHead()
    loss_fn = CompositeLoss()

    with torch.no_grad():
        preds = model(batch["meta"], batch["images"], batch["query"])
        losses, conf = loss_head(preds, batch["targets"])
        loss = loss_fn(losses, conf, batch["targets"].get("query_mask"))

    # Print summary
    print("images:", tuple(batch["images"].shape))
    print("query:", tuple(batch["query"].shape))
    print("preds:", tuple(preds.shape))
    print("loss:", float(loss.item()))
    for k, v in losses.items():
        print(f"{k}: mean={float(v.mean().item()):.6f}, max={float(v.max().item()):.6f}")

    if torch.isnan(loss).any():
        raise RuntimeError("Loss contains NaN.")
    if torch.isnan(preds).any():
        raise RuntimeError("Predictions contain NaN.")


if __name__ == "__main__":
    main()
