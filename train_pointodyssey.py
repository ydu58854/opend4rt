import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from loss_head import LossHead  # type: ignore
    from model import D4RT  # type: ignore
    from train import CompositeLoss, build_optimizer, build_scheduler, train_step  # type: ignore
    from datasets import PointOdysseyConfig, PointOdysseyDataset, d4rt_collate_fn  # type: ignore
else:
    from .loss_head import LossHead
    from .model import D4RT
    from .train import CompositeLoss, build_optimizer, build_scheduler, train_step
    from .datasets import PointOdysseyConfig, PointOdysseyDataset, d4rt_collate_fn


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


def main():
    parser = argparse.ArgumentParser(description="Train D4RT on PointOdyssey")
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--steps", type=int, default=100000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--img-patch-size", type=int, default=9)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--no-aug", action="store_true")
    parser.add_argument("--no-random-stride", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device)
    model = D4RT(img_size=256, patch_size=16, all_frames=48, encoder_depth=12).to(device)

    optimizer = build_optimizer(model.parameters(), lr=args.lr)
    scheduler = build_scheduler(optimizer, warmup_steps=2500, total_steps=max(args.steps, 2501))
    loss_fn = CompositeLoss()
    loss_head = LossHead()

    config = PointOdysseyConfig(
        root=args.data_root,
        split=args.split,
        num_frames=48,
        num_queries=2048,
    )
    if args.no_aug:
        config.use_data_augmentation = False
    if args.no_random_stride:
        config.temporal_random_stride = False

    dataset = PointOdysseyDataset(config)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=d4rt_collate_fn,
        pin_memory=True,
    )

    step = 0
    while step < args.steps:
        for batch in loader:
            batch["meta"]["img_patch_size"] = int(args.img_patch_size)
            batch = move_to_device(batch, device)
            loss = train_step(model, batch, loss_fn, loss_head, optimizer, scheduler)
            if step % 10 == 0:
                print(f"step {step:06d} loss {loss.item():.6f}")
            step += 1
            if step >= args.steps:
                break


if __name__ == "__main__":
    main()
