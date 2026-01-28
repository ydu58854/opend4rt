import argparse
from pathlib import Path

import torch

from loss_head import LossHead
from model import D4RT
from train import CompositeLoss, build_optimizer, build_scheduler, train_step


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
    parser = argparse.ArgumentParser(description="Train D4RT on easydataset")
    parser.add_argument("--data-dir", type=str, default="/inspire/hdd/project/wuliqifa/public/dyh/d4rt/easydataset")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--img-patch-size", type=int, default=9)
    args = parser.parse_args()

    device = torch.device(args.device)
    model = D4RT(img_size=256, patch_size=16, all_frames=48,encoder_depth = 12).to(device)

    optimizer = build_optimizer(model.parameters(), lr=args.lr)
    scheduler = build_scheduler(optimizer, warmup_steps=10, total_steps=max(args.steps, 11))
    loss_fn = CompositeLoss()
    loss_head = LossHead()

    data_dir = Path(args.data_dir)
    files = sorted(data_dir.glob("sample_*.pt"))
    if not files:
        raise FileNotFoundError(f"No sample_*.pt found in {data_dir}")

    for step in range(args.steps):
        sample = torch.load(files[step % len(files)], map_location="cpu")
        sample["meta"]["img_patch_size"] = int(args.img_patch_size)
        batch = move_to_device(sample, device)
        loss = train_step(model, batch, loss_fn, loss_head, optimizer, scheduler)
        print(f"step {step:04d} loss {loss.item():.6f}")


if __name__ == "__main__":
    main()
