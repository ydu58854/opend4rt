import math
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, random_split
from datasets import PointOdysseyConfig, PointOdysseyDataset, d4rt_collate_fn  # type: ignore

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model import D4RT, LossHead  # type: ignore
from train import CompositeLoss, build_optimizer, build_scheduler, train_step  # type: ignore

# -------------------------
# Device utils
# -------------------------
def move_to_device(batch, device):
    meta = {}
    for key, value in batch["meta"].items():
        if torch.is_tensor(value):
            meta[key] = value.to(device, non_blocking=True)
        else:
            meta[key] = value

    targets = {}
    for key, value in batch["targets"].items():
        targets[key] = value.to(device, non_blocking=True)

    return {
        "meta": meta,
        "images": batch["images"].to(device, non_blocking=True),
        "query": batch["query"].to(device, non_blocking=True),
        "targets": targets,
    }

# -------------------------
# Logging utils
# -------------------------
def setup_wandb(cfg, config):
    if not cfg.wandb.use:
        return None
    try:
        import wandb  # type: ignore
    except Exception as exc:
        print(f"[wandb] disabled (import failed): {exc}")
        return None

    wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        name=cfg.wandb.name,
        group=cfg.wandb.group,
        tags=cfg.wandb.tags.split(",") if cfg.wandb.tags else None,
        mode=cfg.wandb.mode,
        config={
            "train": OmegaConf.to_container(cfg, resolve=True),
            "dataset": config.__dict__,
        },
    )
    return wandb


def setup_tensorboard(cfg):
    if not cfg.tensorboard.use:
        return None
    try:
        from torch.utils.tensorboard import SummaryWriter  # type: ignore
    except Exception as exc:
        print(f"[tensorboard] disabled (import failed): {exc}")
        return None
    logdir = Path(cfg.tensorboard.logdir)
    logdir.mkdir(parents=True, exist_ok=True)
    return SummaryWriter(log_dir=str(logdir))

# -------------------------
# Checkpoint
# -------------------------
def save_checkpoint(
    checkpoint_dir: Path,
    tag: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    step: int,
    epoch: int,
    best_loss: float,
):
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    state_dict = model.state_dict()

    payload = {
        "model": state_dict,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "step": step,
        "epoch": epoch,
        "best_loss": best_loss,
    }
    torch.save(payload, checkpoint_dir / f"{tag}.pt")


def _resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return Path(hydra.utils.get_original_cwd()) / path


@hydra.main(
    config_path="/inspire/hdd/project/wuliqifa/public/dyh/d4rt/opend4rt/configs",
    config_name="train_pointodyssey",
    version_base="1.3",
)
def main(cfg: DictConfig):
    # 获取设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用的设备：{device}")

    # Generate timestamp and create checkpoint/tensorboard subfolder name
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    encoder_variant = cfg.model.encoder_pretrained_variant if hasattr(cfg.model, 'encoder_pretrained_variant') and cfg.model.encoder_pretrained_variant else "vit-unknown"
    run_subfolder = f"{encoder_variant}_{run_timestamp}"

    # paths
    data_root = _resolve_path(cfg.data_root)
    base_checkpoint_dir = _resolve_path(cfg.checkpoint_dir)
    checkpoint_dir = base_checkpoint_dir / run_subfolder

    # Create tensorboard logdir under d4rt/runs/ with the same subfolder structure
    base_runs_dir = Path(hydra.utils.get_original_cwd()) / "runs"
    tensorboard_logdir = base_runs_dir / run_subfolder
    cfg.tensorboard.logdir = str(tensorboard_logdir)

    # 打印配置信息
    print(f"[cfg] data_root={data_root}")
    print(f"[cfg] checkpoint_dir={checkpoint_dir}")
    print(f"[cfg] tensorboard_logdir={tensorboard_logdir}")
    print(f"[cfg] steps={cfg.steps} batch_size(per_gpu)={cfg.batch_size} num_workers={cfg.num_workers}")

    # ---- model ----
    model = D4RT(
        img_size=cfg.model.img_size,
        patch_size=cfg.model.patch_size,
        encoder_in_chans=cfg.model.encoder_in_chans,
        encoder_embed_dim=cfg.model.encoder_embed_dim,
        encoder_depth=cfg.model.encoder_depth,
        encoder_num_heads=cfg.model.encoder_num_heads,
        decoder_embed_dim=cfg.model.decoder_embed_dim,
        decoder_depth=cfg.model.decoder_depth,
        decoder_num_heads=cfg.model.decoder_num_heads,
        mlp_ratio=cfg.model.mlp_ratio,
        all_frames=cfg.model.all_frames,
        encoder_pretrained=cfg.model.encoder_pretrained,
        encoder_pretrained_path=str(_resolve_path(cfg.model.encoder_pretrained_path)),
        encoder_pretrained_variant=cfg.model.encoder_pretrained_variant,
        encoder_pretrained_strict=cfg.model.encoder_pretrained_strict,
        encoder_pretrained_verbose=cfg.model.encoder_pretrained_verbose,
    ).to(device)

    optimizer = build_optimizer(model.parameters(), lr=cfg.lr)
    scheduler = build_scheduler(optimizer, warmup_steps=2500, total_steps=max(cfg.steps, 2501))
    loss_fn = CompositeLoss()
    loss_head = LossHead().to(device)

    # ---- dataset ----
    config = PointOdysseyConfig(
        root=str(data_root),
        split=cfg.split,
        num_frames=cfg.dataset.num_frames,
        num_queries=cfg.dataset.num_queries,
    )
    dataset = PointOdysseyDataset(config)

    # split train/val (must be identical on every rank)
    val_loader = None
    total_len = len(dataset)
    train_len = int(total_len * (1 - cfg.dataset.val_fraction))
    val_len = total_len - train_len
    train_dataset, val_dataset = random_split(dataset, [train_len, val_len])

    # ---- samplers ----
    train_sampler = None  # 单卡不使用分布式采样器

    # ---- loaders ----
    loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,              # per-GPU
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=d4rt_collate_fn,
        pin_memory=True,
        persistent_workers=(cfg.num_workers > 0),
        prefetch_factor=2 if cfg.num_workers > 0 else None,
        drop_last=True,
    )

    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            collate_fn=d4rt_collate_fn,
            pin_memory=True,
            persistent_workers=(cfg.num_workers > 0),
            prefetch_factor=2 if cfg.num_workers > 0 else None,
            drop_last=False,
        )

    # ---- logging (rank0 only) ----
    wandb = setup_wandb(cfg, config)
    writer = setup_tensorboard(cfg)

    # ---- train loop ----
    step = 0
    epoch = 0
    best_loss = math.inf

    try:
        while step < cfg.steps:
            for batch in loader:
                batch = move_to_device(batch, device)

                loss, grad_norm, loss_stats = train_step(
                    model,
                    batch,
                    loss_fn,
                    loss_head,
                    optimizer,
                    scheduler,
                    max_grad_norm=cfg.max_grad_norm,
                    return_grad_norm=True,
                    return_losses=True,
                )

                # 打印损失
                if step % 1 == 0:
                    print(
                        f"step {step:06d} loss {loss:.6f} "
                        f"L3D {loss_stats['L3D']:.4f} L2D {loss_stats['L2D']:.4f}"
                    )

                # ---- wandb/tb logging ----
                if wandb is not None and step % cfg.wandb.log_every == 0:
                    wandb.log({
                        "loss": loss.item(),
                        "loss/L3D": loss_stats["L3D"].item(),
                        "loss/L2D": loss_stats["L2D"].item(),
                    }, step=step)

                if writer is not None and step % cfg.wandb.log_every == 0:
                    writer.add_scalar("loss", loss.item(), step)
                    writer.add_scalar("loss/L3D", loss_stats["L3D"].item(), step)

                # ---- checkpoint (rank0 only) ----
                if step % cfg.save_every_steps == 0 and step > 0:
                    save_checkpoint(checkpoint_dir, f"step_{step:06d}", model, optimizer, scheduler, step, epoch, best_loss)

                # 更新步骤
                step += 1

            epoch += 1

    finally:
        if wandb is not None:
            wandb.finish()
        if writer is not None:
            writer.flush()
            writer.close()

if __name__ == "__main__":
    main()
