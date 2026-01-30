import math
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import hydra
from omegaconf import DictConfig, OmegaConf

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
    logdir = Path(cfg.tensorboard.logdir)
    logdir.mkdir(parents=True, exist_ok=True)
    return SummaryWriter(logdir=str(logdir))


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
    payload = {
        "model": model.state_dict(),
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


@hydra.main(config_path="configs", config_name="train_pointodyssey", version_base="1.3")
def main(cfg: DictConfig):
    data_root = _resolve_path(cfg.data_root)
    checkpoint_dir = _resolve_path(cfg.checkpoint_dir)
    tensorboard_logdir = _resolve_path(cfg.tensorboard.logdir)
    cfg.tensorboard.logdir = str(tensorboard_logdir)

    device = torch.device(cfg.device)
    model = D4RT(
        img_size=cfg.model.img_size,
        patch_size=cfg.model.patch_size,
        all_frames=cfg.model.all_frames,
        encoder_depth=cfg.model.encoder_depth,
        encoder_pretrained=cfg.model.encoder_pretrained,
    ).to(device)

    optimizer = build_optimizer(model.parameters(), lr=cfg.lr)
    scheduler = build_scheduler(optimizer, warmup_steps=2500, total_steps=max(cfg.steps, 2501))
    loss_fn = CompositeLoss()
    loss_head = LossHead()

    config = PointOdysseyConfig(
        root=str(data_root),
        split=cfg.split,
        num_frames=cfg.dataset.num_frames,
        num_queries=cfg.dataset.num_queries,
    )
    if cfg.dataset.no_aug:
        config.use_data_augmentation = False
    if cfg.dataset.no_random_stride:
        config.temporal_random_stride = False

    dataset = PointOdysseyDataset(config)
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=d4rt_collate_fn,
        pin_memory=True,
    )

    wandb = setup_wandb(cfg, config)
    writer = setup_tensorboard(cfg)

    step = 0
    epoch = 0
    best_loss = math.inf
    while step < cfg.steps:
        for batch in loader:
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            batch["meta"]["img_patch_size"] = int(cfg.img_patch_size)
            batch = move_to_device(batch, device)
            loss, grad_norm = train_step(
                model,
                batch,
                loss_fn,
                loss_head,
                optimizer,
                scheduler,
                max_grad_norm=cfg.max_grad_norm,
                return_grad_norm=True,
            )
            if step % 10 == 0:
                print(f"step {step:06d} loss {loss.item():.6f}")
            gpu_mem = None
            gpu_mem_max = None
            if device.type == "cuda":
                gpu_mem = torch.cuda.memory_allocated(device) / (1024 ** 2)
                gpu_mem_max = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
            if wandb is not None and step % cfg.wandb.log_every == 0:
                log_payload = {
                    "loss": loss.item(),
                    "lr": optimizer.param_groups[0]["lr"],
                    "best_loss": best_loss,
                    "epoch": epoch,
                    "grad_norm": float(grad_norm.item()),
                }
                if gpu_mem is not None:
                    log_payload["gpu_mem_mb"] = gpu_mem
                    log_payload["gpu_mem_max_mb"] = gpu_mem_max
                wandb.log(log_payload, step=step)
            if writer is not None and step % cfg.wandb.log_every == 0:
                writer.add_scalar("loss", loss.item(), step)
                writer.add_scalar("lr", optimizer.param_groups[0]["lr"], step)
                writer.add_scalar("best_loss", best_loss, step)
                writer.add_scalar("grad_norm", float(grad_norm.item()), step)
                writer.add_scalar("epoch", epoch, step)
                if gpu_mem is not None:
                    writer.add_scalar("gpu/mem_mb", gpu_mem, step)
                    writer.add_scalar("gpu/mem_max_mb", gpu_mem_max, step)
            if cfg.save_every_steps > 0 and step % cfg.save_every_steps == 0 and step > 0:
                save_checkpoint(
                    checkpoint_dir,
                    f"step_{step:06d}",
                    model,
                    optimizer,
                    scheduler,
                    step,
                    epoch,
                    best_loss,
                )
                save_checkpoint(
                    checkpoint_dir,
                    "last",
                    model,
                    optimizer,
                    scheduler,
                    step,
                    epoch,
                    best_loss,
                )
            if loss.item() < best_loss:
                best_loss = loss.item()
                save_checkpoint(
                    checkpoint_dir,
                    "best",
                    model,
                    optimizer,
                    scheduler,
                    step,
                    epoch,
                    best_loss,
                )
            step += 1
            if step >= cfg.steps:
                break
        epoch += 1
        if cfg.save_every_epochs > 0 and epoch % cfg.save_every_epochs == 0:
            save_checkpoint(
                checkpoint_dir,
                f"epoch_{epoch:04d}",
                model,
                optimizer,
                scheduler,
                step,
                epoch,
                best_loss,
            )
            save_checkpoint(
                checkpoint_dir,
                "last",
                model,
                optimizer,
                scheduler,
                step,
                epoch,
                best_loss,
            )

    if wandb is not None:
        wandb.finish()
    if writer is not None:
        writer.flush()
        writer.close()


if __name__ == "__main__":
    main()
