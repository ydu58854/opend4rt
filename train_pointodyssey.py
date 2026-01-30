import math
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split
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
    try:
        from torch.utils.tensorboard import SummaryWriter  # type: ignore
    except Exception as exc:
        print(f"[tensorboard] disabled (import failed): {exc}")
        return None
    logdir = Path(cfg.tensorboard.logdir)
    logdir.mkdir(parents=True, exist_ok=True)
    return SummaryWriter(logdir=str(logdir))


def evaluate(
    model,
    loader,
    loss_fn,
    loss_head,
    device,
    max_batches: int,
):
    model.eval()
    total_loss = 0.0
    total_count = 0
    loss_sums = {"L3D": 0.0, "L2D": 0.0, "Lvis": 0.0, "Ldisp": 0.0, "Lconf": 0.0, "Lnormal": 0.0}
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if max_batches > 0 and batch_idx >= max_batches:
                break
            batch = move_to_device(batch, device)
            preds = model(batch["meta"], batch["images"], batch["query"])
            losses, conf = loss_head(preds, batch["targets"])
            loss = loss_fn(losses, conf, batch["targets"].get("query_mask"))
            total_loss += float(loss.item())
            total_count += 1
            for key in loss_sums:
                loss_sums[key] += float(losses[key].mean().item())
    if total_count == 0:
        return 0.0, {k: 0.0 for k in loss_sums}
    avg_loss = total_loss / total_count
    avg_losses = {k: v / total_count for k, v in loss_sums.items()}
    return avg_loss, avg_losses


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
        encoder_in_chans=cfg.model.encoder_in_chans,
        encoder_embed_dim=cfg.model.encoder_embed_dim,
        encoder_depth=cfg.model.encoder_depth,
        encoder_num_heads=cfg.model.encoder_num_heads,
        decoder_embed_dim=cfg.model.decoder_embed_dim,
        decoder_depth=cfg.model.decoder_depth,
        decoder_num_heads=cfg.model.decoder_num_heads,
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
    val_loader = None
    if cfg.dataset.val_fraction > 0:
        total_len = len(dataset)
        val_len = max(1, int(total_len * cfg.dataset.val_fraction))
        train_len = max(1, total_len - val_len)
        if train_len + val_len > total_len:
            val_len = total_len - train_len
        generator = torch.Generator().manual_seed(cfg.dataset.val_seed)
        train_dataset, val_dataset = random_split(dataset, [train_len, val_len], generator=generator)
    else:
        train_dataset = dataset
        val_dataset = None

    loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=d4rt_collate_fn,
        pin_memory=True,
    )
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
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
            if step % 10 == 0:
                print(
                    "step {step:06d} loss {loss:.6f} "
                    "L3D {L3D:.4f} L2D {L2D:.4f} Lvis {Lvis:.4f} "
                    "Ldisp {Ldisp:.4f} Lconf {Lconf:.4f} Lnormal {Lnormal:.4f}".format(
                        step=step,
                        loss=loss.item(),
                        L3D=loss_stats["L3D"].item(),
                        L2D=loss_stats["L2D"].item(),
                        Lvis=loss_stats["Lvis"].item(),
                        Ldisp=loss_stats["Ldisp"].item(),
                        Lconf=loss_stats["Lconf"].item(),
                        Lnormal=loss_stats["Lnormal"].item(),
                    )
                )
            gpu_mem = None
            gpu_mem_max = None
            if device.type == "cuda":
                gpu_mem = torch.cuda.memory_allocated(device) / (1024 ** 2)
                gpu_mem_max = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
            if wandb is not None and step % cfg.wandb.log_every == 0:
                log_payload = {
                    "loss": loss.item(),
                    "loss/L3D": loss_stats["L3D"].item(),
                    "loss/L2D": loss_stats["L2D"].item(),
                    "loss/Lvis": loss_stats["Lvis"].item(),
                    "loss/Ldisp": loss_stats["Ldisp"].item(),
                    "loss/Lconf": loss_stats["Lconf"].item(),
                    "loss/Lnormal": loss_stats["Lnormal"].item(),
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
                writer.add_scalar("loss/L3D", loss_stats["L3D"].item(), step)
                writer.add_scalar("loss/L2D", loss_stats["L2D"].item(), step)
                writer.add_scalar("loss/Lvis", loss_stats["Lvis"].item(), step)
                writer.add_scalar("loss/Ldisp", loss_stats["Ldisp"].item(), step)
                writer.add_scalar("loss/Lconf", loss_stats["Lconf"].item(), step)
                writer.add_scalar("loss/Lnormal", loss_stats["Lnormal"].item(), step)
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
            if val_loader is not None and cfg.dataset.val_every_steps > 0 and step % cfg.dataset.val_every_steps == 0 and step > 0:
                val_loss, val_losses = evaluate(
                    model,
                    val_loader,
                    loss_fn,
                    loss_head,
                    device,
                    cfg.dataset.val_max_batches,
                )
                if wandb is not None:
                    wandb.log(
                        {
                            "val/loss": val_loss,
                            "val/L3D": val_losses["L3D"],
                            "val/L2D": val_losses["L2D"],
                            "val/Lvis": val_losses["Lvis"],
                            "val/Ldisp": val_losses["Ldisp"],
                            "val/Lconf": val_losses["Lconf"],
                            "val/Lnormal": val_losses["Lnormal"],
                        },
                        step=step,
                    )
                if writer is not None:
                    writer.add_scalar("val/loss", val_loss, step)
                    writer.add_scalar("val/L3D", val_losses["L3D"], step)
                    writer.add_scalar("val/L2D", val_losses["L2D"], step)
                    writer.add_scalar("val/Lvis", val_losses["Lvis"], step)
                    writer.add_scalar("val/Ldisp", val_losses["Ldisp"], step)
                    writer.add_scalar("val/Lconf", val_losses["Lconf"], step)
                    writer.add_scalar("val/Lnormal", val_losses["Lnormal"], step)
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
