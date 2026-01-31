import math
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import hydra
import torch
import torch.distributed as dist
from omegaconf import DictConfig, OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model import D4RT, LossHead  # type: ignore
from train import CompositeLoss, build_optimizer, build_scheduler, train_step  # type: ignore
from datasets import PointOdysseyConfig, PointOdysseyDataset, d4rt_collate_fn  # type: ignore


# -------------------------
# DDP helpers
# -------------------------
def ddp_is_enabled() -> bool:
    return "RANK" in os.environ and "WORLD_SIZE" in os.environ


def ddp_setup() -> Tuple[int, int, int, bool]:
    """
    Returns: (rank, world_size, local_rank, is_ddp)
    """
    if not ddp_is_enabled():
        return 0, 1, 0, False

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    dist.init_process_group(backend="nccl", init_method="env://")
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank, True


def ddp_cleanup():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank: int) -> bool:
    return rank == 0


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
    # 注意参数名是 log_dir
    return SummaryWriter(log_dir=str(logdir))


# -------------------------
# Distributed evaluation (方案 B): all_reduce sums
# -------------------------
@torch.no_grad()
def evaluate_distributed(
    model,
    loader,
    loss_fn,
    loss_head,
    device,
    max_batches: int,
    is_ddp: bool,
) -> Tuple[float, Dict[str, float]]:
    """
    Runs evaluation on each rank's shard (via DistributedSampler),
    then all-reduces sums so every rank obtains the same global averages.
    """
    model.eval()

    # local accumulators (weighted by valid query count)
    total_loss_sum = 0.0
    total_count = 0.0

    # your loss keys
    loss_keys = ["L3D", "L2D", "Lvis", "Ldisp", "Lconf", "Lnormal"]
    loss_sums = {k: 0.0 for k in loss_keys}

    for batch_idx, batch in enumerate(loader):
        if max_batches > 0 and batch_idx >= max_batches:
            break
        batch = move_to_device(batch, device)
        preds = model(batch["meta"], batch["images"], batch["query"])
        losses, conf = loss_head(preds, batch["targets"])
        query_mask = batch["targets"].get("query_mask")
        loss = loss_fn(losses, conf, query_mask)

        if query_mask is not None:
            mask = query_mask.unsqueeze(-1).to(device=device, dtype=losses["L3D"].dtype)
            denom = float(mask.sum().item())
            if denom <= 0:
                continue
            total_loss_sum += float(loss.item()) * denom
            total_count += denom
            for k in loss_keys:
                loss_sums[k] += float((losses[k] * mask).sum().item())
        else:
            total_loss_sum += float(loss.item())
            total_count += 1.0
            for k in loss_keys:
                loss_sums[k] += float(losses[k].mean().item())

    # pack to tensor for all_reduce
    # [loss_sum, count, L3D_sum, L2D_sum, ...]
    vec = torch.tensor(
        [total_loss_sum, float(total_count)] + [loss_sums[k] for k in loss_keys],
        device=device,
        dtype=torch.float64,
    )

    if is_ddp:
        dist.all_reduce(vec, op=dist.ReduceOp.SUM)

    # compute global averages (same on every rank now)
    global_loss_sum = vec[0].item()
    global_count = float(vec[1].item())

    if global_count == 0:
        avg_loss = 0.0
        avg_losses = {k: 0.0 for k in loss_keys}
        return avg_loss, avg_losses

    avg_loss = global_loss_sum / global_count
    avg_losses = {}
    for i, k in enumerate(loss_keys):
        avg_losses[k] = vec[2 + i].item() / global_count

    return avg_loss, avg_losses


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
    # DDP: model.module
    state_dict = model.module.state_dict() if hasattr(model, "module") else model.state_dict()

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
    # ---- DDP init ----
    rank, world_size, local_rank, is_ddp = ddp_setup()

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

    # device (each rank -> its GPU)
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")

    if is_main_process(rank):
        print(f"[DDP] enabled={is_ddp} rank={rank}/{world_size} local_rank={local_rank} device={device}")
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

    if is_ddp:
        # QueryEmbedding uses a single img_patch_size per batch, leaving other patch MLPs unused.
        # Enable unused-parameter detection to avoid DDP reduction errors.
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

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
    config.img_patch_size = int(cfg.img_patch_size)
    if cfg.dataset.no_aug:
        config.use_data_augmentation = False
    if cfg.dataset.no_random_stride:
        config.temporal_random_stride = False

    dataset = PointOdysseyDataset(config)

    # Filter out scenes with NaN traj3D (train/character*, and also gso* if you want)
    # 注意：只要你的 dataset.len / __getitem__ 依赖 dataset.scenes，这样做是有效的。
    before_count = len(dataset)
    if hasattr(dataset, "scenes"):
        dataset.scenes = [
            s for s in dataset.scenes
            if (not s["scene_id"].startswith("character")) and (not s["scene_id"].startswith("gso"))
        ]
    after_count = len(dataset)
    removed = before_count - after_count
    if removed > 0 and is_main_process(rank):
        print(f"[Dataset] filtered {removed} scenes (remaining {after_count}).")

    # ---- split train/val (must be identical on every rank) ----
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

    # ---- samplers ----
    train_sampler = None
    val_sampler = None
    if is_ddp:
        train_sampler = DistributedSampler(
            train_dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True
        )
        if val_dataset is not None:
            val_sampler = DistributedSampler(
                val_dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False
            )

    # ---- loaders ----
    loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,              # per-GPU
        shuffle=(train_sampler is None),
        sampler=train_sampler,
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
            sampler=val_sampler,
            num_workers=cfg.num_workers,
            collate_fn=d4rt_collate_fn,
            pin_memory=True,
            persistent_workers=(cfg.num_workers > 0),
            prefetch_factor=2 if cfg.num_workers > 0 else None,
            drop_last=False,
        )

    # ---- logging (rank0 only) ----
    wandb = setup_wandb(cfg, config) if is_main_process(rank) else None
    writer = setup_tensorboard(cfg) if is_main_process(rank) else None

    # ---- train loop ----
    step = 0
    epoch = 0
    best_loss = math.inf

    try:
        while step < cfg.steps:
            if is_ddp and train_sampler is not None:
                train_sampler.set_epoch(epoch)

            for batch in loader:
                if step >= cfg.steps:
                    break

                # reset mem stats for this device (local)
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)

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

                # ---- print (rank0 only) ----
                if is_main_process(rank) and step % 1 == 0:
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
                    gpu_mem = torch.cuda.memory_allocated(device) / (1024**2)
                    gpu_mem_max = torch.cuda.max_memory_allocated(device) / (1024**2)

                # ---- wandb/tb (rank0 only) ----
                if is_main_process(rank):
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
                            "global_batch": cfg.batch_size * world_size,
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

                # ---- checkpoint (rank0 only) ----
                if is_main_process(rank):
                    if cfg.save_every_steps > 0 and step % cfg.save_every_steps == 0 and step > 0:
                        save_checkpoint(checkpoint_dir, f"step_{step:06d}", model, optimizer, scheduler, step, epoch, best_loss)
                        save_checkpoint(checkpoint_dir, "last", model, optimizer, scheduler, step, epoch, best_loss)

                    # best
                    if loss.item() < best_loss:
                        best_loss = loss.item()
                        save_checkpoint(checkpoint_dir, "best", model, optimizer, scheduler, step, epoch, best_loss)

                # ---- distributed eval (方案 B) ----
                # 所有 rank 都跑各自 shard，然后 all_reduce 得到全局平均；rank0 负责记录
                if val_loader is not None and cfg.dataset.val_every_steps > 0 and step % cfg.dataset.val_every_steps == 0 and step > 0:
                    # val sampler 不 shuffle，所以不必 set_epoch；若你想严格同步，也可以 set_epoch(epoch)
                    val_loss, val_losses = evaluate_distributed(
                        model,
                        val_loader,
                        loss_fn,
                        loss_head,
                        device,
                        cfg.dataset.val_max_batches,
                        is_ddp=is_ddp,
                    )

                    if is_main_process(rank):
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

            epoch += 1

            # epoch checkpoint (rank0 only)
            if is_main_process(rank) and cfg.save_every_epochs > 0 and epoch % cfg.save_every_epochs == 0:
                save_checkpoint(checkpoint_dir, f"epoch_{epoch:04d}", model, optimizer, scheduler, step, epoch, best_loss)
                save_checkpoint(checkpoint_dir, "last", model, optimizer, scheduler, step, epoch, best_loss)

    finally:
        # cleanup
        if is_main_process(rank):
            if wandb is not None:
                wandb.finish()
            if writer is not None:
                writer.flush()
                writer.close()
        ddp_cleanup()


if __name__ == "__main__":
    main()
