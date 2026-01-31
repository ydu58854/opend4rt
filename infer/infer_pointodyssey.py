import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from datasets.utils import load_image_tensor
from model import D4RT  # type: ignore


def load_frames(scene_dir: Path, target_resolution, normalize: bool = True):
    rgb_dir = scene_dir / "rgbs"
    if not rgb_dir.exists():
        raise FileNotFoundError(f"rgbs not found: {rgb_dir}")
    files = sorted(rgb_dir.glob("rgb_*.jpg"))
    if not files:
        raise FileNotFoundError(f"No rgb_*.jpg found in {rgb_dir}")

    frames = []
    for fp in files:
        img = load_image_tensor(fp, target_size=target_resolution, normalize=normalize)
        frames.append(img)
    stacked = torch.stack(frames, dim=0)  # (T,C,H,W)
    images = stacked.permute(1, 0, 2, 3).contiguous()  # (C,T,H,W)

    # original size from first image
    from PIL import Image
    with Image.open(files[0]) as im:
        orig_W, orig_H = im.size

    return images, (orig_W, orig_H)


def ensure_run_dir(base_dir: Path, run_name: str) -> Path:
    run_name = run_name.strip()
    if not run_name:
        raise ValueError("run name is required and cannot be empty")
    if Path(run_name).name != run_name:
        raise ValueError("run name must be a simple name without path separators")
    out_dir = base_dir / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def save_json(path: Path, payload) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def build_meta(images, orig_size, img_patch_size=9, align_corners=True):
    return {
        "aspect_ratio": torch.tensor([[float(orig_size[0]) / float(orig_size[1])]], dtype=torch.float32),
        "img_patch_size": torch.tensor([img_patch_size], dtype=torch.int64),
        "align_corners": torch.tensor([align_corners], dtype=torch.bool),
    }


def move_meta_to_device(meta, device):
    out = {}
    for k, v in meta.items():
        if torch.is_tensor(v):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


def build_uv_grid(H, W):
    xs = torch.linspace(0, W - 1, W)
    ys = torch.linspace(0, H - 1, H)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    u = (xx / (W - 1)).reshape(-1, 1)
    v = (yy / (H - 1)).reshape(-1, 1)
    return torch.cat([u, v], dim=-1)


def encode_once(model, meta, images):
    return model.encoder(meta, images)


def decode_queries(model, meta, images, global_features, queries, batch_size=4096):
    outputs = []
    for i in range(0, queries.shape[1], batch_size):
        q = queries[:, i : i + batch_size]
        q_embed = model.query_embed(meta, q, images)
        pred = model.decoder(q_embed, global_features)
        outputs.append(pred)
    return torch.cat(outputs, dim=1)  # (B, N, 13)


def predict_track(model, meta, images, global_features, u, v, t_src, batch_size=4096):
    _, T, H, W = images.shape
    t = torch.arange(T, dtype=torch.float32).view(-1, 1)
    uv = torch.tensor([[u, v]], dtype=torch.float32).repeat(T, 1)
    t_src_vec = torch.full((T, 1), float(t_src))
    q = torch.cat([uv, t_src_vec, t, t], dim=-1).unsqueeze(0)
    preds = decode_queries(model, meta, images, global_features, q, batch_size=batch_size)
    return preds[0]


def predict_depth_maps(model, meta, images, global_features, batch_size=4096, output_hw=None):
    _, T, H_img, W_img = images.shape
    if output_hw is None:
        H, W = H_img, W_img
    else:
        H, W = output_hw
    uv = build_uv_grid(H, W)  # (HW,2)
    depth_maps = []
    for t in range(T):
        t_src = torch.full((uv.shape[0], 1), float(t))
        t_tgt = torch.full((uv.shape[0], 1), float(t))
        t_cam = torch.full((uv.shape[0], 1), float(t))
        q = torch.cat([uv, t_src, t_tgt, t_cam], dim=-1).unsqueeze(0)
        preds = decode_queries(model, meta, images, global_features, q, batch_size=batch_size)
        z = preds[0, :, 2].reshape(H, W)
        depth_maps.append(z)
    return torch.stack(depth_maps, dim=0)


def predict_pointcloud(model, meta, images, global_features, t_cam_ref=0, batch_size=4096, output_hw=None):
    _, T, H_img, W_img = images.shape
    if output_hw is None:
        H, W = H_img, W_img
    else:
        H, W = output_hw
    uv = build_uv_grid(H, W)  # (HW,2)
    pointclouds = []
    for t in range(T):
        t_src = torch.full((uv.shape[0], 1), float(t))
        t_tgt = torch.full((uv.shape[0], 1), float(t))
        t_cam = torch.full((uv.shape[0], 1), float(t_cam_ref))
        q = torch.cat([uv, t_src, t_tgt, t_cam], dim=-1).unsqueeze(0)
        preds = decode_queries(model, meta, images, global_features, q, batch_size=batch_size)
        xyz = preds[0, :, 0:3].reshape(H, W, 3)
        pointclouds.append(xyz)
    return torch.stack(pointclouds, dim=0)


def umeyama_alignment(src, tgt):
    # src, tgt: (N,3)
    src_mean = src.mean(axis=0)
    tgt_mean = tgt.mean(axis=0)
    src_centered = src - src_mean
    tgt_centered = tgt - tgt_mean
    cov = tgt_centered.T @ src_centered / src.shape[0]
    U, S, Vt = np.linalg.svd(cov)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    t = tgt_mean - R @ src_mean
    return R, t


def estimate_extrinsics(model, meta, images, global_features, i, j, grid_hw=(32, 32), batch_size=4096):
    _, T, H, W = images.shape
    gh, gw = grid_hw
    xs = torch.linspace(0, W - 1, gw)
    ys = torch.linspace(0, H - 1, gh)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    u = (xx / (W - 1)).reshape(-1, 1)
    v = (yy / (H - 1)).reshape(-1, 1)
    uv = torch.cat([u, v], dim=-1)

    t_src = torch.full((uv.shape[0], 1), float(i))
    t_tgt = torch.full((uv.shape[0], 1), float(i))

    q_i = torch.cat([uv, t_src, t_tgt, torch.full((uv.shape[0], 1), float(i))], dim=-1).unsqueeze(0)
    q_j = torch.cat([uv, t_src, t_tgt, torch.full((uv.shape[0], 1), float(j))], dim=-1).unsqueeze(0)

    p_i = decode_queries(model, meta, images, global_features, q_i, batch_size=batch_size)[0, :, 0:3].cpu().numpy()
    p_j = decode_queries(model, meta, images, global_features, q_j, batch_size=batch_size)[0, :, 0:3].cpu().numpy()

    R, t = umeyama_alignment(p_i, p_j)
    return R, t


def estimate_intrinsics(model, meta, images, global_features, i, grid_hw=(32, 32), batch_size=4096):
    _, T, H, W = images.shape
    gh, gw = grid_hw
    xs = torch.linspace(0, W - 1, gw)
    ys = torch.linspace(0, H - 1, gh)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    u = (xx / (W - 1)).reshape(-1, 1)
    v = (yy / (H - 1)).reshape(-1, 1)
    uv = torch.cat([u, v], dim=-1)

    t_src = torch.full((uv.shape[0], 1), float(i))
    t_tgt = torch.full((uv.shape[0], 1), float(i))
    t_cam = torch.full((uv.shape[0], 1), float(i))
    q = torch.cat([uv, t_src, t_tgt, t_cam], dim=-1).unsqueeze(0)

    p = decode_queries(model, meta, images, global_features, q, batch_size=batch_size)[0, :, 0:3].cpu().numpy()
    u_np = uv[:, 0].cpu().numpy()
    v_np = uv[:, 1].cpu().numpy()
    px = p[:, 0]
    py = p[:, 1]
    pz = p[:, 2]
    fx = pz * (u_np - 0.5) / (px + 1e-8)
    fy = pz * (v_np - 0.5) / (py + 1e-8)
    return float(np.median(fx)), float(np.median(fy))


def predict_dense_tracks(
    model,
    meta,
    images,
    global_features,
    vis_threshold=0.5,
    batch_points=256,
    query_batch_size=4096,
    output_hw=None,
):
    _, T, H_img, W_img = images.shape
    if output_hw is None:
        H, W = H_img, W_img
    else:
        H, W = output_hw
    visited = torch.zeros((T, H, W), dtype=torch.bool, device=images.device)
    tracks = []

    while not visited.all():
        unvisited = (~visited).nonzero(as_tuple=False)
        if unvisited.numel() == 0:
            break
        if unvisited.shape[0] > batch_points:
            idx = torch.randperm(unvisited.shape[0], device=images.device)[:batch_points]
            batch = unvisited[idx]
        else:
            batch = unvisited
        # batch: (B, 3) -> t, y, x
        t_src = batch[:, 0].float()
        y = batch[:, 1].float()
        x = batch[:, 2].float()
        u = x / (W - 1)
        v = y / (H - 1)

        # Queries for full track across all frames: (B*T, 5)
        q_list = []
        for k in range(T):
            t_k = torch.full_like(t_src, float(k))
            q = torch.stack([u, v, t_src, t_k, t_k], dim=-1)
            q_list.append(q)
        q_all = torch.cat(q_list, dim=0).unsqueeze(0)

        preds = decode_queries(model, meta, images, global_features, q_all, batch_size=query_batch_size)[0]
        preds = preds.reshape(T, -1, preds.shape[-1])  # (T, B, 13)
        pred_uv = preds[..., 3:5].clamp(0.0, 1.0)
        pred_vis = torch.sigmoid(preds[..., 5])

        for b in range(pred_uv.shape[1]):
            track_uv = pred_uv[:, b, :].detach().cpu().numpy()
            track_vis = pred_vis[:, b].detach().cpu().numpy()
            tracks.append({"uv": track_uv, "vis": track_vis})

        # mark visited
        visible = pred_vis > vis_threshold
        px = (pred_uv[..., 0] * (W - 1)).round().long().clamp(0, W - 1)
        py = (pred_uv[..., 1] * (H - 1)).round().long().clamp(0, H - 1)
        for k in range(T):
            mask = visible[k]
            if mask.any():
                visited[k, py[k, mask], px[k, mask]] = True

    return tracks


def main():
    parser = argparse.ArgumentParser(description="Inference for PointOdyssey")
    parser.add_argument("--scene-dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--target-resolution", type=int, nargs=2, default=(256, 256))
    parser.add_argument("--img-patch-size", type=int, default=9)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument(
        "--mode",
        type=str,
        default="depth",
        choices=["depth", "track", "pointcloud", "intrinsics", "extrinsics", "dense_tracks"],
    )
    parser.add_argument("--t-src", type=int, default=0)
    parser.add_argument("--u", type=float, default=0.5)
    parser.add_argument("--v", type=float, default=0.5)
    parser.add_argument("--t-cam-ref", type=int, default=0)
    parser.add_argument("--i", type=int, default=0)
    parser.add_argument("--j", type=int, default=1)
    parser.add_argument("--vis-threshold", type=float, default=0.5)
    parser.add_argument("--track-batch-points", type=int, default=256)
    parser.add_argument("--output-resolution", type=str, default="target", choices=["target", "orig"])
    args = parser.parse_args()

    base_output_dir = Path(__file__).resolve().parent
    output_dir = ensure_run_dir(base_output_dir, args.run_name)

    device = torch.device(args.device)
    model = D4RT(img_size=256, patch_size=16, all_frames=48, encoder_depth=12)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state, strict=False)
    model.to(device).eval()

    images, orig_size = load_frames(Path(args.scene_dir), args.target_resolution, normalize=True)
    images = images.unsqueeze(0).to(device)
    meta = build_meta(images, orig_size, img_patch_size=args.img_patch_size, align_corners=True)
    meta = move_meta_to_device(meta, device)

    global_features = encode_once(model, meta, images)
    if args.output_resolution == "orig":
        output_hw = (orig_size[1], orig_size[0])
    else:
        output_hw = (args.target_resolution[1], args.target_resolution[0])

    run_info = {
        "run_name": args.run_name,
        "scene_dir": str(Path(args.scene_dir).resolve()),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "mode": args.mode,
        "device": args.device,
        "target_resolution": list(args.target_resolution),
        "output_resolution": args.output_resolution,
        "output_hw": list(output_hw),
        "img_patch_size": args.img_patch_size,
        "batch_size": args.batch_size,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    save_json(output_dir / "run.json", run_info)

    if args.mode == "track":
        preds = predict_track(model, meta, images, global_features, args.u, args.v, args.t_src, args.batch_size)
        out_path = output_dir / "track.npy"
        np.save(out_path, preds.detach().cpu().numpy())
        print(f"track preds {preds.shape} -> {out_path}")
    elif args.mode == "depth":
        depth = predict_depth_maps(model, meta, images, global_features, args.batch_size, output_hw=output_hw)
        out_path = output_dir / "depth.npy"
        np.save(out_path, depth.detach().cpu().numpy())
        print(f"depth maps {depth.shape} -> {out_path}")
    elif args.mode == "pointcloud":
        pc = predict_pointcloud(model, meta, images, global_features, args.t_cam_ref, args.batch_size, output_hw=output_hw)
        out_path = output_dir / "pointcloud.npy"
        np.save(out_path, pc.detach().cpu().numpy())
        print(f"pointcloud {pc.shape} -> {out_path}")
    elif args.mode == "intrinsics":
        fx, fy = estimate_intrinsics(model, meta, images, global_features, args.i)
        out_path = output_dir / "intrinsics.json"
        save_json(out_path, {"fx": fx, "fy": fy})
        print(f"fx={fx:.4f} fy={fy:.4f} -> {out_path}")
    elif args.mode == "extrinsics":
        R, t = estimate_extrinsics(model, meta, images, global_features, args.i, args.j)
        out_path = output_dir / "extrinsics.npz"
        np.savez(out_path, R=R, t=t)
        print(f"R {R.shape} t {t.shape} -> {out_path}")
    elif args.mode == "dense_tracks":
        tracks = predict_dense_tracks(
            model,
            meta,
            images,
            global_features,
            vis_threshold=args.vis_threshold,
            batch_points=args.track_batch_points,
            query_batch_size=args.batch_size,
            output_hw=output_hw,
        )
        if tracks:
            uv = np.stack([t["uv"] for t in tracks], axis=0)
            vis = np.stack([t["vis"] for t in tracks], axis=0)
        else:
            t_len = images.shape[2] if images.ndim == 5 else images.shape[1]
            uv = np.empty((0, t_len, 2), dtype=np.float32)
            vis = np.empty((0, t_len), dtype=np.float32)
        out_path = output_dir / "dense_tracks.npz"
        np.savez(out_path, uv=uv, vis=vis)
        print(f"tracks: {len(tracks)} -> {out_path}")


if __name__ == "__main__":
    main()
