from pathlib import Path
import os
import sys
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from datasets.utils import load_image_tensor
def load_frames(scene_dir, target_resolution, normalize: bool = True, max_frames=48):
    rgb_dir = scene_dir / "rgbs"
    if not rgb_dir.exists():
        raise FileNotFoundError(f"rgbs not found: {rgb_dir}")
    files = sorted(rgb_dir.glob("rgb_*.jpg"))
    if not files:
        raise FileNotFoundError(f"No rgb_*.jpg found in {rgb_dir}")

    frames = []
    for fp in files[:max_frames]:  # 只取前 max_frames 帧
        img = load_image_tensor(fp, target_size=target_resolution, normalize=normalize)
        frames.append(img)
    stacked = torch.stack(frames, dim=0)  # (T,C,H,W)
    images = stacked.permute(1, 0, 2, 3).contiguous()  # (C,T,H,W)

    # original size from first image
    from PIL import Image
    with Image.open(files[0]) as im:
        orig_W, orig_H = im.size

    return images, (orig_W, orig_H)

scene_dir = "/inspire/qb-ilm/project/wuliqifa/public/dyh_data/pointodyssey/test/ani2"
target_resolution = (256,256)
max_frames=48
images , ori =load_frames(Path(scene_dir), target_resolution, max_frames)
print(images.shape,ori)