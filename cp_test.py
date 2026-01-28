import torch
from encoder import D4RTEncoder

ckpt = "/Users/kynyty04/Desktop/opend4rt/checkpoint/pytorch_model.bin"

encoder = D4RTEncoder(
    img_size=256, patch_size=16, embed_dim=768,
    depth=12, num_heads=12, tubelet_size=2, all_frames=48
)

load_result, skipped = encoder.load_videomae_vitb_encoder(ckpt, strict=False)
print("missing:", load_result.missing_keys)
print("unexpected:", load_result.unexpected_keys)
print("skipped:", skipped)

# forward sanity
B, C, T, H, W = 1, 3, 48, 256, 256
images = torch.randn(B, C, T, H, W)
meta = {"aspect_ratio": torch.ones(B, 1)}
with torch.no_grad():
    out = encoder(meta, images)
print("out shape:", out.shape, "mean/std:", out.mean().item(), out.std().item())
