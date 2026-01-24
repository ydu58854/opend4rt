# --------------------------------------------------------
# Based on BEiT, timm, DINO, VideoMAE, VGGT and DeiT code bases
# https://github.com/microsoft/unilm/tree/master/beit
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit
# https://github.com/facebookresearch/dino
# https://github.com/MCG-NJU/VideoMAE
# https://github.com/facebookresearch/vggt
# --------------------------------------------------------'

from functools import partial

import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
import math
import torch.nn.functional as F
from timm.models.layers import trunc_normal_ as __call_trunc_normal_
from timm.models.registry import register_model
from .modules import (
    Block,
    PatchEmbed,
    _cfg,
    get_sinusoid_encoding_table,
)
class QueryEmbedding(nn.Module):
    """
    Query: (B, N, 5) 每个 query = (u, v, t_src, t_tgt, t_cam)
      - u, v in [0, 1]
      - t_src, t_tgt, t_cam in [0, S-1]
      - t_src 表示 (u,v) 所在帧 index

    token1: (u,v) Fourier -> FC
    token2/3/4: t_src/t_tgt/t_cam -> Embedding
    token5: 从 images(B,T,C,H,W) 取出 t_src 帧中以(u,v)定位像素为中心的 3x3 patch
            patch展平 -> MLP -> token5
    最终 query_token = token1 + token2 + token3 + token4 + token5
    """

    def __init__(
        self,
        embed_dim: int = 768,
        num_frames: int = 48 ,          # S
        num_freqs: int = 10,      # Fourier 频带数量
        include_uv: bool = False,
        dropout: float = 0.0,
        image_in_chans: int = 3,          # images 的 C
        patch_mlp_ratio: int = 4,
        out_mlp_ratio : int = 4,
        img_patch_sizes = (3,6,9,12,15),
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_frames = num_frames
        self.num_freqs = num_freqs
        self.include_uv = include_uv
        self.image_in_chans = image_in_chans
        self.img_patch_sizes = tuple(img_patch_sizes)
        self.patch_mlps = nn.ModuleDict()
            
        for k in self.img_patch_sizes:
            in_dim = image_in_chans * (k * k)
            hidden = int(in_dim * patch_mlp_ratio)
            self.patch_mlps[str(k)] = nn.Sequential(
                nn.Linear(in_dim, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, embed_dim),
                nn.Dropout(dropout),
            )        
        
        # Fourier 频带：1,2,4,...
        freq_bands = 2.0 ** torch.arange(num_freqs, dtype=torch.float32)
        self.register_buffer("freq_bands", freq_bands, persistent=False)

        # Fourier 输出维度：sin/cos(u) + sin/cos(v) => 4F，(可选 +2 原uv)
        fourier_dim = 4 * num_freqs + (2 if include_uv else 0)

        # token1: (u,v) -> embed_dim
        self.uv_proj = nn.Sequential(
            nn.Linear(fourier_dim, embed_dim),
            nn.Dropout(dropout),
        )


        # token2/3/4: 时间 embedding
        self.t_src_emb = nn.Embedding(num_frames, embed_dim)
        self.t_tgt_emb = nn.Embedding(num_frames, embed_dim)
        self.t_cam_emb = nn.Embedding(num_frames, embed_dim)

        out_mlp_hidden = embed_dim * out_mlp_ratio
        self.out_mlp = nn.Sequential(
            nn.Linear(embed_dim, out_mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_mlp_hidden, embed_dim),
            nn.Dropout(dropout),
        )


        self.out_drop = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.t_src_emb.weight, std=0.02)
        nn.init.normal_(self.t_tgt_emb.weight, std=0.02)
        nn.init.normal_(self.t_cam_emb.weight, std=0.02)

        for m in self.uv_proj.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        for mlp in self.patch_mlps.values():
            for m in mlp.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)


        if self.out_mlp is not None:
            for m in self.out_mlp.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)       

    def _fourier_uv(self, uv: torch.Tensor) -> torch.Tensor:
        """
        uv: (B, N, 2) in [0,1]
        return: (B, N, 4F [+2])
        """
        u = uv[..., 0:1]  # (B,N,1)
        v = uv[..., 1:2]

        u_proj = (2.0 * math.pi) * u * self.freq_bands.view(1, 1, -1)  # (B,N,F)
        v_proj = (2.0 * math.pi) * v * self.freq_bands.view(1, 1, -1)

        feat = torch.cat(
            [torch.sin(u_proj), torch.cos(u_proj),
             torch.sin(v_proj), torch.cos(v_proj)],
            dim=-1,  # (B,N,4F)
        )
        if self.include_uv:
            feat = torch.cat([uv, feat], dim=-1)  # (B,N,2+4F)
        return feat

    @staticmethod
    def _extract_kxk_patches_grid_sample(
        images: torch.Tensor,   # (B, T, C, H, W)
        t_src: torch.Tensor,    # (B, N) long
        uv: torch.Tensor,       # (B, N, 2) float in [0,1]
        img_patch_size: int = 3,    # k
        mode: str = "bilinear",
        padding_mode: str = "border",
        align_corners: bool = True,
    ) -> torch.Tensor:
        """
        return: patches (B, N, C, k, k), 可导到 uv
        支持 k=3/6/9/12（或任意正整数）
        """
        B, T, C, H, W = images.shape
        B2, N = t_src.shape
        assert B2 == B
        assert img_patch_size > 0

        # 1) 逐 query 取对应帧: frames (B,N,C,H,W) -> (BN,C,H,W)
        b_idx = torch.arange(B, device=images.device).view(B, 1).expand(B, N)
        frames = images[b_idx, t_src]                      # (B,N,C,H,W)
        frames_bn = frames.reshape(B * N, C, H, W)         # (BN,C,H,W)

        # 2) 构造 k×k 像素偏移（允许偶数 -> 半像素偏移）
        # offsets in pixel units, centered at 0
        k = img_patch_size
        offs = torch.arange(k, device=images.device, dtype=uv.dtype) - (k - 1) / 2.0
        
        yy, xx = torch.meshgrid(offs, offs, indexing="ij")            # (k,k)
        offsets = torch.stack([xx, yy], dim=-1).reshape(1, 1, k, k, 2)  # (1,1,k,k,2)

        # 3) uv -> pixel coords (float, 不 round)
        u = uv[..., 0].clamp(0.0, 1.0)
        v = uv[..., 1].clamp(0.0, 1.0)
        x_pix = u * (W - 1)
        y_pix = v * (H - 1)

        center = torch.stack([x_pix, y_pix], dim=-1).unsqueeze(2).unsqueeze(3)  # (B,N,1,1,2)
        grid_pix = center + offsets                                             # (B,N,k,k,2)

        # 4) pixel coords -> [-1,1] grid
        if align_corners:
            gx = 2.0 * grid_pix[..., 0] / (W - 1) - 1.0
            gy = 2.0 * grid_pix[..., 1] / (H - 1) - 1.0
        else:
            gx = (2.0 * grid_pix[..., 0] + 1.0) / W - 1.0
            gy = (2.0 * grid_pix[..., 1] + 1.0) / H - 1.0

        grid = torch.stack([gx, gy], dim=-1)                  # (B,N,k,k,2)
        grid_bn = grid.reshape(B * N, k, k, 2)                # (BN,k,k,2)

        # 5) grid_sample -> (BN,C,k,k) -> (B,N,C,k,k)
        patches_bn = F.grid_sample(
            frames_bn, grid_bn,
            mode=mode,
            padding_mode=padding_mode,
            align_corners=align_corners,
        )
        patches = patches_bn.reshape(B, N, C, k, k)
        return patches


    def forward(self, meta, query: torch.Tensor, images: torch.Tensor | None = None) -> torch.Tensor:
        """
        query:  (B, N, 5)
        images: (B, C, T, H, W) 
        return: (B, N, D)
        """
        if meta is None:
            raise ValueError("meta = None")
        if query is None:
            raise ValueError("query = None")
        if "img_patch_size" not in meta:
            raise KeyError("meta must contain 'img_patch_size'")

        v = meta["img_patch_size"]
        ac = meta["align_corners"]
        if torch.is_tensor(v):
            img_patch_size = int(v.item())
        else:
            img_patch_size = int(v)
        if torch.is_tensor(ac):
            ac = int(ac.item())
        else:
            ac = int(v)
            
        if img_patch_size != 0:
            assert images is not None
            # want (B,T,C,H,W)
            if images.shape[1] == self.image_in_chans:      # (B,C,T,H,W)
                images = images.transpose(1,2)
            elif images.shape[2] == self.image_in_chans:    # already (B,T,C,H,W)
                pass
            else:
                raise ValueError(f"Cannot infer layout from images.shape={images.shape}")



        if img_patch_size!=0 and img_patch_size not in self.img_patch_sizes :
            raise ValueError(f"Unsupported img_patch_size={img_patch_size}. Supported: {self.img_patch_sizes}")

        #images: (B, T, C, H, W) 
        if query.dim() != 3 or query.size(-1) != 5:
            raise ValueError(f"Expected query shape [B, N, 5], got {tuple(query.shape)}")

        B, N, _ = query.shape
        tok_dtype = self.t_src_emb.weight.dtype
        uv = query[..., 0:2].to(dtype=tok_dtype)  # (B,N,2)
        uv = uv.clamp(0.0, 1.0)

        # 时间索引：转 long 并 clamp
        T = images.shape[1]
        if T != self.num_frames:
            raise ValueError("T != images.shape[1]")
        t_src = query[..., 2].round().long().clamp(0, self.num_frames - 1)  # (B,N)
        t_tgt = query[..., 3].round().long().clamp(0, self.num_frames - 1)
        t_cam = query[..., 4].round().long().clamp(0, self.num_frames - 1)

        # token1
        uv_f = uv.to(torch.float32)。           #float32对于fourier计算更加友好
        uv_feat = self._fourier_uv(uv_f)        # (B,N,4F[+2])
        token1 = self.uv_proj(uv_feat).to(tok_dtype)        # (B,N,D)

        # token2/3/4
        token2 = self.t_src_emb(t_src)        # (B,N,D)
        token3 = self.t_tgt_emb(t_tgt)
        token4 = self.t_cam_emb(t_cam)

        if img_patch_size != 0:
            # token5
            if images is None:
                raise ValueError("enable_token5=True requires images input of shape [B,T,C,H,W].")
            if images.dim() != 5:
                raise ValueError(f"Expected images shape [B,T,C,H,W], got {tuple(images.shape)}")
            if images.shape[0] != B:
                raise ValueError(f"images batch {images.shape[0]} != query batch {B}")
            if images.shape[2] != self.image_in_chans:
                raise ValueError(f"images C={images.shape[2]} != image_in_chans={self.image_in_chans}")

            patches = self._extract_kxk_patches_grid_sample(
                images, t_src=t_src, uv=uv,
                img_patch_size=img_patch_size,
                mode="bilinear", padding_mode="border", align_corners=ac
            ) 
            k = img_patch_size
            patch_flat = patches.reshape(B, N, self.image_in_chans * (k * k)).to(dtype=token1.dtype)


            token5 = self.patch_mlps[str(img_patch_size)](patch_flat)

            out = token1 + token2 + token3 + token4 + token5
        else:
            out = token1 + token2 + token3 + token4

        out = self.out_drop(out)
        
        out = self.out_mlp(out)
        
        return out