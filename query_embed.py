from functools import partial

import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
import math
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

        # token5 相关
        image_in_chans: int = 3,          # images 的 C
        patch_mlp_ratio: int = 4,
        out_mlp_ratio : int = 4,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_frames = num_frames
        self.num_freqs = num_freqs
        self.include_uv = include_uv
        self.image_in_chans = image_in_chans

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

        # token5: 3x3 patch -> MLP -> embed_dim
        in_dim = image_in_chans * 9
        patch_mlp_hidden = in_dim * patch_mlp_ratio  
        out_mlp_hidden = embed_dim * out_mlp_ratio
        self.patch_mlp = nn.Sequential(
            nn.Linear(in_dim, patch_mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(patch_mlp_hidden, embed_dim),
            nn.Dropout(dropout),
        )
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

        if self.patch_mlp is not None:
            for m in self.patch_mlp.modules():
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
    def _extract_3x3_patches(images: torch.Tensor,
                            t_src: torch.Tensor,
                            uv: torch.Tensor) -> torch.Tensor:
        """
        images: (B, T, C, H, W)
        t_src:  (B, N) long in [0,T-1]
        uv:     (B, N, 2) float in [0,1]
        return: patches (B, N, C, 3, 3)  (中心为 uv 对应的离散像素)
        """
        B, T, C, H, W = images.shape
        B2, N = t_src.shape
        assert B2 == B

        # uv -> 像素坐标（离散）
        u = uv[..., 0].clamp(0.0, 1.0)  # (B,N)
        v = uv[..., 1].clamp(0.0, 1.0)

        # 映射到 [0, W-1], [0, H-1]
        x = (u * (W - 1)).round().long()  # (B,N)
        y = (v * (H - 1)).round().long()

        # 取 3x3 offsets
        dx = torch.tensor([-1, 0, 1], device=images.device, dtype=torch.long)
        dy = torch.tensor([-1, 0, 1], device=images.device, dtype=torch.long)

        # (B,N,3)
        x_idx = (x.unsqueeze(-1) + dx.view(1, 1, 3)).clamp(0, W - 1)
        y_idx = (y.unsqueeze(-1) + dy.view(1, 1, 3)).clamp(0, H - 1)

        # 先选出每个 query 对应的帧：frames (B,N,C,H,W)
        # 用高级索引实现逐 query 选帧
        b_idx = torch.arange(B, device=images.device).view(B, 1).expand(B, N)  # (B,N)
        frames = images[b_idx, t_src]  # (B,N,C,H,W)

        # 抠 patch：
        # 先在 H 维取 3 行 -> (B,N,C,3,W)
        frames_flat = frames.reshape(B * N, C, H, W)         # (BN,C,H,W)
        y_flat = y_idx.reshape(B * N, 3)                     # (BN,3)
        x_flat = x_idx.reshape(B * N, 3)                     # (BN,3)

        bn = B * N
        bn_idx = torch.arange(bn, device=images.device)

        # rows: (BN, C, 3, W)
        rows = frames_flat[bn_idx[:, None], :, y_flat]       # 这里用到了广播索引
        # rows 结果 shape: (BN, 3, C, W) 或 (BN, C, 3, W) 取决于索引写法
        # 为稳定起见，手动整理维度：
        if rows.dim() == 4 and rows.shape[1] == 3:           # (BN,3,C,W)
            rows = rows.permute(0, 2, 1, 3)                  # -> (BN,C,3,W)

        # 再在 W 维取 3 列：patch (BN,C,3,3)
        # x_flat: (BN,3) -> 用 gather
        x_g = x_flat.unsqueeze(1).unsqueeze(2).expand(bn, C, 3, 3)  # (BN,C,3,3)
        patch = torch.gather(rows, dim=3, index=x_g)                # (BN,C,3,3)

        patches = patch.reshape(B, N, C, 3, 3)                      # (B,N,C,3,3)
        return patches

    def forward(self, query: torch.Tensor, images: torch.Tensor | None = None) -> torch.Tensor:
        """
        query:  (B, N, 5)
        images: (B, C, T, H, W) 
        return: (B, N, D)
        """
        images = images.transpose(1,2)
        #images: (B, T, C, H, W) 
        if query.dim() != 3 or query.size(-1) != 5:
            raise ValueError(f"Expected query shape [B, N, 5], got {tuple(query.shape)}")

        B, N, _ = query.shape

        uv = query[..., 0:2].to(dtype=torch.float32)  # (B,N,2)
        uv = uv.clamp(0.0, 1.0)

        # 时间索引：转 long 并 clamp
        t_src = query[..., 2].round().long().clamp(0, self.num_frames - 1)  # (B,N)
        t_tgt = query[..., 3].round().long().clamp(0, self.num_frames - 1)
        t_cam = query[..., 4].round().long().clamp(0, self.num_frames - 1)

        # token1
        uv_feat = self._fourier_uv(uv)        # (B,N,4F[+2])
        token1 = self.uv_proj(uv_feat)        # (B,N,D)

        # token2/3/4
        token2 = self.t_src_emb(t_src)        # (B,N,D)
        token3 = self.t_tgt_emb(t_tgt)
        token4 = self.t_cam_emb(t_cam)


        # token5
        if images is None:
            raise ValueError("enable_token5=True requires images input of shape [B,T,C,H,W].")
        if images.dim() != 5:
            raise ValueError(f"Expected images shape [B,T,C,H,W], got {tuple(images.shape)}")
        if images.shape[0] != B:
            raise ValueError(f"images batch {images.shape[0]} != query batch {B}")
        if images.shape[2] != self.image_in_chans:
            raise ValueError(f"images C={images.shape[2]} != image_in_chans={self.image_in_chans}")

        patches = self._extract_3x3_patches(images, t_src=t_src, uv=uv)  # (B,N,C,3,3)
        patch_flat = patches.reshape(B, N, self.image_in_chans * 9).to(dtype=token1.dtype)  # (B,N,9C)
        token5 = self.patch_mlp(patch_flat)  # (B,N,D)

        out = token1 + token2 + token3 + token4 + token5

        out = self.out_drop(out)
        
        out = self.out_mlp(out)
        
        return out