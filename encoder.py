# --------------------------------------------------------
# Based on BEiT, timm, DINO, VideoMAE, VGGT and DeiT code bases
# https://github.com/microsoft/unilm/tree/master/beit
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit
# https://github.com/facebookresearch/dino
# https://github.com/MCG-NJU/VideoMAE
# https://github.com/facebookresearch/vggt
# --------------------------------------------------------'
import sys
sys.path.append(".")

from functools import partial

import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
from timm.models.layers import trunc_normal_ as __call_trunc_normal_
from .modules import (
    Block,
    PatchEmbed,
    _cfg,
    get_sinusoid_encoding_table,
)
def trunc_normal_(tensor, mean=0., std=1.):
    __call_trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)

class D4RTEncoder(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self,
                 img_size=256,
                 patch_size=16,
                 in_chans=3,
                 embed_dim=768,
                 depth=12,
                 num_register_tokens=0,   #默认是0，因为原文没有用register，但是保留之前写的接口，方便以后调试
                 num_heads=12,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_rate=0.,
                 aa_order=["frame","global"],
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 norm_layer=nn.LayerNorm,
                 init_values=None,
                 tubelet_size=2,
                 use_learnable_pos_emb=True,
                 with_cp=False,
                 all_frames=48,
                 cos_attn=False):
        super().__init__()
        self.all_frames = all_frames
        assert(depth%2==0)
        self.aa_num=depth//2
        self.aa_order=aa_order
        self.depth=depth
        self.tubelet_size = tubelet_size
        self.num_register_tokens=num_register_tokens
        self.aspect_ratio_fc = nn.Linear(1, embed_dim)
        self.patch_start_idx = 1 + num_register_tokens           # ar 和 register
        # num_features for consistency with other models
        self.num_features = self.embed_dim = embed_dim
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            num_frames=all_frames,
            tubelet_size=tubelet_size)
        num_patches = self.patch_embed.num_patches
        self.with_cp = with_cp
        assert(all_frames% tubelet_size ==0)
        S=all_frames//tubelet_size      #默认是除以2
        self.register_token = nn.Parameter(torch.randn(1, S, num_register_tokens, embed_dim))
        nn.init.normal_(self.register_token, std=0.02)
        assert img_size % patch_size ==0
        num_patches = img_size**2  //  patch_size**2 * S
        if use_learnable_pos_emb:
            self.pos_embed = nn.Parameter(
                torch.zeros(1, num_patches, embed_dim))
        else:
            # sine-cosine positional embeddings
            self.pos_embed = get_sinusoid_encoding_table(
                num_patches, embed_dim)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)
               ]  # stochastic depth decay rule
        self.global_blocks = nn.ModuleList([
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[2*i+1],
                norm_layer=norm_layer,
                init_values=init_values,
                cos_attn=cos_attn,
                attn_type="self") for i in range(self.aa_num)
        ])
        self.frame_blocks = nn.ModuleList([
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[2*i],
                norm_layer=norm_layer,
                init_values=init_values,
                cos_attn=cos_attn,
                attn_type="self") for i in range(self.aa_num)
        ])
        # for name, value in (("_resnet_mean", _RESNET_MEAN), ("_resnet_std", _RESNET_STD)):
        #     self.register_buffer(name, torch.FloatTensor(value).view(1, 1, 3, 1, 1), persistent=False)
            
        self.norm = norm_layer(embed_dim)


        if use_learnable_pos_emb:
            trunc_normal_(self.pos_embed, std=.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.frame_blocks)+len(self.global_blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed','register_token'}         


    def forward(self, meta, images):
        #images : [B, C, T, H, W]
        images_tokens = self.patch_embed(images)
        B, S, P, C = images_tokens.shape   #我修改了patch_embed的接口，这里没有问题
        # [B, S, P, embed_dim]
        images_tokens = images_tokens.reshape(B, S*P ,C)
        images_tokens = images_tokens + self.pos_embed.type_as(images_tokens).to(images_tokens.device)
        images_tokens = images_tokens.reshape(B, S, P, C)
        num_register_tokens=self.num_register_tokens
        assert(S == self.all_frames//self.tubelet_size)       #固定48帧，合并之后为24
        register_token = slice_expand_and_flatten_per_frame(self.register_token, B, S)
        aspect_ratio=meta["aspect_ratio"]      
        #TODO:上游要实现这个接口，传入ar,目前默认B个batch里面ar都是一样的，ar是单个float；后续可以改成（B，）的tensor

        ar = meta["aspect_ratio"]
        if not torch.is_tensor(ar):
            ar = torch.tensor(ar, device=images.device)
        ar = ar.to(device=images.device, dtype=images_tokens.dtype)

        if ar.numel() == 1:
            ar = ar.view(1,1).expand(B,1)     # (B,1)
        else:
            ar = ar.view(B,1)                 # (B,1)

        ar = ar[:, None, :].expand(B, S, 1).reshape(B*S, 1)  # (B*S,1)
        aspect_token = self.aspect_ratio_fc(ar).unsqueeze(1) # (B*S,1,C)

        images_tokens = images_tokens.reshape(B*S,P,C)
        tokens = torch.cat([aspect_token, register_token, images_tokens], dim=1)
        _, P, C = tokens.shape      #更新P为tokens总长度
        frame_idx = 0
        global_idx = 0
        for _ in range(self.aa_num):
            for aa_type in self.aa_order:
                if aa_type == "frame":
                    
                    if tokens.shape != (B * S, P, C):
                        tokens = tokens.reshape(B, S, P, C).reshape(B * S, P, C)           
                        
                    if self.with_cp:
                        tokens = cp.checkpoint(self.frame_blocks[frame_idx], tokens)
                    else:
                        tokens = self.frame_blocks[frame_idx](tokens)
                    frame_idx += 1
                    tokens = tokens.reshape(B, S, P, C)

                    
                elif aa_type == "global":
                    
                    if tokens.shape != (B, S * P, C):
                        tokens = tokens.reshape(B, S, P, C).reshape(B, S * P, C)
                        
                    if self.with_cp:
                        tokens = cp.checkpoint(self.global_blocks[global_idx], tokens)
                    else:
                        tokens = self.global_blocks[global_idx](tokens)
                    global_idx += 1
                    tokens = tokens.reshape(B, S, P, C)
                else:
                    raise ValueError(f"Unknown attention type: {aa_type}")

        tokens=tokens.reshape(B, S * P, C)
        tokens = self.norm(tokens)
        return tokens
def slice_expand_and_flatten_per_frame(token_tensor, B, S):
    """
    token_tensor: (1, max_frames, X, C)
    return: (B*S, X, C) with per-frame unique tokens
    """
    # (1, S, X, C)
    tok = token_tensor[:, :S, ...]
    # (B, S, X, C)
    tok = tok.expand(B, S, *tok.shape[2:])
    # (B*S, X, C)
    tok = tok.reshape(B * S, *tok.shape[2:])
    return tok

