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
import math
from .modules import (
    Block,
    PatchEmbed,
    _cfg,
    get_sinusoid_encoding_table,
)
class D4RTDecoder(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self,
                 embed_dim=768,
                 depth=8,
                 num_heads=12,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 norm_layer=nn.LayerNorm,
                 init_values=None,
                 cos_attn=False):
        super().__init__()
        # num_features for consistency with other models
        self.num_features = self.embed_dim = embed_dim
        self.head = nn.Linear(embed_dim, 13)     #预测13个维度的4D信息
        """
        4D Predictions (13 dims)
        3D : XYZ position (3 dims)
        2D : UV position (2 dims)
        vis : Point visibility (1 dim)
        disp : Point motion (3 dims)
        normal : Surface normal (3 dims)
        conf : Confidence (1 dim)
        """

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)
               ]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                init_values=init_values,
                cos_attn=cos_attn,
                attn_type="cross") for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)
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
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {}


    def forward(self, query, Global_Scene_Rep):
        for blk in self.blocks:
            query = blk(query, Global_Scene_Rep)

        predictions = self.head(self.norm(query))
        return predictions
