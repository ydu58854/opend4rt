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
from timm.models.layers import trunc_normal_ as __call_trunc_normal_
from timm.models.registry import register_model
from .modules import (
    Block,
    PatchEmbed,
    _cfg,
    get_sinusoid_encoding_table,
)
from .encoder import D4RTEncoder
from .decoder import D4RTDecoder
from .query_embed import QueryEmbedding
class D4RT(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(
        self,
        img_size=256,
        patch_size=16,
        encoder_in_chans=3,
        encoder_embed_dim=768,
        encoder_depth=40,
        encoder_num_heads=12,  
        decoder_embed_dim=768,
        decoder_depth=8,
        decoder_num_heads=8,
        mlp_ratio=4.,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.,
        norm_layer=nn.LayerNorm,
        init_values=0.,
        use_learnable_pos_emb=False,
        tubelet_size=2,
        num_classes=0,  # avoid the error from create_fn in timm
        in_chans=0,  # avoid the error from create_fn in timm
        with_cp=False,
        all_frames=48,
        cos_attn=False,
        embed_freqs = 10.0,
        embed_include_uv = False,
        patch_mlp_ratio = 4.0 ,
        img_patch_sizes = (3,6,9,12,15),
    ):
        super().__init__()
        self.encoder = D4RTEncoder(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=encoder_in_chans,
            embed_dim=encoder_embed_dim,
            depth=encoder_depth,
            num_heads=encoder_num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            init_values=init_values,
            tubelet_size=tubelet_size,
            use_learnable_pos_emb=use_learnable_pos_emb,
            with_cp=with_cp,
            all_frames=all_frames,
            cos_attn=cos_attn)

        self.decoder = D4RTDecoder(
            embed_dim=decoder_embed_dim,
            depth=decoder_depth,
            num_heads=decoder_num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            init_values=init_values,
            cos_attn=cos_attn)
        
        self.query_embed = QueryEmbedding(
            embed_dim = decoder_embed_dim,
            num_frames = all_frames ,          # S
            num_freqs = embed_freqs,      # Fourier 频带数量
            include_uv = embed_include_uv,
            dropout = drop_rate,
            image_in_chans = encoder_in_chans,          # images 的 C
            patch_mlp_ratio = patch_mlp_ratio,
            out_mlp_ratio = mlp_ratio,
            img_patch_sizes = img_patch_sizes,
        )



        self.pos_embed = get_sinusoid_encoding_table(
            self.encoder.patch_embed.num_patches, decoder_embed_dim)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return self.encoder.get_num_layers() + self.decoder.get_num_layers()

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'register_token'}

    def forward(self, meta, images, query):
        global_scene_rep = self.encoder(images, meta) # [B, Nc, C1]
        query = self.query_embed(meta, query,images )  # [B, Nq, C2]
        predictions = self.decoder(query,global_scene_rep)

        return predictions