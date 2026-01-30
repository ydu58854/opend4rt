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
from typing import cast
import os
import re
import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
import torch.nn.functional as F
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
                 depth=40,
                 num_heads=12,
                 mlp_ratio=4.,
                 qkv_bias=True,
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
        if depth == 40:
            mlp_ratio = 48/11           #只有vit-g的mlp_ratio和另外的不同
        self.depth=depth
        self.tubelet_size = tubelet_size
        self.aspect_ratio_fc = nn.Linear(1, embed_dim)          # ar 和 register
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
        assert img_size % patch_size ==0
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
                drop_path=dpr[2*i],                   #注意，如果改了aa_order，这里的奇偶性也得改，奇偶里面内含了顺序
                norm_layer=norm_layer,
                init_values=init_values,
                cos_attn=cos_attn,
                attn_type="self") for i in range(self.aa_num)
        ])
        # for name, value in (("_resnet_mean", _RESNET_MEAN), ("_resnet_std", _RESNET_STD)):
        #     self.register_buffer(name, torch.FloatTensor(value).view(1, 1, 3, 1, 1), persistent=False)
            
        self.fc_norm = norm_layer(embed_dim)


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

    @torch.jit.ignore()
    def no_weight_decay(self):
        return {'pos_embed'}         


    def load_videomae_vit_encoder(self, checkpoint_path: str, strict: bool = True):
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        sd = ckpt.get("state_dict") or ckpt.get("model") or ckpt.get("module") or ckpt

        def strip_prefix(k: str) -> str:
            for p in ("module.", "model.", "encoder.", "backbone."):
                if k.startswith(p):
                    k = k[len(p):]
            return k

        allowed_block_suffixes = {
            "attn.proj.bias",
            "attn.proj.weight",
            "attn.q_bias",
            "attn.qkv.weight",
            "attn.v_bias",
            "mlp.fc1.bias",
            "mlp.fc1.weight",
            "mlp.fc2.bias",
            "mlp.fc2.weight",
            "norm1.bias",
            "norm1.weight",
            "norm2.bias",
            "norm2.weight",
        }
        allowed_top_level = {
            "fc_norm.bias",
            "fc_norm.weight",
            "norm.bias",
            "norm.weight",
            "patch_embed.proj.bias",
            "patch_embed.proj.weight",
        }

        block_re = re.compile(r"^blocks\.(\d+)\.(.+)$")

        if not (hasattr(self, "frame_blocks") and hasattr(self, "global_blocks")):
            raise RuntimeError("Target encoder must have `frame_blocks` and `global_blocks` for interleaved mapping.")

        target_sd = self.state_dict()

        def map_block_idx(k: int):
            return ("frame_blocks", k // 2) if (k % 2 == 0) else ("global_blocks", k // 2)

        def accept(new_key: str, tensor: torch.Tensor, src_key: str):
            if new_key not in target_sd:
                return False, f"{src_key} -> {new_key} (missing in target)"
            if tuple(tensor.shape) != tuple(target_sd[new_key].shape):
                return False, (
                    f"{src_key} -> {new_key} "
                    f"(shape ckpt {tuple(tensor.shape)} != tgt {tuple(target_sd[new_key].shape)})"
                )
            return True, ""

        @torch.no_grad()
        def resize_patch_embed_weight_3d(w: torch.Tensor, target_hw: int):
            """
            w: [Cout, Cin, T, H, W]  (Conv3d kernel)
            仅对 H,W 做插值 resize 到 target_hw x target_hw
            """
            assert w.ndim == 5, f"patch_embed.proj.weight must be 5D, got {w.ndim}"
            Cout, Cin, T, H, W = w.shape
            if H == target_hw and W == target_hw:
                return w
            if H != W:
                raise ValueError(f"Only supports square kernels, got H={H}, W={W}")

            # (Cout,Cin,T,H,W) -> (Cout*Cin*T, 1, H, W)
            w2 = w.contiguous().view(Cout * Cin * T, 1, H, W)

            # bicubic 插值到目标大小
            w2 = F.interpolate(w2, size=(target_hw, target_hw), mode="bicubic", align_corners=False)

            # -> (Cout,Cin,T,target_hw,target_hw)
            w2 = w2.view(Cout, Cin, T, target_hw, target_hw).contiguous()
            return w2

        remapped = {}
        ignored = []
        errors = []
        loaded = []

        for k, v in sd.items():
            k0 = strip_prefix(k)

            # 顶层 4 个
            if k0 in allowed_top_level:
                # ====== 仅改 patch_embed 的逻辑开始 ======
                if k0 == "patch_embed.proj.weight":
                    if k0 not in target_sd:
                        errors.append(f"{k0} -> {k0} (missing in target)")
                        continue

                    tgt = target_sd[k0]
                    # 允许：只差 H,W 的情况，做插值再加载
                    if tuple(v.shape) != tuple(tgt.shape):
                        # 期望都是 5D: [Cout, Cin, T, H, W]
                        if v.ndim == 5 and tgt.ndim == 5 and tuple(v.shape[:3]) == tuple(tgt.shape[:3]):
                            tgt_h, tgt_w = int(tgt.shape[-2]), int(tgt.shape[-1])
                            if tgt_h == tgt_w:
                                try:
                                    v_resized = resize_patch_embed_weight_3d(v, target_hw=tgt_h)
                                except Exception as e:
                                    errors.append(f"{k0} (resize failed: {e})")
                                    continue

                                # resize 后再做最终 shape 校验
                                if tuple(v_resized.shape) != tuple(tgt.shape):
                                    errors.append(
                                        f"{k0} (after resize shape {tuple(v_resized.shape)} != tgt {tuple(tgt.shape)})"
                                    )
                                    continue

                                remapped[k0] = v_resized
                                loaded.append(f"{k0} (resized {tuple(v.shape[-2:])} -> {tuple(tgt.shape[-2:])})")
                                continue
                            else:
                                errors.append(f"{k0} (target kernel not square: {tgt_h}x{tgt_w})")
                                continue
                        else:
                            errors.append(
                                f"{k0} (unsupported shape mismatch: ckpt {tuple(v.shape)} vs tgt {tuple(tgt.shape)})"
                            )
                            continue

                    # shape 一致则正常加载
                    remapped[k0] = v
                    loaded.append(k0)
                    continue

                # bias 仍然严格匹配
                if k0 == "patch_embed.proj.bias":
                    ok, why = accept(k0, v, k0)
                    if ok:
                        remapped[k0] = v
                        loaded.append(k0)
                    else:
                        errors.append(why)
                    continue
                # ====== 仅改 patch_embed 的逻辑结束 ======
                
                if k0.startswith("norm."):
                    # 优先直接加载到 norm.*（如果目标确实有）
                    if k0 in target_sd:
                        ok, why = accept(k0, v, k0)
                        if ok:
                            remapped[k0] = v
                            loaded.append(k0)
                        else:
                            errors.append(why)
                        continue

                    # 否则尝试映射到 fc_norm.*
                    k1 = "fc_norm." + k0[len("norm."):]  # norm.weight -> fc_norm.weight
                    ok, why = accept(k1, v, k0)
                    if ok:
                        remapped[k1] = v
                        loaded.append(f"{k0} -> {k1}")
                    else:
                        errors.append(why)
                    continue

                # fc_norm.* 维持原逻辑：严格匹配
                ok, why = accept(k0, v, k0)
                if ok:
                    remapped[k0] = v
                    loaded.append(k0)
                else:
                    errors.append(why)
                continue

            # blocks.N.suffix
            m = block_re.match(k0)
            if m:
                bidx = int(m.group(1))
                suffix = m.group(2)

                if suffix not in allowed_block_suffixes:
                    ignored.append(k0)
                    continue

                dst_list, dst_i = map_block_idx(bidx)

                # 越界硬错误
                if dst_list == "frame_blocks" and dst_i >= len(self.frame_blocks):
                    errors.append(f"{k0} -> {dst_list}.{dst_i}.{suffix} (OOR: len(frame_blocks)={len(self.frame_blocks)})")
                    continue
                if dst_list == "global_blocks" and dst_i >= len(self.global_blocks):
                    errors.append(f"{k0} -> {dst_list}.{dst_i}.{suffix} (OOR: len(global_blocks)={len(self.global_blocks)})")
                    continue

                new_key = f"{dst_list}.{dst_i}.{suffix}"

                # ✅ q_bias / v_bias 的正确逻辑：目标有没有注册决定能不能加载
                if suffix in ("attn.q_bias", "attn.v_bias"):
                    if new_key not in target_sd:
                        if strict:
                            errors.append(f"{k0} -> {new_key} (target has no q_bias/v_bias; did you set qkv_bias=False?)")
                        else:
                            ignored.append(k0)
                        continue

                ok, why = accept(new_key, v, k0)
                if ok:
                    remapped[new_key] = v
                    loaded.append(f"{k0} -> {new_key}")
                else:
                    errors.append(why)
                continue

            # 其它一律忽略
            ignored.append(k0)

        # ===== 反向 expected 检查：如果目标里有 q_bias/v_bias，就要求都加载到 =====
        expected = set(allowed_top_level)

        fixed_suffixes = allowed_block_suffixes
        for i in range(len(self.frame_blocks)):
            for suf in fixed_suffixes:
                kexp = f"frame_blocks.{i}.{suf}"
                if kexp in target_sd:
                    expected.add(kexp)
        for i in range(len(self.global_blocks)):
            for suf in fixed_suffixes:
                kexp = f"global_blocks.{i}.{suf}"
                if kexp in target_sd:
                    expected.add(kexp)

        expected_in_target = [k for k in expected if k in target_sd]
        missing_expected = [k for k in expected_in_target if k not in remapped]

        if strict and (errors or missing_expected):
            msg = ["Strict interleaved VideoMAE encoder load FAILED."]
            if errors:
                msg.append(f"\n[Errors] ({len(errors)})")
                msg.extend(errors[:80])
                if len(errors) > 80:
                    msg.append("... (truncated)")
            if missing_expected:
                msg.append(f"\n[Expected but not loaded] ({len(missing_expected)})")
                msg.extend(missing_expected[:80])
                if len(missing_expected) > 80:
                    msg.append("... (truncated)")
            raise RuntimeError("\n".join(msg))
        elif not strict and (errors or missing_expected):
            print(
                "[Encoder pretrained][WARN] non-strict load has mismatches: "
                f"errors={len(errors)}, missing_expected={len(missing_expected)}"
            )
            if errors:
                preview = errors[:20]
                print("[Encoder pretrained][WARN] error samples:")
                for item in preview:
                    print(f"  - {item}")
                if len(errors) > len(preview):
                    print("  ... (truncated)")
            if missing_expected:
                preview = missing_expected[:20]
                print("[Encoder pretrained][WARN] missing_expected samples:")
                for item in preview:
                    print(f"  - {item}")
                if len(missing_expected) > len(preview):
                    print("  ... (truncated)")

        load_res = self.load_state_dict(remapped, strict=False)
        info = {
            "loaded_count": len(loaded),
            "ignored_count": len(ignored),
            "error_count": len(errors),
            "missing_expected_count": len(missing_expected),
            "loaded_samples": loaded[:20],
            "error_samples": errors[:20],
            "missing_expected_samples": missing_expected[:20],
        }
        return load_res, info




    def load_videomae_encoder(self, checkpoint_path: str, variant: str = "vit-b", strict: bool = True):
        if variant == "vit-b":
            checkpoint_path = os.path.join(checkpoint_path,"VideoMAE2","mae-b","pytorch_model.bin")
            return self.load_videomae_vit_encoder(checkpoint_path, strict=strict)
        elif variant == "vit-l":
            checkpoint_path = os.path.join(checkpoint_path,"VideoMAE2","mae-l","pytorch_model.bin")
            return self.load_videomae_vit_encoder(checkpoint_path, strict=strict)
        elif variant == "vit-h":
            checkpoint_path = os.path.join(checkpoint_path,"VideoMAE2","mae-h","pytorch_model.bin")
            return self.load_videomae_vit_encoder(checkpoint_path, strict=strict)
        elif variant == "vit-g":
            checkpoint_path = os.path.join(checkpoint_path,"VideoMAE2","mae-g","vit_g_ps14_ak_ft_ckpt_7_clean.pth")
            return self.load_videomae_vit_encoder(checkpoint_path, strict=strict)
            
        raise ValueError(f"Unsupported VideoMAE variant: {variant}")


    def forward(self, meta, images):
        #images : [B, C, T, H, W]
        images_tokens: torch.Tensor = self.patch_embed(images)
        B, S, P, C = images_tokens.shape   #我修改了patch_embed的接口，这里没有问题
        # [B, S, P, embed_dim]
        images_tokens = images_tokens.reshape(B, S*P ,C)
        images_tokens = images_tokens + self.pos_embed.type_as(images_tokens).to(images_tokens.device)
        images_tokens = images_tokens.reshape(B, S, P, C)
        assert(S == self.all_frames//self.tubelet_size)       #固定48帧，合并之后为24
        #TODO:上游要实现这个接口，传入ar,目前默认B个batch里面ar都是一样的，ar是单个float；后续可以改成（B，）的tensor

        ar = meta["aspect_ratio"]
        if not torch.is_tensor(ar):
            raise ValueError("aspect ratio should be a tensor")
        ar = ar.to(device=images.device, dtype=images_tokens.dtype)
        if ar.shape != (B,1):
            raise ValueError("aspect ratio should have shape (B,1), got {}".format(ar.shape))

        ar = ar.view(B,1)                 # (B,1)

        ar = ar[:, None, :].expand(B, S, 1).reshape(B*S, 1)  # (B*S,1)
        aspect_token = self.aspect_ratio_fc(ar).unsqueeze(1) # (B*S,1,C)

        images_tokens = images_tokens.reshape(B*S,P,C)
        tokens: torch.Tensor = torch.cat([aspect_token, images_tokens], dim=1)
        _, P, C = tokens.shape      #更新P为tokens总长度
        frame_idx = 0
        global_idx = 0
        for _ in range(self.aa_num):
            for aa_type in self.aa_order:
                if aa_type == "frame":
                    
                    if tokens.shape != (B * S, P, C):
                        tokens = tokens.reshape(B, S, P, C).reshape(B * S, P, C)           
                        
                    if self.with_cp:
                        tokens = cast(torch.Tensor, cp.checkpoint(self.frame_blocks[frame_idx], tokens))
                    else:
                        tokens = self.frame_blocks[frame_idx](tokens)
                    frame_idx += 1
                    tokens = tokens.reshape(B, S, P, C)

                    
                elif aa_type == "global":
                    
                    if tokens.shape != (B, S * P, C):
                        tokens = tokens.reshape(B, S, P, C).reshape(B, S * P, C)
                        
                    if self.with_cp:
                        tokens = cast(torch.Tensor, cp.checkpoint(self.global_blocks[global_idx], tokens))
                    else:
                        tokens = self.global_blocks[global_idx](tokens)
                    global_idx += 1
                    tokens = tokens.reshape(B, S, P, C)
                else:
                    raise ValueError(f"Unknown attention type: {aa_type}")

        tokens=tokens.reshape(B, S * P, C)
        tokens = self.fc_norm(tokens)
        return tokens
