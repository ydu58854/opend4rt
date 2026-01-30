import sys
from pathlib import Path
import os
import re
import random

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model import D4RTEncoder  # type: ignore


# ========= 你需要改的：构建你的 encoder =========
def build_encoder():
    encoder_b = D4RTEncoder(img_size=256, patch_size=16, embed_dim=768, depth=12, num_heads=12, tubelet_size=2, all_frames=48)
    encoder_l = D4RTEncoder(img_size=256, patch_size=16, embed_dim=1024, depth=24, num_heads=16, tubelet_size=2, all_frames=48)
    encoder_h = D4RTEncoder(img_size=256, patch_size=16, embed_dim=1280, depth=32, num_heads=16, tubelet_size=2, all_frames=48)
    encoder_g = D4RTEncoder(img_size=256, patch_size=16, mlp_ratio=4.363636363636363, embed_dim=1408, depth=40, num_heads=16, tubelet_size=2, all_frames=48)
    return encoder_g  # 在这里选择返回的种类，然后终端命令也选择对应路径


# ========= 工具：从 checkpoint 取出 backbone 级别 state_dict =========
def extract_backbone_state_dict(ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt.get("state_dict") or ckpt.get("model") or ckpt.get("module") or ckpt

    def strip_prefix(k: str) -> str:
        for p in ("module.", "model.", "encoder.", "backbone."):
            if k.startswith(p):
                k = k[len(p):]
        return k

    out = {}
    for k, v in sd.items():
        out[strip_prefix(k)] = v
    return out


# ========= 交错映射：ckpt blocks.k -> frame/global =========
ALLOWED_BLOCK_SUFFIXES = [
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
]

# ckpt 允许出现的 top-level keys（注意：ckpt 通常是 norm.*，target 可能是 fc_norm.*）
CKPT_TOP_LEVEL_KEYS = [
    "norm.bias",
    "norm.weight",
    "fc_norm.bias",     # 兼容极少数 ckpt
    "fc_norm.weight",
    "patch_embed.proj.bias",
    "patch_embed.proj.weight",
]

BLOCK_RE = re.compile(r"^blocks\.(\d+)\.(.+)$")


@torch.no_grad()
def resize_patch_embed_weight_3d(w: torch.Tensor, target_hw: int) -> torch.Tensor:
    """
    w: [Cout, Cin, T, H, W]
    仅对 H/W 做 bicubic 插值到 target_hw
    """
    assert w.ndim == 5, f"patch_embed.proj.weight must be 5D, got {w.ndim}"
    Cout, Cin, T, H, W = w.shape
    if H == target_hw and W == target_hw:
        return w
    if H != W:
        raise ValueError(f"Only supports square kernels, got H={H}, W={W}")

    w2 = w.contiguous().view(Cout * Cin * T, 1, H, W)
    w2 = F.interpolate(w2, size=(target_hw, target_hw), mode="bicubic", align_corners=False)
    w2 = w2.view(Cout, Cin, T, target_hw, target_hw).contiguous()
    return w2


def can_resize_patch_embed(ckpt_w: torch.Tensor, tgt_w: torch.Tensor) -> bool:
    # 允许：只差 HW，且都是 5D，前三维一致，target kernel 方形
    if ckpt_w.ndim != 5 or tgt_w.ndim != 5:
        return False
    if tuple(ckpt_w.shape[:3]) != tuple(tgt_w.shape[:3]):
        return False
    th, tw = int(tgt_w.shape[-2]), int(tgt_w.shape[-1])
    return th == tw


def map_ckpt_key_to_target(enc, ckpt_key: str):
    """
    返回 target_key 或 None（表示不在白名单/不加载）。
    固定规则：
      - blocks.k.* : k 偶数->frame_blocks[k//2], 奇数->global_blocks[k//2]
      - 顶层：norm.* / patch_embed.proj.*
      - 额外：ckpt norm.* -> target fc_norm.*（如果 target 没有 norm.* 但有 fc_norm.*）
    """
    tgt_sd = enc.state_dict()

    # --- top-level ---
    if ckpt_key in CKPT_TOP_LEVEL_KEYS:
        # norm -> fc_norm 映射（只有当 target 没有 norm.* 但有 fc_norm.* 时才发生）
        if ckpt_key.startswith("norm."):
            if ckpt_key in tgt_sd:
                return ckpt_key
            fc_key = "fc_norm." + ckpt_key[len("norm."):]
            if fc_key in tgt_sd:
                return fc_key
            return ckpt_key  # 让上层报 missing in target

        # fc_norm 在 ckpt 里也可能存在（兼容）
        if ckpt_key.startswith("fc_norm."):
            if ckpt_key in tgt_sd:
                return ckpt_key
            norm_key = "norm." + ckpt_key[len("fc_norm."):]
            if norm_key in tgt_sd:
                return norm_key
            return ckpt_key

        # patch_embed 直接同名
        return ckpt_key

    # --- blocks ---
    m = BLOCK_RE.match(ckpt_key)
    if not m:
        return None

    bidx = int(m.group(1))
    suffix = m.group(2)
    if suffix not in ALLOWED_BLOCK_SUFFIXES:
        return None

    if bidx % 2 == 0:
        dst_list = "frame_blocks"
        dst_i = bidx // 2
        return f"{dst_list}.{dst_i}.{suffix}"
    else:
        dst_list = "global_blocks"
        dst_i = bidx // 2
        return f"{dst_list}.{dst_i}.{suffix}"


def collect_expected_target_keys(enc):
    """
    依据目标 state_dict，构造“应该被加载”的 keys：
      - 顶层：target 有 fc_norm.* 就只期待 fc_norm.*；否则期待 norm.*
      - patch_embed.proj.*
      - frame/global 每个 block 的 13 个（仅加入 target 真正存在的 keys）
    """
    tgt_sd = enc.state_dict()
    expected = []

    # ---- norm / fc_norm：只选 target 存在的那一组 ----
    if ("fc_norm.weight" in tgt_sd) and ("fc_norm.bias" in tgt_sd):
        expected += ["fc_norm.bias", "fc_norm.weight"]
    else:
        # fallback to norm
        if "norm.bias" in tgt_sd:
            expected.append("norm.bias")
        if "norm.weight" in tgt_sd:
            expected.append("norm.weight")

    # ---- patch_embed ----
    for k in ("patch_embed.proj.bias", "patch_embed.proj.weight"):
        if k in tgt_sd:
            expected.append(k)

    # ---- blocks ----
    for i in range(len(enc.frame_blocks)):
        for suf in ALLOWED_BLOCK_SUFFIXES:
            k = f"frame_blocks.{i}.{suf}"
            if k in tgt_sd:
                expected.append(k)

    for i in range(len(enc.global_blocks)):
        for suf in ALLOWED_BLOCK_SUFFIXES:
            k = f"global_blocks.{i}.{suf}"
            if k in tgt_sd:
                expected.append(k)

    return expected


def tensor_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    return torch.equal(a.cpu(), b.cpu())


def tensor_close(a: torch.Tensor, b: torch.Tensor, rtol=1e-5, atol=1e-6) -> bool:
    return torch.allclose(a.cpu(), b.cpu(), rtol=rtol, atol=atol)


def main():
    ckpt_path = os.environ.get("CKPT", "")
    if not ckpt_path:
        print("请用环境变量指定 CKPT 路径，例如：")
        print("  CKPT=/path/to/pytorch_model.bin python ckpt_load_test.py")
        return

    strict = os.environ.get("STRICT", "1") == "1"
    sample_blocks = int(os.environ.get("SAMPLE_BLOCKS", "8"))
    seed = int(os.environ.get("SEED", "0"))
    random.seed(seed)
    torch.manual_seed(seed)

    # ====== 1) 构建 encoder ======
    enc = build_encoder()
    enc.eval()

    # ====== 2) 读取 ckpt state_dict（已剥 backbone 前缀）======
    ckpt_sd = extract_backbone_state_dict(ckpt_path)

    # ====== 3) 调你的 loader ======
    out = enc.load_videomae_vit_encoder(ckpt_path, strict=strict)
    if isinstance(out, tuple) and len(out) == 2:
        load_res, info = out
    else:
        load_res, info = out, {}

    print("\n========== Loader returned ==========")
    print("load_state_dict.missing_keys (first 20):", getattr(load_res, "missing_keys", [])[:20])
    print("load_state_dict.unexpected_keys (first 20):", getattr(load_res, "unexpected_keys", [])[:20])
    if info:
        print("info:", info)

    tgt_sd_after = enc.state_dict()
    expected_target_keys = collect_expected_target_keys(enc)

    # 建立 target_key -> ckpt_key（通过 map_ckpt_key_to_target）
    target_to_ckpt = {}
    for ck in ckpt_sd.keys():
        tk = map_ckpt_key_to_target(enc, ck)
        if tk is None:
            continue
        target_to_ckpt.setdefault(tk, ck)

    missing_in_ckpt = []
    missing_in_target = []
    shape_mismatch = []
    resizable_patch = []  # 记录允许插值的 patch_embed mismatch

    for tk in expected_target_keys:
        ck = target_to_ckpt.get(tk, None)
        if ck is None:
            missing_in_ckpt.append(tk)
            continue
        if tk not in tgt_sd_after:
            missing_in_target.append(tk)
            continue

        ckpt_t = ckpt_sd[ck]
        tgt_t = tgt_sd_after[tk]

        if tuple(ckpt_t.shape) != tuple(tgt_t.shape):
            # ✅ patch_embed.proj.weight 允许通过插值解决
            if tk == "patch_embed.proj.weight" and can_resize_patch_embed(ckpt_t, tgt_t):
                resizable_patch.append((ck, tk, tuple(ckpt_t.shape), tuple(tgt_t.shape)))
            else:
                shape_mismatch.append((ck, tk, tuple(ckpt_t.shape), tuple(tgt_t.shape)))

    print("\n========== Structural checks ==========")
    print(f"Expected target keys: {len(expected_target_keys)}")
    print(f"Missing mapped ckpt entries: {len(missing_in_ckpt)}")
    print(f"Missing target entries (unexpected): {len(missing_in_target)}")
    print(f"Shape mismatch pairs (hard): {len(shape_mismatch)}")
    print(f"Patch_embed resizable mismatches (allowed): {len(resizable_patch)}")

    if missing_in_ckpt[:10]:
        print("  missing_in_ckpt (first 10):")
        for x in missing_in_ckpt[:10]:
            print("   -", x)

    if shape_mismatch[:10]:
        print("  shape_mismatch (first 10):")
        for ck, tk, s1, s2 in shape_mismatch[:10]:
            print(f"   - {ck} -> {tk}: ckpt{s1} vs tgt{s2}")

    if resizable_patch[:5]:
        print("  resizable_patch (first 5):")
        for ck, tk, s1, s2 in resizable_patch[:5]:
            print(f"   - {ck} -> {tk}: ckpt{s1} ~> tgt{s2} (will compare after resize)")

    # ====== 5) 数值自检 ======
    numeric_fail = []

    def check_pair_with_mapping(ckpt_key: str):
        """
        给定 ckpt_key，自动映射到 target_key 后比较数值。
        patch_embed.proj.weight 允许先 resize 再比较。
        """
        if ckpt_key not in ckpt_sd:
            numeric_fail.append(f"{ckpt_key} (missing in ckpt)")
            return

        target_key = map_ckpt_key_to_target(enc, ckpt_key)
        if target_key is None:
            return

        if target_key not in tgt_sd_after:
            numeric_fail.append(f"{ckpt_key} -> {target_key} (missing in target)")
            return

        ckpt_t = ckpt_sd[ckpt_key]
        tgt_t = tgt_sd_after[target_key]

        # patch_embed.proj.weight：允许 resize 再比较
        if target_key == "patch_embed.proj.weight" and tuple(ckpt_t.shape) != tuple(tgt_t.shape):
            if not can_resize_patch_embed(ckpt_t, tgt_t):
                numeric_fail.append(f"{ckpt_key}->{target_key} (shape mismatch not resizable)")
                return
            th = int(tgt_t.shape[-1])
            ckpt_rs = resize_patch_embed_weight_3d(ckpt_t, target_hw=th)
            # 用 allclose 更稳（插值会有浮点误差）
            if not tensor_close(ckpt_rs, tgt_t, rtol=1e-5, atol=1e-6):
                numeric_fail.append(f"{ckpt_key}->{target_key} (value mismatch after resize)")
            return

        # 其它：必须 shape 一致 + bitwise 一致
        if tuple(ckpt_t.shape) != tuple(tgt_t.shape):
            numeric_fail.append(f"{ckpt_key}->{target_key} (shape mismatch)")
            return
        if not tensor_equal(ckpt_t, tgt_t):
            numeric_fail.append(f"{ckpt_key}->{target_key} (value mismatch)")
            return

    # 顶层关键 key：按 ckpt 的名字检查（norm 会自动映射到 fc_norm）
    for k in ("norm.bias", "norm.weight", "fc_norm.bias", "fc_norm.weight", "patch_embed.proj.bias", "patch_embed.proj.weight"):
        if k in ckpt_sd:
            check_pair_with_mapping(k)

    # 抽样 block：从 ckpt 中解析 block index
    ckpt_block_indices = sorted({
        int(m.group(1))
        for kk in ckpt_sd.keys()
        for m in [BLOCK_RE.match(kk)]
        if m is not None
    })
    if ckpt_block_indices:
        chosen = random.sample(ckpt_block_indices, k=min(sample_blocks, len(ckpt_block_indices)))
        chosen.sort()
        print("\nChosen ckpt block indices for numeric check:", chosen)

        for bidx in chosen:
            for suf in ALLOWED_BLOCK_SUFFIXES:
                ck = f"blocks.{bidx}.{suf}"
                if ck in ckpt_sd:
                    check_pair_with_mapping(ck)

    print("\n========== Numeric checks ==========")
    print(f"Numeric mismatches: {len(numeric_fail)}")
    if numeric_fail[:40]:
        print("  numeric_fail (first 40):")
        for x in numeric_fail[:40]:
            print("   -", x)

    # ====== 6) 最终结论 ======
    # ok 条件：missing_in_ckpt=0、hard shape_mismatch=0、numeric_fail=0
    ok = (len(missing_in_ckpt) == 0) and (len(shape_mismatch) == 0) and (len(numeric_fail) == 0)
    print("\n========== RESULT ==========")
    if ok:
        print("✅ PASS: checkpoint 能按 interleave 规则正确加载；norm->fc_norm 映射与 patch_embed 插值均通过验证。")
    else:
        print("❌ FAIL: 存在缺失/硬 shape mismatch/数值不一致，请看上面的列表定位。")


if __name__ == "__main__":
    main()
