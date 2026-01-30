import argparse
import json
from collections import Counter
from pathlib import Path

import torch


def summarize_shapes(state_dict, limit=None):
    items = list(state_dict.items())
    items = sorted(items, key=lambda kv: kv[0])
    if limit is not None:
        items = items[:limit]
    return [{"key": k, "shape": tuple(v.shape)} for k, v in items]


def main():
    parser = argparse.ArgumentParser(description="Inspect VideoMAE checkpoint structure")
    parser.add_argument("checkpoint", type=str)
    # 默认 None = 全部 keys；你也可以手动传 --limit 40 之类
    parser.add_argument("--limit", type=int, default=None)
    # 默认输出到 checkpoint 同目录的 vitb.txt
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    checkpoint = torch.load(ckpt_path, map_location="cpu")

    top_level = list(checkpoint.keys()) if isinstance(checkpoint, dict) else []
    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get("module") or checkpoint.get("state_dict") or checkpoint.get("model") or checkpoint
    else:
        state_dict = checkpoint

    prefixes = []
    for key in state_dict.keys():
        prefixes.append(key.split(".")[0])
    prefix_counts = Counter(prefixes)

    summary = {
        "checkpoint_path": str(ckpt_path),
        "top_level_keys": top_level,
        "num_state_dict_keys": len(state_dict),
        "prefix_counts": dict(prefix_counts.most_common()),
        "pos_embed_shape": tuple(state_dict["pos_embed"].shape) if "pos_embed" in state_dict else None,
        "patch_embed_weight_shape": tuple(state_dict["patch_embed.proj.weight"].shape)
        if "patch_embed.proj.weight" in state_dict
        else None,
        # 这里会打印全部 keys（按字母序）
        "sample_keys": summarize_shapes(state_dict, limit=args.limit),
    }

    text = json.dumps(summary, indent=2)

    # 输出文件：默认在 checkpoint 同目录下 vitb.txt
    out_path = Path(args.out) if args.out is not None else (ckpt_path.parent / "vit_g_hybrid_pt_1200e_k710_ft.txt")
    out_path.write_text(text + "\n", encoding="utf-8")

    # 终端也打印一份
    print(text)
    print(f"\n[Saved] {out_path}")


if __name__ == "__main__":
    main()
