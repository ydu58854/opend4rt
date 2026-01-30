import argparse
import json
from collections import Counter
from pathlib import Path

import torch


def summarize_shapes(state_dict, limit=40):
    items = list(state_dict.items())
    items = sorted(items, key=lambda kv: kv[0])
    sample = items[:limit]
    return [{"key": k, "shape": tuple(v.shape)} for k, v in sample]


def main():
    parser = argparse.ArgumentParser(description="Inspect VideoMAE checkpoint structure")
    parser.add_argument("checkpoint", type=str)
    parser.add_argument("--limit", type=int, default=40)
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    checkpoint = torch.load(ckpt_path, map_location="cpu")

    top_level = list(checkpoint.keys()) if isinstance(checkpoint, dict) else []
    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get("model") or checkpoint.get("state_dict") or checkpoint
    else:
        state_dict = checkpoint

    prefixes = []
    for key in state_dict.keys():
        prefix = key.split(".")[0]
        prefixes.append(prefix)
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
        "sample_keys": summarize_shapes(state_dict, limit=args.limit),
    }

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
