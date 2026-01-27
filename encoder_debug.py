import argparse
import os
import torch

from encoder import D4RTEncoder
from model import D4RT


def build_encoder(args):
    return D4RTEncoder(
        img_size=args.img_size,
        patch_size=args.patch_size,
        in_chans=args.in_chans,
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        qkv_bias=args.qkv_bias,
        qk_scale=args.qk_scale,
        drop_rate=args.drop_rate,
        attn_drop_rate=args.attn_drop_rate,
        drop_path_rate=args.drop_path_rate,
        norm_layer=torch.nn.LayerNorm,
        init_values=args.init_values,
        tubelet_size=args.tubelet_size,
        use_learnable_pos_emb=args.use_learnable_pos_emb,
        with_cp=args.with_cp,
        all_frames=args.all_frames,
        cos_attn=args.cos_attn,
    )


def build_d4rt(args):
    return D4RT(
        img_size=args.img_size,
        patch_size=args.patch_size,
        encoder_in_chans=args.in_chans,
        encoder_embed_dim=args.embed_dim,
        encoder_depth=args.depth,
        encoder_num_heads=args.num_heads,
        decoder_embed_dim=args.embed_dim,
        decoder_depth=args.decoder_depth,
        decoder_num_heads=args.decoder_num_heads,
        mlp_ratio=args.mlp_ratio,
        qkv_bias=args.qkv_bias,
        qk_scale=args.qk_scale,
        drop_rate=args.drop_rate,
        attn_drop_rate=args.attn_drop_rate,
        drop_path_rate=args.drop_path_rate,
        norm_layer=torch.nn.LayerNorm,
        init_values=args.init_values,
        use_learnable_pos_emb=args.use_learnable_pos_emb,
        tubelet_size=args.tubelet_size,
        with_cp=args.with_cp,
        all_frames=args.all_frames,
        cos_attn=args.cos_attn,
        embed_freqs=args.embed_freqs,
        embed_include_uv=args.embed_include_uv,
        patch_mlp_ratio=args.patch_mlp_ratio,
        img_patch_sizes=tuple(args.img_patch_sizes),
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Run encoder forward for sanity check")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--in-chans", type=int, default=3)
    parser.add_argument("--embed-dim", type=int, default=768)
    parser.add_argument("--depth", type=int, default=40)
    parser.add_argument("--num-heads", type=int, default=12)
    parser.add_argument("--mlp-ratio", type=float, default=4.0)
    parser.add_argument("--qkv-bias", action="store_true")
    parser.add_argument("--qk-scale", type=float, default=None)
    parser.add_argument("--drop-rate", type=float, default=0.0)
    parser.add_argument("--attn-drop-rate", type=float, default=0.0)
    parser.add_argument("--drop-path-rate", type=float, default=0.0)
    parser.add_argument("--init-values", type=float, default=0.0)
    parser.add_argument("--tubelet-size", type=int, default=2)
    parser.add_argument("--use-learnable-pos-emb", action="store_true")
    parser.add_argument("--with-cp", action="store_true")
    parser.add_argument("--all-frames", type=int, default=48)
    parser.add_argument("--cos-attn", action="store_true")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-d4rt", action="store_true")
    parser.add_argument("--num-queries", type=int, default=256)
    parser.add_argument("--img-patch-size", type=int, default=0)
    parser.add_argument("--align-corners", action="store_true")
    parser.add_argument("--decoder-depth", type=int, default=8)
    parser.add_argument("--decoder-num-heads", type=int, default=8)
    parser.add_argument("--embed-freqs", type=int, default=10)
    parser.add_argument("--embed-include-uv", action="store_true")
    parser.add_argument("--patch-mlp-ratio", type=float, default=4.0)
    parser.add_argument("--img-patch-sizes", type=int, nargs="+", default=[3, 6, 9, 12, 15])
    return parser.parse_args()


def print_stats(name, x):
    if not torch.is_tensor(x):
        return
    print(f"{name} shape:", tuple(x.shape))
    print(f"{name} dtype:", x.dtype)
    print(f"{name} device:", x.device)
    print(f"{name} min:", x.min().item())
    print(f"{name} max:", x.max().item())
    print(f"{name} mean:", x.mean().item())
    print(f"{name} std:", x.std().item())
    print(f"{name} has_nan:", torch.isnan(x).any().item())
    print(f"{name} has_inf:", torch.isinf(x).any().item())


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    images = torch.randn(
        args.batch_size,
        args.in_chans,
        args.all_frames,
        args.img_size,
        args.img_size,
        device=device,
    )
    ar = float(args.img_size) / float(args.img_size)
    ar=torch.tensor([ar], device=device)
    ar = ar.reshape(args.batch_size, 1)
    meta = {
        "aspect_ratio": ar,
        "img_patch_size": int(args.img_patch_size),
        "align_corners": bool(args.align_corners),
    }

    if args.run_d4rt:
        model = build_d4rt(args).to(device)
        model.eval()

        def encoder_hook(_module, _input, output):
            print_stats("encoder_out", output)

        def query_hook(_module, _input, output):
            print_stats("query_embed_out", output)

        def decoder_hook(_module, inputs, _output):
            if isinstance(inputs, tuple) and len(inputs) >= 2:
                print_stats("decoder_in_query", inputs[0])
                print_stats("decoder_in_global", inputs[1])

        h1 = model.encoder.register_forward_hook(encoder_hook)
        h2 = model.query_embed.register_forward_hook(query_hook)
        h3 = model.decoder.register_forward_hook(decoder_hook)

        uv = torch.rand(args.batch_size, args.num_queries, 2, device=device)
        t_src = torch.randint(0, args.all_frames, (args.batch_size, args.num_queries, 1), device=device)
        t_tgt = torch.randint(0, args.all_frames, (args.batch_size, args.num_queries, 1), device=device)
        t_cam = torch.randint(0, args.all_frames, (args.batch_size, args.num_queries, 1), device=device)
        query = torch.cat([uv, t_src, t_tgt, t_cam], dim=-1).to(dtype=images.dtype)

        with torch.no_grad():
            _ = model(meta, images, query)

        h1.remove()
        h2.remove()
        h3.remove()
    else:
        encoder = build_encoder(args).to(device)
        encoder.eval()
        with torch.no_grad():
            out = encoder(meta, images)
        print_stats("encoder_out", out)


if __name__ == "__main__":
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        pass
    main()
