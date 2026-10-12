"""Train, evaluate held-out goals, or reconstruct an image at native resolution."""

import argparse
import json
from dataclasses import fields
from pathlib import Path

import torch
from PIL import Image, ImageOps
from torchvision.transforms.functional import to_tensor
from torchvision.utils import save_image

from .config import P_INDEX, STROKE_FIELDS, Config
from .planning import plan_image
from .training import dataset, evaluate, load_checkpoint, seed_all, train


def integers(value):
    return tuple(int(x) for x in value.split(","))


def parser():
    root = argparse.ArgumentParser(description=__doc__)
    sub = root.add_subparsers(dest="command", required=True)
    training = sub.add_parser("train")
    for field in fields(Config):
        if field.name == "plan_counts":
            training.add_argument("--plan_counts", type=integers, default=None)
        elif field.name == "grids":
            training.add_argument("--grids", type=integers, default=field.default)
        elif isinstance(field.default, bool):
            training.add_argument("--" + field.name, action="store_true")
        else:
            kwargs = {}
            if field.name == "plan_init":
                kwargs = {"choices": ["stratified", "random"]}
            training.add_argument(
                "--" + field.name,
                type=type(field.default),
                default=field.default,
                **kwargs,
            )
    training.add_argument("--data", default="./data")
    training.add_argument("--output", default="runs/experiment")
    training.add_argument("--resume", default="")
    training.add_argument("--fake", action="store_true")
    training.add_argument("--wandb", action="store_true")
    training.add_argument("--project", default="paint-world-model")
    for name, default in [
        ("epochs", 300),
        ("batch_size", 256),
        ("workers", 4),
        ("warmup", 500),
        ("plan_every", 10),
        ("plan_images", 3),
        ("log_every", 1),
    ]:
        training.add_argument("--" + name, type=int, default=default)
    for name, default in [("lr", 5e-4), ("wd", 5e-4), ("clip", 1.0)]:
        training.add_argument("--" + name, type=float, default=default)
    for command in ("evaluate", "plan"):
        p = sub.add_parser(command)
        p.add_argument("--checkpoint", required=True)
        p.add_argument("--output", default=f"outputs/{command}")
        p.add_argument(
            "--device", default="cuda" if torch.cuda.is_available() else "cpu"
        )
        p.add_argument("--plan_iters", type=int)
        p.add_argument("--plan_restarts", type=int)
        if command == "plan":
            p.add_argument("--image", required=True)
            p.add_argument("--tile_batch_size", type=int, default=4)
            p.add_argument("--no_video", action="store_true")
            p.add_argument("--video_fps", type=int, default=30)
            p.add_argument("--video_strokes_per_frame", type=int, default=8)
            p.add_argument("--video_scale", type=int, default=1)
        else:
            p.add_argument("--data", default="./data")
            p.add_argument("--fake", action="store_true")
            p.add_argument("--images", type=int, default=32)
            p.add_argument("--batch_size", type=int, default=4)
    return root


def main():
    args = parser().parse_args()
    if args.command == "train":
        for name in ("epochs", "batch_size", "plan_every", "plan_images", "log_every"):
            if getattr(args, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if args.workers < 0:
            raise ValueError("workers must be nonnegative")
        cfg = Config(
            **{field.name: getattr(args, field.name) for field in fields(Config)}
        )
        train(args, cfg)
        return
    device = torch.device(args.device)
    if args.command == "plan" and not args.no_video:
        from .video import require_video_writer

        for name in ("video_fps", "video_strokes_per_frame", "video_scale"):
            if getattr(args, name) <= 0:
                raise ValueError(f"{name} must be positive")
        require_video_writer()
    model, cfg, _ = load_checkpoint(args.checkpoint, device)
    for name in ("plan_iters", "plan_restarts"):
        if getattr(args, name) is not None:
            setattr(cfg, name, getattr(args, name))
    cfg.__post_init__()
    seed_all(cfg.seed)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2) + "\n")
    if args.command == "plan":
        with Image.open(args.image) as im:
            image = to_tensor(ImageOps.exif_transpose(im).convert("RGB")).to(device)
        final, stages, metrics, strokes, layout = plan_image(
            model, image, cfg, args.tile_batch_size
        )
        save_image(final, output / "painting.png")
        for i, stage in enumerate(stages):
            save_image(stage, output / f"stage_{i}.png")
        torch.save(
            {
                "strokes": strokes,
                "stroke_fields": STROKE_FIELDS,
                "physical_p": strokes[..., P_INDEX] * cfg.max_p,
                "layout": layout,
                "config": cfg.to_dict(),
                "coordinate_system": "tile-local normalized xy; row-major tiles",
            },
            output / "strokes.pt",
        )
        if not args.no_video:
            from .video import save_stroke_video

            save_stroke_video(
                output / "painting.mp4", strokes, layout, cfg,
                fps=args.video_fps,
                strokes_per_frame=args.video_strokes_per_frame,
                scale=args.video_scale,
            )
            print(f"Stroke replay saved to {output / 'painting.mp4'}", flush=True)
    else:
        if args.images <= 0 or args.batch_size <= 0:
            raise ValueError("images and batch_size must be positive")
        ds = dataset(args.data, False, args.fake, download=not args.fake)
        n = min(args.images, len(ds))
        sums = {}
        for start in range(0, n, args.batch_size):
            goals = torch.stack(
                [ds[i][0] for i in range(start, min(start + args.batch_size, n))]
            ).to(device)
            result = evaluate(model, goals, cfg, output, f"batch_{start:05d}")
            for key, value in result.items():
                sums[key] = sums.get(key, 0.0) + value * len(goals)
        metrics = {key: value / n for key, value in sums.items()}
        metrics["images"] = n
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))
