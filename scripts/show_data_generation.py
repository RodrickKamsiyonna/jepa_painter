"""Create shareable figures of a fixed-power random painting trajectory."""

import argparse
import json
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms.functional import to_pil_image, to_tensor

from painting.config import P_INDEX, SPB, Config
from painting.rendering import render, render_stage
from painting.strokes import make_trajectory


def font(size):
    for name in ("/System/Library/Fonts/Supplemental/Arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default(size=size)


def panel(frames, labels, details, title, subtitle, columns, output):
    cell, margin, top = 256, 28, 138
    rows = (len(frames) + columns - 1) // columns
    sheet = Image.new(
        "RGB", (margin * 2 + columns * cell, top + rows * 310 + 66), "#f5f6f8"
    )
    draw = ImageDraw.Draw(sheet)
    draw.text((margin, 24), title, font=font(30), fill="#162238")
    draw.text((margin, 73), subtitle, font=font(19), fill="#48566b")
    for i, frame in enumerate(frames):
        x, y = margin + (i % columns) * cell, top + (i // columns) * 310
        sheet.paste(
            to_pil_image(frame).resize((224, 224), Image.Resampling.NEAREST), (x, y)
        )
        draw.text((x, y + 237), labels[i], font=font(20), fill="#162238")
        draw.text((x, y + 267), details[i], font=font(15), fill="#48566b")
    draw.text(
        (margin, sheet.height - 42),
        "Random strokes + colors sampled from target. Data generation, not model planning.",
        font=font(17),
        fill="#48566b",
    )
    sheet.save(output)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default="examples/cifar10_automobile.jpg")
    parser.add_argument("--class_name", default="automobile")
    parser.add_argument("--power", type=float, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="outputs/cifar10-automobile-p16")
    parser.add_argument("--video", action="store_true")
    args = parser.parse_args()
    cfg = Config(seed=args.seed)
    if not 1 <= args.power <= cfg.max_p:
        raise ValueError("power must be between 1 and max_p")
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    with Image.open(args.image) as source:
        if source.size != (32, 32):
            raise ValueError("Expected a native 32x32 image")
        goal = to_tensor(source.convert("RGB"))[None]
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    strokes, _ = make_trajectory(goal, cfg)
    strokes[..., P_INDEX] = args.power / cfg.max_p
    canvas = torch.ones_like(goal)
    frames, counts, metrics = [canvas[0]], [0], []
    offset = 0
    for grid in cfg.grids:
        flat = strokes[:, offset : offset + grid * grid].flatten(1, 2)
        sequential = render(canvas, flat, cfg, sequential=True)
        canvas = render_stage(canvas, flat, cfg)
        torch.testing.assert_close(canvas, sequential, atol=1e-6, rtol=1e-6)
        assert torch.isfinite(canvas).all()
        offset += grid * grid
        frames.append(canvas[0])
        counts.append(offset * SPB)
        metrics.append(
            {
                "grid": grid,
                "new_strokes": grid * grid * SPB,
                "cumulative_strokes": offset * SPB,
                "pixel_mse": (canvas - goal).square().mean().item(),
            }
        )
    panel(
        [goal[0]] + frames,
        [
            "Target",
            "White canvas",
            "1 x 1 grid",
            "2 x 2 grid",
            "4 x 4 grid",
            "8 x 8 grid",
        ],
        ["CIFAR-10: " + args.class_name] + [f"{n:,} strokes total" for n in counts],
        "How a painting trajectory is generated",
        f"32 x 32 image  |  brush power p = {args.power:g}  |  16 strokes per region  |  coarse to fine",
        3,
        output / "process.png",
    )
    first = torch.ones_like(goal)
    progression = []
    for i in range(SPB):
        first = render(first, strokes[:, 0, i : i + 1], cfg)
        progression.append(first[0])
    panel(
        progression,
        [f"Stroke {i + 1}" for i in range(SPB)],
        ["First stage: 1 region"] * SPB,
        "The first 16 strokes, one at a time",
        f"CIFAR-10 {args.class_name}  |  p = {args.power:g}  |  starting from white",
        4,
        output / "first_16_strokes.png",
    )
    for i, frame in enumerate([goal[0]] + frames):
        to_pil_image(frame).save(output / f"frame_{i}.png")
    report = {
        "class": args.class_name,
        "power": args.power,
        "seed": args.seed,
        "config": cfg.to_dict(),
        "stages": metrics,
        "checks_passed": True,
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    torch.save(
        {
            "target": goal,
            "strokes": strokes,
            "frames": torch.stack(frames),
            "config": cfg.to_dict(),
        },
        output / "trajectory.pt",
    )
    if args.video:
        from painting.video import save_stroke_video

        save_stroke_video(
            output / "drawing.mp4", strokes, (32, 32, 1, 1), cfg, scale=16
        )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
