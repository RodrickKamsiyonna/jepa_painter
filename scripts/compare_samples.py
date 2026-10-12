"""Compare 8/16 stamps per curve at p=10/100 on identical CIFAR-10 strokes."""

import argparse
import json
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from torchvision.transforms.functional import to_pil_image, to_tensor

from painting.config import P_INDEX, SPB, Config
from painting.rendering import coverage, render_stage
from painting.strokes import make_trajectory


@torch.no_grad()
def compare(image, output, seed):
    torch.manual_seed(seed)
    base, _ = make_trajectory(image, Config(seed=seed))
    blank = torch.ones_like(image)
    records, frames = [], []
    for power in (10, 100):
        for samples in (8, 16):
            cfg = Config(samples=samples, seed=seed)
            strokes = base.clone()
            strokes[..., P_INDEX] = power / cfg.max_p
            assert torch.equal(strokes[..., :P_INDEX], base[..., :P_INDEX])
            canvas = blank
            snapshots, stages = [], []
            offset = 0
            for g in cfg.grids:
                flat = strokes[:, offset : offset + g * g].flatten(1, 2)
                rendered = render_stage(canvas, flat, cfg)
                mask = coverage(flat, 32, 32, cfg)
                sequential = canvas.clone()
                for i in range(flat.shape[1]):
                    alpha = flat[:, i, 10, None, None, None] * mask[:, i : i + 1]
                    sequential = (
                        sequential * (1 - alpha) + flat[:, i, 7:10, None, None] * alpha
                    )
                error = (rendered - sequential).abs().max().item()
                assert error < 1e-5 and torch.isfinite(rendered).all()
                assert rendered.min() >= 0 and rendered.max() <= 1
                offset += g * g
                stages.append(
                    {
                        "grid": g,
                        "cumulative_strokes": offset * SPB,
                        "pixel_mse": (rendered - image).square().mean().item(),
                        "zero_coverage_strokes": int(
                            (mask.flatten(2).sum(-1) == 0).sum()
                        ),
                        "compositing_max_error": error,
                    }
                )
                canvas = rendered
                snapshots.append(canvas)
            record = {
                "power": power,
                "samples": samples,
                "stages": stages,
                "final_pixel_mse": stages[-1]["pixel_mse"],
            }
            records.append(record)
            frames.append((record, snapshots))
            print(
                f"p={power}, samples={samples}: MSE={record['final_pixel_mse']:.8f}",
                flush=True,
            )

    size, label = 144, 140
    sheet = Image.new("RGB", (label + 5 * size, 32 + 4 * 176), "white")
    draw = ImageDraw.Draw(sheet)
    for i, name in enumerate(("Target", "1x1: 16", "2x2: 80", "4x4: 336", "8x8: 1360")):
        draw.text((label + i * size + 4, 9), name, fill="black")
    final = Image.new("RGB", (3 * 224, 2 * 256), "white")
    final_draw = ImageDraw.Draw(final)
    for row in range(2):
        final_draw.text((5, row * 256 + 7), "Target: CIFAR-10 frog", fill="black")
        final.paste(
            to_pil_image(image[0]).resize((224, 224), Image.Resampling.NEAREST),
            (0, row * 256 + 32),
        )
    for row, (record, snapshots) in enumerate(frames):
        y = 32 + row * 176
        draw.text(
            (4, y + 55),
            f"p={record['power']}, samples={record['samples']}",
            fill="black",
        )
        for col, frame in enumerate([image] + snapshots):
            sheet.paste(
                to_pil_image(frame[0]).resize((size, size), Image.Resampling.NEAREST),
                (label + col * size, y),
            )
        x = (1 + row % 2) * 224
        y = (row // 2) * 256
        final_draw.text(
            (x + 5, y + 5),
            f"p={record['power']}, samples={record['samples']}",
            fill="black",
        )
        final_draw.text(
            (x + 5, y + 18), f"MSE={record['final_pixel_mse']:.6f}", fill="black"
        )
        final.paste(
            to_pil_image(snapshots[-1][0]).resize((224, 224), Image.Resampling.NEAREST),
            (x, y + 32),
        )
    sheet.save(output / "stages.png")
    final.save(output / "finals.png")
    report = {
        "seed": seed,
        "identical_geometry_colors_width_opacity": True,
        "strokes_per_region": SPB,
        "total_strokes": 1360,
        "results": records,
        "all_checks_passed": True,
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    torch.save(
        {
            "target": image,
            "base_strokes": base,
            "results": [
                {
                    "power": r["power"],
                    "samples": r["samples"],
                    "canvases": torch.stack(s),
                }
                for r, s in frames
            ],
        },
        output / "comparison.pt",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default="examples/cifar10_frog.png")
    parser.add_argument("--output", default="outputs/samples-comparison")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    torch.set_num_threads(2)
    with Image.open(args.image) as image:
        if image.size != (32, 32):
            raise ValueError("Expected one native 32x32 image")
        goal = to_tensor(image.convert("RGB"))[None]
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    compare(goal, output, args.seed)


if __name__ == "__main__":
    main()
