"""Inspect random per-stroke powers on one local CIFAR-10 example; no downloads."""

import argparse
import json
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from torchvision.transforms.functional import to_pil_image, to_tensor

from painting.config import ADIM, P_INDEX, SPB, STROKE_FIELDS, Config
from painting.rendering import coverage, render, render_stage
from painting.strokes import make_trajectory


def draw_frames(rows, output):
    size, label = 128, 140
    comparison = Image.new("RGB", (label + 6 * size, 36 + len(rows) * 164), "white")
    draw = ImageDraw.Draw(comparison)
    headings = ["Target", "Blank", "1x1: 16", "2x2: 80", "4x4: 336", "8x8: 1360"]
    for i, heading in enumerate(headings):
        draw.text((label + i * size + 4, 10), heading, fill="black")
    for row, (name, frames) in enumerate(rows):
        y = 36 + row * 164
        draw.text((5, y + 54), name, fill="black")
        for col, frame in enumerate(frames):
            comparison.paste(
                to_pil_image(frame[0]).resize((size, size), Image.Resampling.NEAREST),
                (label + col * size, y),
            )
    comparison.save(output / "comparison.png")


@torch.no_grad()
def check(image, cfg, output):
    torch.manual_seed(cfg.seed)
    strokes, canvases = make_trajectory(image, cfg)
    assert strokes.shape == (1, 85, SPB, ADIM)
    assert torch.isfinite(strokes).all() and torch.isfinite(canvases).all()
    assert (strokes[..., :6] >= 0).all() and (strokes[..., :6] <= 1).all()
    assert (strokes[..., 7:11] >= 0).all() and (strokes[..., 7:11] <= 1).all()
    assert strokes[..., 10].min() >= cfg.alpha_min
    assert (strokes[..., 6] * 31).min() >= cfg.min_w_px - 1e-6
    physical = strokes[..., P_INDEX] * cfg.max_p
    assert physical.min() >= 1 and physical.max() <= cfg.max_p
    assert canvases.min() >= 0 and canvases.max() <= 1
    report = {
        "config": cfg.to_dict(),
        "stroke_fields": STROKE_FIELDS,
        "physical_p_min": physical.min().item(),
        "physical_p_max": physical.max().item(),
        "physical_p_mean": physical.mean().item(),
        "stages": [],
    }
    blank = torch.ones_like(image)
    previous = blank
    offset = 0
    for k, g in enumerate(cfg.grids):
        region = strokes[:, offset : offset + g * g]
        idx = torch.arange(g * g)
        origin = torch.stack((idx % g, idx // g), -1) / g
        local = (
            region[..., :6].reshape(1, g * g, SPB, 3, 2) - origin[None, :, None, None]
        ) * g
        assert local.min() >= 0.05 - 1e-5 and local.max() <= 0.95 + 1e-5
        flat = region.flatten(1, 2)
        mask = coverage(flat, 32, 32, cfg)
        seq = previous.clone()
        for i in range(len(flat[0])):
            alpha = flat[:, i, 10, None, None, None] * mask[:, i : i + 1]
            seq = seq * (1 - alpha) + flat[:, i, 7:10, None, None] * alpha
        error = (seq - canvases[k]).abs().max().item()
        assert error < 1e-5
        offset += g * g
        report["stages"].append(
            {
                "grid": g,
                "regions": g * g,
                "strokes_per_region": SPB,
                "stage_strokes": g * g * SPB,
                "cumulative_strokes": offset * SPB,
                "pixel_mse": (canvases[k] - image).square().mean().item(),
                "zero_coverage_strokes": int((mask.flatten(2).sum(-1) == 0).sum()),
                "compositing_max_error": error,
            }
        )
        previous = canvases[k]
    rows = [(f"Random p: 1-{cfg.max_p:g}", [image, blank] + list(canvases))]
    comparison_metrics = {}
    for fixed_p in [p for p in (2.0, 10.0, 100.0) if p <= cfg.max_p]:
        fixed = strokes.clone()
        fixed[..., P_INDEX] = fixed_p / cfg.max_p
        canvas = blank
        frames = [image, blank]
        offset = 0
        for g in cfg.grids:
            canvas = render_stage(
                canvas, fixed[:, offset : offset + g * g].flatten(1, 2), cfg
            )
            frames.append(canvas)
            offset += g * g
        comparison_metrics[f"p={fixed_p:g}"] = (canvas - image).square().mean().item()
        rows.append((f"Fixed p={fixed_p:g}", frames))
    draw_frames(rows, output)
    sheet = Image.new("RGB", (4 * 128, 4 * 152), "white")
    draw = ImageDraw.Draw(sheet)
    for i in range(SPB):
        x, y = (i % 4) * 128, (i // 4) * 152
        stamp = render(blank, strokes[:, 0, i : i + 1], cfg)
        sheet.paste(
            to_pil_image(stamp[0]).resize((128, 128), Image.Resampling.NEAREST),
            (x, y + 24),
        )
        draw.text((x + 4, y + 5), f"#{i + 1} p={physical[0, 0, i]:.2f}", fill="black")
    sheet.save(output / "random_16_strokes.png")
    torch.save(
        {
            "config": cfg.to_dict(),
            "stroke_fields": STROKE_FIELDS,
            "strokes": strokes,
            "physical_p": physical,
            "canvases": canvases,
            "target": image,
        },
        output / "trajectory.pt",
    )
    report["fixed_power_comparison_mse"] = comparison_metrics
    report["all_checks_passed"] = True
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default="examples/cifar10_frog.png")
    parser.add_argument("--output", default="outputs/data-generation")
    parser.add_argument("--max_p", type=float, default=100.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    cfg = Config(max_p=args.max_p, seed=args.seed)
    torch.set_num_threads(2)
    with Image.open(args.image) as im:
        if im.size != (32, 32):
            raise ValueError("Supply a native 32x32 image")
        image = to_tensor(im.convert("RGB"))[None]
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    print(json.dumps(check(image, cfg, output), indent=2))


if __name__ == "__main__":
    main()
