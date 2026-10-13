"""Stream the selected plan as strokes composited onto white image tiles."""

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .planning import join_tiles
from .rendering import render_stage


def require_video_writer():
    try:
        import imageio.v2 as imageio
        import imageio_ffmpeg  # noqa: F401
    except ImportError as exc:
        raise RuntimeError('Install video support with: pip install -e ".[video]"') from exc
    return imageio


@torch.no_grad()
def replay_frames(strokes, layout, cfg, strokes_per_frame=8):
    """Advance every tile together, preserving action and stroke order.

    Yield the entire cropped image, starting with white. Grouping strokes changes
    playback speed only; every planned stroke is still composited in order.
    Memory is bounded by the tile canvases and one group of stroke coverages.
    """
    if strokes_per_frame <= 0:
        raise ValueError("strokes_per_frame must be positive")
    if strokes.ndim != 4 or len(strokes) != layout[2] * layout[3]:
        raise ValueError("Expected tile x action x stroke x field data matching layout")
    flat = strokes.flatten(1, 2)
    canvas = torch.ones(len(strokes), 3, cfg.res, cfg.res, device=strokes.device)
    yield join_tiles(canvas, layout)
    for start in range(0, flat.shape[1], strokes_per_frame):
        canvas = render_stage(canvas, flat[:, start : start + strokes_per_frame], cfg)
        yield join_tiles(canvas, layout)


def save_stroke_video(path, strokes, layout, cfg, fps=30, strokes_per_frame=8, scale=1):
    """Encode an MP4 without retaining full-resolution animation frames in memory."""
    if fps <= 0 or scale <= 0 or strokes_per_frame <= 0:
        raise ValueError("Video fps, scale and strokes_per_frame must be positive")
    imageio = require_video_writer()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # yuv420p requires even dimensions. Pad the right/bottom edge instead of
    # resizing the painting; enable progressive playback before full download.
    with imageio.get_writer(
        str(path), fps=fps, codec="libx264", pixelformat="yuv420p", macro_block_size=1,
        output_params=["-movflags", "+faststart"],
    ) as writer:
        for frame in replay_frames(strokes, layout, cfg, strokes_per_frame):
            if scale != 1:
                frame = F.interpolate(frame[None], scale_factor=scale, mode="nearest")[0]
            pixels = frame.clamp(0, 1).mul(255).round().byte().permute(1, 2, 0).cpu().numpy()
            pixels = np.pad(
                pixels, ((0, pixels.shape[0] % 2), (0, pixels.shape[1] % 2), (0, 0)),
                mode="edge",
            )
            writer.append_data(np.ascontiguousarray(pixels))
        # Hold the rendered final painting for one second.
        for _ in range(fps):
            writer.append_data(pixels)
