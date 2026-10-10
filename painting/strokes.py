"""Original stroke distribution and target-guided color selection."""

import math

import torch
import torch.nn.functional as F

from .config import ADIM, P_INDEX, SPB
from .rendering import render_stage


def make_strokes(u, origin, win, lw, cfg):
    p0, p2 = 0.1 + 0.8 * u[..., :2], 0.1 + 0.8 * u[..., 4:6]
    p1 = ((p0 + p2) / 2 + 0.3 * (2 * u[..., 2:4] - 1)).clamp(0.05, 0.95)

    def glob(p):
        return origin + p * win[..., None]

    width = ((0.24 + 0.12 * u[..., 6:7]) * cfg.width_scale * lw[..., None]).clamp_min(
        cfg.min_w_px / (cfg.res - 1)
    )
    alpha = cfg.alpha_min + (1 - cfg.alpha_min) * u[..., 10:11]
    # Sample physical p in [1, max_p]; only p/max_p enters the action encoder.
    p_normalized = (1 + (cfg.max_p - 1) * u[..., P_INDEX : P_INDEX + 1]) / cfg.max_p
    return torch.cat(
        (glob(p0), glob(p1), glob(p2), width, u[..., 7:10], alpha, p_normalized), -1
    )


def to_strokes(u, g, cfg):
    idx = torch.arange(u.shape[1], device=u.device)
    origin = (torch.stack((idx % g, idx // g), -1).to(u.dtype) / g).view(1, -1, 1, 2)
    side = torch.full((1, u.shape[1], 1), 1 / g, device=u.device, dtype=u.dtype)
    return make_strokes(u, origin, side, side, cfg)


def gblur(x, sigma):
    if sigma < 0.05:
        return x
    r = math.ceil(3 * sigma)
    k = torch.arange(-r, r + 1, device=x.device, dtype=x.dtype)
    k = torch.exp(-k.square() / (2 * sigma**2))
    k = k / k.sum()
    c = x.shape[1]
    x = F.pad(x, (r, r, r, r), mode="replicate")
    x = F.conv2d(x, k.view(1, 1, 1, -1).repeat(c, 1, 1, 1), groups=c)
    return F.conv2d(x, k.view(1, 1, -1, 1).repeat(c, 1, 1, 1), groups=c)


def paint_colors(img, st, g, cfg, jitter=0.0):
    b, n, s, _ = st.shape
    pm = 0.25 * st[..., :2] + 0.5 * st[..., 2:4] + 0.25 * st[..., 4:6]
    blurred = gblur(
        img,
        cfg.color_blur * max(0.30 / g, cfg.min_w_px / (cfg.res - 1)) * (cfg.res - 1),
    )
    col = F.grid_sample(
        blurred,
        (pm * 2 - 1).reshape(b, n * s, 1, 2),
        padding_mode="border",
        align_corners=True,
    )
    col = col[..., 0].permute(0, 2, 1).reshape(b, n, s, 3)
    if jitter:
        col = col + jitter * torch.randn_like(col)
    return col.clamp(0, 1)


@torch.no_grad()
def make_trajectory(img, cfg):
    canvas = torch.ones_like(img)
    strokes, canvases = [], []
    for g in cfg.grids:
        u = torch.rand(len(img), g * g, SPB, ADIM, device=img.device)
        st = to_strokes(u, g, cfg)
        st = torch.cat(
            (
                st[..., :7],
                paint_colors(img, st, g, cfg, cfg.color_jitter),
                st[..., 10:],
            ),
            -1,
        )
        canvas = render_stage(canvas, st.flatten(1, 2), cfg)
        strokes.append(st)
        canvases.append(canvas)
    return torch.cat(strokes, 1), torch.stack(canvases)
