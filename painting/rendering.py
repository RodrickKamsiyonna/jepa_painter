"""Generalized Gaussian stamps on quadratic Bezier curves.

Actions store p/max_p. Physical power is decoded only for pixel coverage;
training and latent planning operate on normalized action representations.
"""

import math

import torch

from .config import ADIM, P_INDEX


def gaussian_stamp(along, across, length, width, power):
    """Coverage with length/width as FWHM, evaluated without large powers.

    exp(p*log(distance)) is capped at 80 per axis. Beyond that, opacity is
    below float32 compositing precision. This avoids overflow at p=100 and
    retains finite position, width and power gradients.
    """

    def powered_distance(distance, extent):
        ratio = (2 * distance / extent.clamp_min(1e-7)).abs()
        log_ratio = ratio.clamp_min(torch.finfo(ratio.dtype).tiny).log()
        return (power * log_ratio).clamp(max=math.log(80.0)).exp()

    return (
        -math.log(2)
        * (powered_distance(along, length) + powered_distance(across, width))
    ).exp()


def coverage(a, height, width, cfg):
    """Union of rotated stamps, with an independently selected p per stroke."""
    if a.shape[-1] != ADIM:
        raise ValueError("Renderer expects 12-field strokes ending with p/max_p")
    # Pixel coverage uses float32 even if the latent planner uses autocast.
    a = a.float()
    b, n, _ = a.shape
    device, dtype = a.device, a.dtype
    ys = torch.arange(height, device=device, dtype=dtype)[None, None, :, None]
    xs = torch.arange(width, device=device, dtype=dtype)[None, None, None, :]
    p0, p1, p2 = a[..., :2], a[..., 2:4], a[..., 4:6]
    wp = a[..., 6].clamp_min(1e-7) * (width - 1)
    power = (a[..., P_INDEX] * cfg.max_p)[..., None, None]
    trans = torch.ones(b, n, height, width, device=device, dtype=dtype)
    for i in range(cfg.samples):
        t = i / max(1, cfg.samples - 1)
        u = 1 - t
        pos = u * u * p0 + 2 * u * t * p1 + t * t * p2
        vel = 2 * (u * (p1 - p0) + t * (p2 - p1))
        vx, vy = vel[..., 0] * (width - 1), vel[..., 1] * (height - 1)
        speed = (vx.square() + vy.square() + 1e-8).sqrt()
        tx = torch.where(speed > 1e-4, vx / speed, torch.ones_like(speed))
        ty = torch.where(speed > 1e-4, vy / speed, torch.zeros_like(speed))
        length = 2 * speed / max(1, cfg.samples - 1) + 0.75 * wp
        dx = xs - pos[..., 0, None, None] * (width - 1)
        dy = ys - pos[..., 1, None, None] * (height - 1)
        along = dx * tx[..., None, None] + dy * ty[..., None, None]
        across = -dx * ty[..., None, None] + dy * tx[..., None, None]
        stamp = gaussian_stamp(
            along, across, length[..., None, None], wp[..., None, None], power
        )
        trans = trans * (1 - stamp)
    return (1 - trans).clamp(0, 1)


def render(canvas, strokes, cfg, sequential=False):
    alpha = strokes[..., 10, None, None] * coverage(strokes, *canvas.shape[-2:], cfg)
    if sequential:
        out = canvas
        for i in range(strokes.shape[1]):
            mask = alpha[:, i : i + 1]
            out = out * (1 - mask) + strokes[:, i, 7:10, None, None] * mask
        return out
    reverse = torch.flip(torch.cumprod(torch.flip(1 - alpha, [1]), 1), [1])
    later = torch.cat((reverse[:, 1:], torch.ones_like(reverse[:, :1])), 1)
    return canvas * reverse[:, :1] + torch.einsum(
        "bnhw,bnc->bchw", alpha * later, strokes[..., 7:10]
    )


@torch.no_grad()
def render_stage(canvas, strokes, cfg):
    return torch.cat(
        [
            render(
                canvas[i : i + cfg.render_chunk], strokes[i : i + cfg.render_chunk], cfg
            )
            for i in range(0, len(canvas), cfg.render_chunk)
        ]
    )
