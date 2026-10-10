"""Joint latent planning and independent tiled image reconstruction."""

import math
from itertools import pairwise

import torch
import torch.nn.functional as F

from .config import ADIM, SPB
from .model import compose
from .rendering import render_stage
from .strokes import make_strokes, paint_colors


def plan_layout(cfg, device):
    counts = list(cfg.plan_counts or [g * g for g in cfg.grids])
    lw = torch.cat([torch.full((n,), 1 / g) for n, g in zip(counts, cfg.grids)]).to(
        device
    )
    return counts, lw, (cfg.plan_window * lw).clamp(max=1.0)


def init_origins(counts, win, batch, mode, device):
    out = []
    offset = 0
    for n in counts:
        w = win[offset : offset + n].view(1, n, 1)
        offset += n
        if mode == "random":
            origin = torch.rand(batch, n, 2, device=device) * (1 - w)
        else:
            m = math.ceil(math.sqrt(n))
            idx = torch.arange(m * m, device=device)
            cell = torch.stack((idx % m, idx // m), -1).float()
            pick = torch.rand(batch, m * m, device=device).argsort(1)[:, :n]
            center = (cell[pick] + 0.5) / m + (
                torch.rand(batch, n, 2, device=device) - 0.5
            ) * 0.1 / m
            origin = torch.minimum((center - w / 2).clamp_min(0), 1 - w)
        out.append(origin)
    return torch.cat(out, 1)


def plan(net, goal, cfg):
    """Optimize all actions together, select restarts using final latent error."""
    if goal.ndim != 4 or goal.shape[1:] != (3, cfg.res, cfg.res):
        raise ValueError("plan expects B x 3 x 32 x 32 goals")
    was_training = net.training
    requires = [p.requires_grad for p in net.parameters()]
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    try:
        return _plan(net, goal, cfg)
    finally:
        for p, flag in zip(net.parameters(), requires):
            p.requires_grad_(flag)
        net.train(was_training)


def _plan(net, goal, cfg):
    b, device, r = len(goal), goal.device, cfg.plan_restarts
    counts, lw, win = plan_layout(cfg, device)
    a = sum(counts)
    bounds = [0]
    for count in counts:
        bounds.append(bounds[-1] + count)
    spans = list(pairwise(bounds))
    ends = [end - 1 for end in bounds[1:]]
    targets = goal.repeat_interleave(r, 0)
    br = b * r
    w3, l3 = win[None, :, None], lw[None, :, None]
    with torch.no_grad():
        zg = net.encode(targets).float()
        z0 = net.encode(torch.ones_like(goal[:1]))[0].float()
        u = torch.rand(br, a, SPB, ADIM, device=device)
        origin = init_origins(counts, win, br, cfg.plan_init, device)
        st = make_strokes(u, origin.unsqueeze(2), w3, l3, cfg)
        col = torch.cat(
            [
                paint_colors(targets, st[:, start:end], g, cfg)
                for (start, end), g in zip(spans, cfg.grids)
            ],
            1,
        )
        u = torch.cat((u[..., :7], col, u[..., 10:]), -1)
        fraction = (origin / (1 - win).clamp_min(1e-6)[None, :, None]).clamp(
            1e-3, 1 - 1e-3
        )
    ru = torch.nn.Parameter(torch.logit(u.clamp(1e-3, 1 - 1e-3)))
    ro = torch.nn.Parameter(torch.logit(fraction))

    def decode():
        origin = (ro.sigmoid() * (1 - win)[None, :, None]).unsqueeze(2)
        return make_strokes(ru.sigmoid(), origin, w3, l3, cfg)

    def actions(st):
        use_amp = cfg.plan_amp and device.type == "cuda"
        dtype = (
            torch.bfloat16
            if use_amp and torch.cuda.is_bf16_supported()
            else torch.float16
        )
        with torch.autocast(device.type, dtype=dtype, enabled=use_amp):
            return net.actions(st)

    opt = torch.optim.Adam((ru, ro), lr=cfg.plan_lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, cfg.plan_iters, eta_min=cfg.plan_lr * 0.05
    )
    with torch.enable_grad():
        for _ in range(cfg.plan_iters):
            s, xa = actions(decode())
            pred = compose(z0, s, xa, [a - 1])[0]
            loss = (pred - zg).square().mean(-1).sum()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            scheduler.step()
    with torch.no_grad():
        st = decode()
        s, xa = actions(st)
        zs = compose(z0, s, xa, ends)
        best = (zs[-1] - zg).square().mean(-1).view(b, r).argmin(1)
        pick = torch.arange(b, device=device) * r + best
        st, zs, zg = st[pick], zs[:, pick], zg[pick]
        canvas = torch.ones_like(goal)
        snaps = [canvas]
        info = {}
        for k, (start, end) in enumerate(spans):
            canvas = render_stage(canvas, st[:, start:end].flatten(1, 2), cfg)
            snaps.append(canvas)
            encoded = net.encode(canvas).float()
            info[f"latent_loss_s{k + 1}"] = (zs[k] - zg).square().mean().item()
            info[f"latent_gap_s{k + 1}"] = (encoded - zs[k]).square().mean().item()
            info[f"pixel_mse_s{k + 1}"] = F.mse_loss(canvas, goal).item()
        info["final_latent_loss"] = info[f"latent_loss_s{len(spans)}"]
        info["final_latent_gap"] = info[f"latent_gap_s{len(spans)}"]
        info["final_pixel_mse"] = info[f"pixel_mse_s{len(spans)}"]
    return canvas, snaps, info, st


def split_tiles(image, size=32):
    """Row-major tiles, replicate padding on right and bottom only."""
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError("Expected an RGB tensor of shape 3 x H x W")
    h, w = image.shape[-2:]
    padded = F.pad(image[None], (0, (-w) % size, 0, (-h) % size), mode="replicate")[0]
    rows, cols = padded.shape[1] // size, padded.shape[2] // size
    tiles = (
        padded.view(3, rows, size, cols, size)
        .permute(1, 3, 0, 2, 4)
        .reshape(rows * cols, 3, size, size)
    )
    return tiles, (h, w, rows, cols)


def join_tiles(tiles, layout):
    h, w, rows, cols = layout
    size = tiles.shape[-1]
    return (
        tiles.reshape(rows, cols, 3, size, size)
        .permute(2, 0, 3, 1, 4)
        .reshape(3, rows * size, cols * size)[:, :h, :w]
    )


def plan_image(net, image, cfg, tile_batch_size=4):
    if tile_batch_size <= 0:
        raise ValueError("tile_batch_size must be positive")
    tiles, layout = split_tiles(image, cfg.res)
    stage_batches, stroke_batches = [], []
    totals = {}
    for start in range(0, len(tiles), tile_batch_size):
        batch = tiles[start : start + tile_batch_size]
        _, snaps, info, strokes = plan(net, batch, cfg)
        for key, value in info.items():
            totals[key] = totals.get(key, 0.0) + value * len(batch)
        stage_batches.append(torch.stack(snaps).cpu())
        stroke_batches.append(strokes.cpu())
    stages = torch.cat(stage_batches, 1)
    images = [join_tiles(stage, layout) for stage in stages]
    mse = F.mse_loss(images[-1], image.cpu()).item()
    return (
        images[-1],
        images,
        {"tile_mean_" + key: value / len(tiles) for key, value in totals.items()}
        | {"pixel_mse": mse, "tiles": len(tiles)},
        torch.cat(stroke_batches),
        layout,
    )
