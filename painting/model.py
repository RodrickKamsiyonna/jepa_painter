"""Encoder, ordered action encoder, latent composition and training losses."""

import math

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.distributed.nn.functional import all_reduce
from torchvision.models import resnet18

from .config import ADIM, SPB


def compose(z0, s, xa, ends):
    s, xa = s.float(), xa.float()
    cumulative = F.logsigmoid(xa).cumsum(1)
    injection = torch.sigmoid(-xa) * s
    return torch.stack(
        [
            torch.exp(cumulative[:, e]) * z0
            + (
                torch.exp(cumulative[:, e : e + 1] - cumulative[:, : e + 1])
                * injection[:, : e + 1]
            ).sum(1)
            for e in ends
        ]
    )


class Encoder(nn.Module):
    def __init__(self, zdim):
        super().__init__()
        r = resnet18(weights=None, norm_layer=lambda c: nn.GroupNorm(8, c))
        r.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        r.maxpool = nn.Identity()
        r.fc = nn.Identity()
        self.net = r
        self.proj = nn.Sequential(
            nn.Linear(512, 1024), nn.GELU(), nn.Linear(1024, zdim)
        )
        self.register_buffer(
            "mean", torch.tensor((0.4914, 0.4822, 0.4465)).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor((0.2470, 0.2435, 0.2616)).view(1, 3, 1, 1)
        )

    def forward(self, x):
        return self.proj(self.net((x - self.mean) / self.std))


class ActionEncoder(nn.Module):
    def __init__(self, zdim, d, layers):
        super().__init__()
        self.register_buffer("freq", math.pi * 2.0 ** torch.arange(5))
        self.inp = nn.Sequential(nn.Linear(ADIM * 11, d), nn.GELU(), nn.Linear(d, d))
        self.slot = nn.Parameter(torch.randn(1, SPB, d) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d, 4, 4 * d, 0.0, "gelu", batch_first=True, norm_first=True
        )
        self.tr = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(SPB * d, 2 * zdim)
        with torch.no_grad():
            self.head.bias[zdim:].fill_(4.0)

    def forward(self, x):
        if x.shape[-1] != ADIM:
            raise ValueError("Actions must have 12 fields, ending with p/max_p")
        f = x[..., None] * self.freq
        feat = torch.cat((x, f.sin().flatten(-2), f.cos().flatten(-2)), -1)
        return self.head(
            self.norm(self.tr(self.inp(feat) + self.slot)).flatten(1)
        ).chunk(2, -1)


class SigReg(nn.Module):
    def __init__(self, dim, n_dir, seed):
        super().__init__()
        t = torch.linspace(0, 3, 17)
        weights = torch.full((17,), 2 * 3 / 16)
        weights[0] = weights[-1] = 3 / 16
        phi = torch.exp(-t.square() / 2)
        self.register_buffer("t", t)
        self.register_buffer("phi", phi)
        self.register_buffer("w", weights * phi)
        self.dim, self.n_dir, self.seed = dim, n_dir, seed

    def forward(self, z, step):
        z = z.float()
        world = 1
        gen = torch.Generator(device=z.device).manual_seed(self.seed * 1_000_003 + step)
        a = torch.randn(self.dim, self.n_dir, generator=gen, device=z.device)
        a = a / a.norm(dim=0, keepdim=True)
        x = (z @ a).unsqueeze(-1) * self.t
        c, s = x.cos().mean(0), x.sin().mean(0)
        if dist.is_initialized():
            world = dist.get_world_size()
            c, s = all_reduce(c) / world, all_reduce(s) / world
        return (((c - self.phi).square() + s.square()) @ self.w).mean() * (
            len(z) * world
        )


class WorldModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.enc = Encoder(cfg.zdim)
        self.act = ActionEncoder(cfg.zdim, cfg.act_dim, cfg.act_layers)
        self.sig = SigReg(cfg.zdim, cfg.sigreg_dirs, cfg.seed)
        self.ends = [int(e) for e in np.cumsum([g * g for g in cfg.grids]) - 1]

    def encode(self, x):
        return self.enc(x)

    def actions(self, strokes):
        b, a = strokes.shape[:2]
        s, xa = self.act(strokes.flatten(0, 1))
        return s.view(b, a, -1), xa.view(b, a, -1)

    def forward(self, v1, v2, canvases, strokes, step=0):
        k, b = canvases.shape[:2]
        white = torch.ones_like(canvases[0, :1])
        z = self.enc(torch.cat((white, canvases.flatten(0, 1), v1, v2))).float()
        z0, zc = z[0], z[1 : 1 + k * b].view(k, b, -1)
        z1, z2 = z[1 + k * b :].chunk(2, 0)
        s, xa = self.actions(strokes)
        zp = compose(z0, s, xa, self.ends)
        stage = (zp - zc).square().mean((1, 2))
        dyn = stage.mean()
        ssl = (z1 - z2).square().mean()
        zr = torch.cat((zc.flatten(0, 1), z1, z2))
        sig = self.sig(zr, step)
        loss = dyn + self.cfg.ssl_w * ssl + self.cfg.sigreg_w * sig
        with torch.no_grad():
            prev = torch.cat((z0.view(1, 1, -1).expand(1, b, -1), zc[:-1]))
            baseline = (prev - zc).square().mean((1, 2))
            stats = torch.cat(
                (torch.stack((loss, dyn, ssl, sig, zr.std(0).mean())), stage, baseline)
            )
        return loss, stats
