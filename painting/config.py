"""Serializable experiment configuration; checkpoints are the source of truth."""

import math
from dataclasses import asdict, dataclass

SPB, ADIM = 16, 12
P_INDEX = 11
STROKE_FIELDS = (
    "x0",
    "y0",
    "x1",
    "y1",
    "x2",
    "y2",
    "width",
    "r",
    "g",
    "b",
    "alpha",
    "p_normalized",
)


@dataclass
class Config:
    grids: tuple = (1, 2, 4, 8)
    res: int = 32
    max_p: float = 100.0
    min_w_px: float = 0.5
    width_scale: float = 1.0
    alpha_min: float = 0.35
    samples: int = 8
    render_chunk: int = 8
    color_blur: float = 0.35
    color_jitter: float = 0.02
    zdim: int = 512
    act_dim: int = 128
    act_layers: int = 4
    sigreg_dirs: int = 1024
    sigreg_w: float = 0.009
    ssl_w: float = 1.0
    seed: int = 42
    plan_counts: tuple | None = None
    plan_window: float = 1.0
    plan_init: str = "stratified"
    plan_restarts: int = 1
    plan_iters: int = 300
    plan_lr: float = 0.01
    plan_amp: bool = False

    def __post_init__(self):
        if self.res != 32:
            raise ValueError(
                "The learned canvas resolution is 32; use tiled planning for larger images."
            )
        if not math.isfinite(self.max_p) or not 1 <= self.max_p <= 1e6:
            raise ValueError("max_p must be finite and between 1 and 1,000,000")
        for name in (
            "min_w_px",
            "width_scale",
            "render_chunk",
            "samples",
            "plan_iters",
            "plan_restarts",
            "plan_window",
            "plan_lr",
            "zdim",
            "act_dim",
            "act_layers",
            "sigreg_dirs",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if not self.grids or any(g <= 0 for g in self.grids):
            raise ValueError("grids must contain positive integers")
        if not 0 <= self.alpha_min <= 1:
            raise ValueError("alpha_min must be in [0, 1]")
        if self.act_dim % 4:
            raise ValueError("act_dim must be divisible by four attention heads")
        if self.plan_init not in ("random", "stratified"):
            raise ValueError("plan_init must be random or stratified")
        if min(self.ssl_w, self.sigreg_w, self.color_blur, self.color_jitter) < 0:
            raise ValueError("Loss weights, blur and jitter must be nonnegative")
        if self.plan_counts is not None and (
            len(self.plan_counts) != len(self.grids)
            or any(n <= 0 for n in self.plan_counts)
        ):
            raise ValueError("plan_counts must have one positive count per stage")

    def to_dict(self):
        return asdict(self)
