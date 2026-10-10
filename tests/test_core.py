import math

import pytest
import torch

from painting.config import ADIM, P_INDEX, SPB, Config
from painting.model import WorldModel, compose
from painting.planning import join_tiles, plan, plan_image, split_tiles
from painting.rendering import coverage, gaussian_stamp, render
from painting.strokes import make_trajectory, to_strokes


@pytest.mark.parametrize("shape", [(32, 32), (512, 512), (45, 67), (7, 11)])
def test_tile_roundtrip(shape):
    image = torch.rand(3, *shape)
    tiles, layout = split_tiles(image)
    assert torch.equal(join_tiles(tiles, layout), image)
    assert len(tiles) == ((shape[0] + 31) // 32) * ((shape[1] + 31) // 32)


def test_composition():
    torch.manual_seed(0)
    s, xa, z0 = torch.randn(3, 85, 16), torch.randn(3, 85, 16) * 2 + 2, torch.randn(16)
    ends = [0, 4, 20, 84]
    expected = []
    z = z0.expand(3, 16)
    for j in range(85):
        alpha = xa[:, j].sigmoid()
        z = alpha * z + (1 - alpha) * s[:, j]
        if j in ends:
            expected.append(z)
    torch.testing.assert_close(
        compose(z0, s, xa, ends), torch.stack(expected), atol=1e-5, rtol=1e-4
    )


@pytest.mark.parametrize("max_p", [1.0, 10.0, 100.0])
def test_power_sampling_and_bounds(max_p):
    cfg = Config(max_p=max_p)
    u = torch.rand(2, 4, SPB, ADIM)
    u[:, :, 0, P_INDEX] = 0
    u[:, :, 1, P_INDEX] = 1
    st = to_strokes(u, 2, cfg)
    physical = st[..., P_INDEX] * max_p
    torch.testing.assert_close(physical, 1 + (max_p - 1) * u[..., P_INDEX])
    assert physical.min() >= 1 and physical.max() <= max_p
    assert st[..., P_INDEX].min() >= 1 / max_p and st[..., P_INDEX].max() <= 1
    assert st.shape[-1] == 12


@pytest.mark.parametrize("power", [1.0, 2.0, 4.0, 10.0, 100.0])
def test_renderer(power):
    cfg = Config()
    st = to_strokes(torch.rand(2, 4, SPB, ADIM), 2, cfg).flatten(1, 2)
    st[..., P_INDEX] = power / cfg.max_p
    canvas = torch.rand(2, 3, 32, 32)
    actual = render(canvas, st, cfg)
    torch.testing.assert_close(
        actual, render(canvas, st, cfg, sequential=True), atol=1e-6, rtol=1e-5
    )
    assert torch.isfinite(actual).all() and actual.min() >= 0 and actual.max() <= 1


@pytest.mark.parametrize("power", [1.0, 2.0, 3.0, 4.0, 10.0, 100.0])
def test_generalized_gaussian_formula(power):
    along = torch.linspace(-3, 3, 31)
    across = torch.linspace(-2, 2, 31)
    length, width = torch.tensor(2.0), torch.tensor(1.0)
    actual = gaussian_stamp(along, across, length, width, torch.tensor(power))
    expected = torch.exp(
        -math.log(2)
        * (
            (2 * along / length).abs().pow(power)
            + (2 * across / width).abs().pow(power)
        )
    )
    torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-5)
    assert torch.isfinite(actual).all()
    # Length/width are full widths at half maximum for all powers.
    torch.testing.assert_close(
        gaussian_stamp(
            length / 2, torch.tensor(0.0), length, width, torch.tensor(power)
        ),
        torch.tensor(0.5),
    )


@pytest.mark.parametrize("power", [2.0, 100.0])
def test_stable_renderer_gradients(power):
    cfg = Config(min_w_px=0.25)
    st = to_strokes(torch.rand(1, 4, SPB, ADIM), 2, cfg).flatten(1, 2)
    st[..., P_INDEX] = power / cfg.max_p
    st.requires_grad_()
    mask = coverage(st, 32, 32, cfg)
    mask.sum().backward()
    assert torch.isfinite(mask).all() and torch.isfinite(st.grad).all()


def test_large_power_approaches_rectangle():
    along = torch.tensor([0.0, 0.7, 1.2, 2.0])
    across = torch.zeros_like(along)
    result = gaussian_stamp(
        along, across, torch.tensor(2.0), torch.tensor(1.0), torch.tensor(100.0)
    )
    torch.testing.assert_close(
        result, torch.tensor([1.0, 1.0, 0.0, 0.0]), atol=1e-7, rtol=0
    )


@pytest.mark.parametrize("max_p", [10.0, 100.0])
def test_training_and_planning(max_p):
    cfg = Config(
        grids=(1,),
        max_p=max_p,
        zdim=16,
        act_dim=16,
        act_layers=1,
        sigreg_dirs=8,
        plan_iters=2,
    )
    image = torch.rand(2, 3, 32, 32)
    st, canvases = make_trajectory(image, cfg)
    st.requires_grad_()
    model = WorldModel(cfg)
    loss, _ = model(image, image.flip(3), canvases, st)
    loss.backward()
    assert torch.isfinite(loss) and model.enc.proj[-1].weight.grad.norm() > 0
    assert model.act.head.weight.grad.norm() > 0
    assert st.grad[..., P_INDEX].abs().sum() > 0
    final, snaps, info, actions = plan(model, image, cfg)
    assert model.training and all(p.requires_grad for p in model.parameters())
    assert final.shape == image.shape and actions.shape == (2, 1, SPB, 12)
    physical = actions[..., P_INDEX] * max_p
    assert physical.min() >= 1 and physical.max() <= max_p
    assert len(snaps) == 2 and all(
        torch.isfinite(torch.tensor(v)) for v in info.values()
    )
    tiled, _, _, _, _ = plan_image(model, torch.rand(3, 33, 45), cfg, 2)
    assert tiled.shape == (3, 33, 45)


def test_planner_optimizes_power():
    class PowerModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dummy = torch.nn.Parameter(torch.tensor(0.0))
            self.seen = []

        def encode(self, image):
            return image.mean((1, 2, 3)).unsqueeze(-1)

        def actions(self, strokes):
            power = strokes[..., P_INDEX].mean(2).unsqueeze(-1)
            self.seen.append(power.detach().clone())
            return power, torch.full_like(power, -10.0)

    cfg = Config(grids=(1,), plan_iters=20, plan_lr=0.2)
    model = PowerModel()
    torch.manual_seed(2)
    goal = torch.full((1, 3, 32, 32), 0.8)
    _, _, _, st = plan(model, goal, cfg)
    before = (model.seen[0] - 0.8).abs().mean()
    after = (st[..., P_INDEX].mean() - 0.8).abs()
    assert after < before


@pytest.mark.parametrize("value", [0.0, 0.5, -1.0, float("nan"), float("inf"), 1e7])
def test_invalid_max_p(value):
    with pytest.raises(ValueError):
        Config(max_p=value)


def test_checkpoint_roundtrip(tmp_path):
    from painting.training import capture_rng, load_checkpoint, save_checkpoint

    cfg = Config(max_p=25, zdim=16, act_dim=16, act_layers=1)
    model = WorldModel(cfg)
    opt = torch.optim.AdamW(model.parameters())
    path = tmp_path / "model.pt"
    save_checkpoint(path, model, opt, 3, 42, cfg, [capture_rng()], {})
    restored, actual_cfg, ck = load_checkpoint(path, torch.device("cpu"))
    assert actual_cfg == cfg and ck["epoch"] == 3 and ck["action_dim"] == 12
    for key, value in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[key], value)


def test_old_checkpoint_rejected(tmp_path):
    from painting.training import load_checkpoint

    path = tmp_path / "old.pt"
    torch.save({"format_version": 1}, path)
    with pytest.raises(ValueError, match="12-field"):
        load_checkpoint(path, torch.device("cpu"))


def test_native_image_pipeline(monkeypatch):
    from painting import planning

    def identity_plan(net, goals, cfg):
        return (
            goals,
            [torch.ones_like(goals), goals],
            {},
            torch.zeros(len(goals), 1, SPB, ADIM),
        )

    monkeypatch.setattr(planning, "plan", identity_plan)
    image = torch.rand(3, 512, 512)
    final, _, metrics, strokes, _ = planning.plan_image(None, image, Config(), 7)
    assert torch.equal(image, final) and metrics["tiles"] == 256
    assert metrics["pixel_mse"] == 0 and strokes.shape[0] == 256


def test_planner_restores_state_on_failure(monkeypatch):
    from painting import planning

    cfg = Config(zdim=16, act_dim=16, act_layers=1)
    model = WorldModel(cfg)
    next(model.parameters()).requires_grad_(False)
    flags = [p.requires_grad for p in model.parameters()]

    def fail(*args):
        raise RuntimeError("test failure")

    monkeypatch.setattr(planning, "_plan", fail)
    with pytest.raises(RuntimeError, match="test failure"):
        planning.plan(model, torch.rand(1, 3, 32, 32), cfg)
    assert model.training and flags == [p.requires_grad for p in model.parameters()]
