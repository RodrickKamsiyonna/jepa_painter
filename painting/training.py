"""Distributed training, deterministic held-out evaluation and checkpoints."""

import datetime
import json
import math
import os
import platform
import random
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torchvision
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import transforms as T
from torchvision.datasets import CIFAR10, FakeData
from torchvision.utils import make_grid, save_image

from .config import ADIM, STROKE_FIELDS, Config
from .model import WorldModel, compose
from .planning import plan
from .strokes import make_trajectory


class TrainViews:
    def __init__(self, res=32):
        self.raw = T.ToTensor()
        self.aug = T.Compose(
            [
                T.RandomResizedCrop(
                    res, scale=(0.3, 1.0), interpolation=T.InterpolationMode.BICUBIC
                ),
                T.RandomHorizontalFlip(0.5),
                T.RandomApply([T.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
                T.RandomGrayscale(0.2),
                T.RandomApply([T.GaussianBlur(3, sigma=(0.1, 2.0))], p=0.5),
                T.RandomSolarize(128, p=0.2),
                T.ToTensor(),
            ]
        )

    def __call__(self, image):
        return self.raw(image), self.aug(image), self.aug(image)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_checkpoint(path, device):
    # Load only trusted checkpoints: optimizer/RNG state requires pickle.
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if ck.get("format_version") != 2 or ck.get("action_dim") != ADIM:
        raise ValueError(
            "This checkpoint predates the 12-field p/max_p action. Retrain with the new action schema."
        )
    cfg = Config(**ck["config"])
    model = WorldModel(cfg).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, cfg, ck


def save_checkpoint(path, model, opt, epoch, step, cfg, rng_states, train_args):
    temporary = path.with_suffix(".tmp")
    torch.save(
        {
            "format_version": 2,
            "action_dim": ADIM,
            "stroke_fields": STROKE_FIELDS,
            "model": model.state_dict(),
            "opt": opt.state_dict(),
            "epoch": epoch,
            "gstep": step,
            "config": cfg.to_dict(),
            "rng_states": rng_states,
            "train_args": train_args,
        },
        temporary,
    )
    temporary.replace(path)


def capture_rng():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all(),
    }


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def dataset(data, train, fake=False, views=False, download=False):
    transform = TrainViews() if views else T.ToTensor()
    return (
        FakeData(128 if train else 32, (3, 32, 32), 10, transform=transform)
        if fake
        else CIFAR10(data, train=train, download=download, transform=transform)
    )


def evaluate(model, goals, cfg, output, prefix="eval"):
    """Shared in-training/post-training evaluation; keeps training RNG intact."""
    state = capture_rng()
    was = model.training
    try:
        model.eval()
        seed_all(cfg.seed + 10000)
        _, snaps, metrics, _ = plan(model, goals, cfg)
        with torch.no_grad():
            strokes, canvases = make_trajectory(goals, cfg)
            z0 = model.encode(torch.ones_like(goals[:1]))[0]
            s, xa = model.actions(strokes)
            pred = compose(z0, s, xa, model.ends)
            actual = model.encode(canvases.flatten(0, 1)).view(
                len(cfg.grids), len(goals), -1
            )
            previous = torch.cat(
                (z0.view(1, 1, -1).expand(1, len(goals), -1), actual[:-1])
            )
            for k in range(len(cfg.grids)):
                metrics[f"dynamics_mse_s{k + 1}"] = (
                    (pred[k] - actual[k]).square().mean().item()
                )
                metrics[f"dynamics_baseline_s{k + 1}"] = (
                    (previous[k] - actual[k]).square().mean().item()
                )
            trajectory = [torch.ones_like(goals)] + list(canvases)
        for name, frames in [("plan", snaps), ("trajectory", trajectory)]:
            grid = make_grid(
                torch.nn.functional.interpolate(
                    torch.cat([goals] + frames), scale_factor=4, mode="nearest"
                ),
                nrow=len(goals),
            )
            save_image(grid, output / f"{prefix}_{name}.png")
        (output / f"{prefix}_metrics.json").write_text(
            json.dumps(metrics, indent=2) + "\n"
        )
        return metrics
    finally:
        restore_rng(state)
        model.train(was)


def train(args, cfg):
    rank, world, local = (
        int(os.getenv("RANK", "0")),
        int(os.getenv("WORLD_SIZE", "1")),
        int(os.getenv("LOCAL_RANK", "0")),
    )
    device = (
        torch.device("cuda", local)
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    if world > 1:
        if device.type != "cuda":
            raise ValueError("Distributed training requires CUDA")
        torch.cuda.set_device(local)
        dist.init_process_group("nccl", timeout=datetime.timedelta(minutes=60))
    try:
        _train(args, cfg, rank, world, local, device)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _train(args, cfg, rank, world, local, device):
    seed_all(cfg.seed + rank)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    resume = None
    if args.resume:
        model, saved_cfg, resume = load_checkpoint(args.resume, device)
        cfg = saved_cfg
    else:
        model = WorldModel(cfg).to(device)
    if args.batch_size % world or args.batch_size < world:
        raise ValueError("batch_size must be a positive multiple of world size")
    if rank == 0:
        if not args.fake:
            dataset(args.data, True, download=True)
            dataset(args.data, False, download=True)
        (output / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2) + "\n")
        metadata = {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "numpy": np.__version__,
            "cuda": torch.version.cuda,
            "world_size": world,
            "training_arguments": vars(args),
            "action_dim": ADIM,
            "power_encoding": "p/max_p; p in [1, max_p]",
        }
        (output / "environment.json").write_text(json.dumps(metadata, indent=2) + "\n")
    if world > 1:
        dist.barrier()
    ds = dataset(args.data, True, args.fake, views=True)
    test = dataset(args.data, False, args.fake)
    goals = torch.stack(
        [test[i][0] for i in range(min(args.plan_images, len(test)))]
    ).to(device)
    sampler = DistributedSampler(ds, world, rank) if world > 1 else None
    loader = DataLoader(
        ds,
        batch_size=args.batch_size // world,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    if not len(loader):
        raise ValueError("batch_size exceeds training dataset size")
    opt = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.wd, betas=(0.9, 0.95)
    )
    start, step = 0, 0
    if resume:
        opt.load_state_dict(resume["opt"])
        start, step = resume["epoch"], resume["gstep"]
        if len(resume["rng_states"]) != world:
            raise ValueError("Resume requires the original world size")
        restore_rng(resume["rng_states"][rank])
    wrapped = (
        DDP(model, device_ids=[local], broadcast_buffers=False) if world > 1 else model
    )
    wb = None
    if args.wandb and rank == 0:
        import wandb

        wb = wandb.init(project=args.project, config={**vars(args), **cfg.to_dict()})
    total = args.epochs * len(loader)
    for epoch in range(start, args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        model.train()
        accumulated = None
        for (image, v1, v2), _ in loader:
            image, v1, v2 = [x.to(device, non_blocking=True) for x in (image, v1, v2)]
            strokes, canvases = make_trajectory(image, cfg)
            lr = (
                args.lr
                * min(1.0, (step + 1) / max(1, args.warmup))
                * 0.5
                * (1 + math.cos(math.pi * min(1.0, step / total)))
            )
            for group in opt.param_groups:
                group["lr"] = lr
            opt.zero_grad(set_to_none=True)
            loss, stats = wrapped(v1, v2, canvases, strokes, step)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            opt.step()
            stats = stats.detach()
            if world > 1:
                dist.all_reduce(stats)
                stats /= world
            accumulated = stats if accumulated is None else accumulated + stats
            step += 1
            if rank == 0 and step % args.log_every == 0:
                values = stats.tolist()
                record = dict(
                    zip(
                        (
                            "train/loss",
                            "train/dyn",
                            "train/ssl",
                            "train/sigreg",
                            "train/z_std",
                        ),
                        values[:5],
                    )
                ) | {"train/lr": lr, "epoch": epoch + 1, "step": step}
                k = len(cfg.grids)
                for index in range(k):
                    record[f"train/stage{index + 1}_pred"] = values[5 + index]
                    record[f"train/stage{index + 1}_baseline"] = values[5 + k + index]
                with (output / "train.jsonl").open("a") as stream:
                    stream.write(json.dumps(record) + "\n")
                if wb:
                    wb.log(record, step=step)
        if world > 1:
            dist.barrier()
        if rank == 0:
            print(
                f"epoch={epoch + 1} statistics={(accumulated / len(loader)).tolist()}",
                flush=True,
            )
            if (epoch + 1) % args.plan_every == 0 or epoch + 1 == args.epochs:
                metrics = evaluate(model, goals, cfg, output, "latest")
                print(json.dumps(metrics), flush=True)
                if wb:
                    wb.log({f"eval/{k}": v for k, v in metrics.items()}, step=step)
        states = [None] * world if rank == 0 else None
        if world > 1:
            dist.gather_object(capture_rng(), states, dst=0)
        else:
            states = [capture_rng()]
        if rank == 0:
            save_checkpoint(
                output / "latest.pt",
                model,
                opt,
                epoch + 1,
                step,
                cfg,
                states,
                vars(args),
            )
        if world > 1:
            dist.barrier()
    if wb:
        wb.finish()
