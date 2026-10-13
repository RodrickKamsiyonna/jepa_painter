# Latent-composition painting world model

A research implementation for CIFAR-10 at 32×32. Every stroke uses a generalized
Gaussian brush, with its own randomly sampled or planned power. Larger images are
planned as independent 32×32 tiles and reconstructed at their original size.

## Install

Python 3.10 or newer:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[test]'
pytest -q
```

Optional tracking: `pip install -e '.[tracking]'`, then pass `--wandb`.

## Stroke actions and power

Each action contains **16 ordered strokes**. Each stroke has twelve fields:

```text
[x0, y0, x1, y1, x2, y2, width, r, g, b, alpha, p_normalized]
```

Coordinates and widths are canvas-normalized; RGB and opacity lie in [0, 1].
`p_normalized = p / max_p`, where `max_p` defaults to **100**. Training samples each
physical power independently and uniformly in **[1, max_p]** alongside the other
stroke parameters. The random/optimization coordinate `u_p` maps to
`p = 1 + (max_p - 1) * u_p`. The action encoder receives only `p / max_p`.

Planning jointly optimizes the bounded coordinates of every action, including
`u_p`, through the learned latent dynamics. Rendering converts normalized power
back using `p = p_normalized * max_p`. `max_p` is checkpointed and cannot be changed
at evaluation time: changing it would reinterpret the action representation.
Values from 1 to 1,000,000 are accepted; the default 100 covers soft through nearly
hard rectangular stamps.

The stamp profile in tangent coordinates is:

```text
coverage = exp(-ln(2) * (abs(2*along/length)^p + abs(2*across/width)^p))
```

Length and width are full widths at half maximum. Power 2 gives an elliptical
Gaussian; higher powers approach a rectangular stamp. Stamps follow the original
quadratic Bézier curves and combine by coverage union, followed by ordered alpha
compositing. The implementation evaluates powers in log space and caps each
powered-distance term at 80, where opacity is below float32 compositing precision,
to avoid overflow for sharp brushes. Width floors still apply (`--min_w_px`,
`--width_scale`). Very narrow, hard stamps can miss pixel centers.

Default grids 1,2,4,8 use 1,4,16,64 regions respectively, with 16 strokes per region:
16,64,256,1,024 new strokes and **1,360 total**. The action count is 85.

## Train

```bash
latent-paint train --max_p 100 --output runs/experiment
```

Smaller CPU smoke run using synthetic data:

```bash
latent-paint train --fake --max_p 100 --grids 1 --epochs 1 \
  --batch_size 8 --workers 0 --zdim 16 --act_dim 16 --act_layers 1 \
  --sigreg_dirs 8 --plan_iters 2 --plan_images 2 --output runs/smoke
```

CUDA distributed training (batch size is global):

```bash
torchrun --standalone --nproc_per_node=2 -m painting train \
  --max_p 100 --batch_size 256 --output runs/experiment
```

Runs save configuration, library/environment metadata, JSONL training metrics,
held-out evaluation grids/metrics, and an atomic `latest.pt` checkpoint. The shared
evaluation routine runs on fixed held-out goals at `--plan_every` epochs and the
last epoch. It preserves training RNG state and records stage dynamics errors,
previous-canvas baselines, latent target errors, predicted-versus-rendered latent
gaps, and pixel MSE. It does not update model weights.

Resume:

```bash
latent-paint train --resume runs/experiment/latest.pt --epochs 600 \
  --output runs/experiment
```

Checkpoint configuration supplies architecture, rendering and planning settings.
Current training arguments supply dataset, epochs, optimizer schedule and logging.
Changing total epochs changes the cosine schedule. RNG state is saved per rank;
resume requires the original world size. Device, library and worker changes can
affect reproducibility. Load trusted checkpoints only (Python serialization).

**The new checkpoint format uses 12-field actions. Earlier 11-field brush-type or
fixed-power checkpoints are rejected with an explicit retraining message.**

## Evaluate and plan

```bash
latent-paint evaluate --checkpoint runs/experiment/latest.pt --data ./data \
  --images 100 --batch_size 4 --output outputs/evaluation

latent-paint plan --checkpoint runs/experiment/latest.pt --image example.png \
  --tile_batch_size 4 --output outputs/painting
```

Evaluation uses sample-weighted aggregate metrics and saves per-batch metrics and
grids. Its seed resets per batch; keep batching and sample order fixed for fair
comparisons. Evaluation may download CIFAR-10 if absent; the standalone example
script below never downloads a dataset.

A 512×512 goal becomes 256 row-major tiles; 32×32 uses one tile. Other dimensions
use replicated right/bottom padding and crop back to the original dimensions.
EXIF orientation is applied and inputs are converted to RGB. Tiles are independent,
so seams and a lack of global composition are possible.

All actions within each tile are optimized jointly, with free action-window
positions and coarse-to-fine window/brush scales. Restart selection uses final
latent error. There is no MPC or stage-by-stage optimization.

Image planning saves `painting.png`, blank and per-stage PNGs, metrics/configuration,
and `strokes.pt`. Stored strokes use tile-local normalized xy and normalized power;
`stroke_fields`, `physical_p`, and original-size/layout metadata are included.
`--plan_iters` and `--plan_restarts` override inference effort.

## Inspect generated data on one CIFAR-10 image

A single 32×32 frog preview and its source metadata are in `examples/`. No full
dataset was downloaded to obtain it.

```bash
python scripts/check_data_generation.py --max_p 100 \
  --output outputs/data-generation
```

This checks all 1,360 random-power strokes, action bounds, finite pixels and
sequential-compositing agreement. It saves a stage comparison of random per-stroke
powers versus fixed powers 2,10,100 with identical geometry/colors, a sheet of the
16 individual coarse-region strokes labeled with physical power, trajectory tensors,
and a JSON report. These images demonstrate data generation, not trained-model
performance. Supply another native 32×32 image using `--image`.

Compare 8 versus 16 curve samples at fixed powers 10 and 100 on identical strokes:

```bash
python scripts/compare_samples.py --output outputs/samples-comparison
```

This saves final and per-stage comparison grids, trajectory tensors and metrics.
`samples` also changes longitudinal stamp length and overlap in the renderer.

## Objective and layout

Training retains mean open-loop dynamics error at each stage end, direct two-view
goal-image SSL, and SIGReg on real-canvas and SSL representations. There is no
stage-four-to-goal alignment loss, teacher, SSL predictor or stop-gradient.
The latent recurrence is `z_next = sigmoid(xa)*z + sigmoid(-xa)*s`.

- `painting/config.py`: experiment configuration and stroke schema.
- `painting/rendering.py`: generalized Gaussian coverage and compositing.
- `painting/strokes.py`: random imitation trajectories and color selection.
- `painting/model.py`: image/action encoders, recurrence and losses.
- `painting/planning.py`: joint normalized-action optimization and image tiling.
- `painting/training.py`: training, checkpoints and shared evaluation.
- `painting/video.py`: streamed playback of planned strokes across all image tiles.
- `painting/cli.py`: command-line entry points.
- `scripts/check_data_generation.py`: reproducible single-image diagnostics.
- `tests/test_core.py`: formula/bounds, high-power stability, power gradients and
  optimization, model gradients, checkpoints and native-size reconstruction.

The code is Apache-2.0 licensed (see `LICENSE`). Add author and project citation
metadata before publication. No research performance claims are implied by tests.

### Stroke drawing video

Install video support with `pip install -e ".[video]"`. The `plan` command now
writes `painting.mp4` alongside the final PNG and stroke tensors. It starts with
the entire image white and advances every 32×32 tile together, then holds the
rendered final painting for one second. The target image is not substituted for
the rendered result. Planning optimization happens first; the video replays the
selected plan rather than its optimization iterations.

```bash
python -m painting plan --checkpoint /kaggle/working/paint-run/latest.pt \
  --image /kaggle/working/goal.png --output /kaggle/working/painting \
  --video_fps 30 --video_strokes_per_frame 8 --video_scale 1
```

Use `--video_strokes_per_frame 1` to show each stroke separately, and
`--video_scale 8` to enlarge a 32×32 painting for viewing. Larger images retain
their native dimensions with scale 1. All tiles advance in parallel; no grid
lines are painted into the image. Use `--no_video` to export only static outputs
without installing the video dependencies. MP4 encoding is lossy; the PNG and
replay tensors preserve the original rendered pixels.
Videos use H.264 with `yuv420p` for browser and mobile compatibility. Odd image
dimensions receive one extra edge pixel on the right or bottom; PNGs retain the
original dimensions.
