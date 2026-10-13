# Latent-composition painting world model

A painting world model trained on CIFAR-10 images at 32×32 resolution. It learns
how brush strokes change a canvas, then plans strokes to paint a target image.
Larger images are split into 32×32 tiles and combined at their original size.

## Install

Use Python 3.10 or newer. From the repository folder, run:

```bash
pip install -e ".[video]"
```

All commands below use `python -m painting`. In a Kaggle notebook, put `!` before
each shell command and run it from the repository folder.

## Train

```bash
python -m painting train \
  --data ./data \
  --output paint-run \
  --epochs 300 \
  --batch_size 64 \
  --workers 2 \
  --max_p 100
```

CIFAR-10 downloads to `--data` if it is missing. Checkpoints, metrics and evaluation
images are saved in `--output`. The latest checkpoint is `paint-run/latest.pt`.

For two GPUs:

```bash
OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=2 -m painting train \
  --data ./data --output paint-run --epochs 300 --batch_size 64 --workers 2
```

The batch size is shared across GPUs: 64 means 32 images per GPU with two GPUs.

### Resume training

```bash
python -m painting train \
  --resume paint-run/latest.pt \
  --data ./data \
  --output paint-run \
  --epochs 600 \
  --batch_size 64 \
  --workers 2
```

This restores the model, optimizer and saved epoch. `--epochs 600` means 600 total
epochs, including those already completed. Use the same GPU count and batch size
as the original run. For two GPUs, use the `torchrun` launcher shown above.
Brush and model settings are restored from the checkpoint.

On Kaggle, the checkpoint path may be `/kaggle/working/paint-run/latest.pt`.
If it was saved as a Kaggle dataset, use its path under `/kaggle/input/` and keep
the output folder under `/kaggle/working/`.

### Weights & Biases

Tracking is off by default. Install the tracking extra, log in, and add `--wandb`
to your training command:

```bash
pip install -e ".[tracking]"
wandb login
```

Use `--project` to set the W&B project name.

## Evaluate

```bash
python -m painting evaluate \
  --checkpoint paint-run/latest.pt \
  --data ./data \
  --images 100 \
  --batch_size 4 \
  --output outputs/evaluation
```

This plans paintings for held-out CIFAR-10 images and saves comparison images and
error metrics. Keep the image count, batch size and planning settings the same
when comparing checkpoints.

## Paint an image

```bash
python -m painting plan \
  --checkpoint paint-run/latest.pt \
  --image example.png \
  --tile_batch_size 4 \
  --output outputs/painting
```

The result is saved as `outputs/painting/painting.png`. The folder also contains
stage images, planned strokes and metrics. Use `--plan_iters` and
`--plan_restarts` to adjust planning effort.

A 32×32 image uses one tile. A 512×512 image uses 256 tiles. Partial edge tiles are
padded during planning and cropped when the image is assembled. Each tile is
planned independently, so seams can appear between tiles.

## Main settings

| Setting | Default | What it controls |
| --- | --- | --- |
| `--grids` | `1,2,4,8` | Coarse-to-fine training stages |
| `--max_p` | `100` | Maximum brush power |
| `--samples` | `8` | Stamps placed along each Bézier curve |
| `--min_w_px` | `0.5` | Minimum stroke width in pixels |
| `--width_scale` | `1.0` | Stroke width multiplier; smaller values make finer strokes |
| `--plan_iters` | `300` | Planning optimization steps |
| `--plan_restarts` | `1` | Candidate plans per target |
| `--plan_every` | `10` | Number of epochs between training evaluations |
| `--plan_images` | `3` | Images used for evaluation during training |

Each action contains **16 strokes**. This is fixed in the code. The default grids
use 1, 4, 16 and 64 regions, giving **1,360 strokes** across all four stages.
More curve samples change stamp spacing and overlap, so they can change the
painting rather than simply improve its accuracy.

See all options with:

```bash
python -m painting train --help
python -m painting evaluate --help
python -m painting plan --help
```

## Brush and model

Strokes use generalized Gaussian stamps along quadratic Bézier curves. Power
`p = 2` produces a Gaussian brush; higher powers approach a hard rectangular
brush. Training randomly samples each stroke's power between 1 and `max_p`.
The model receives `p / max_p`, and planning optimizes power along with the other
stroke parameters.

Each stroke stores:

```text
[x0, y0, x1, y1, x2, y2, width, r, g, b, alpha, p_normalized]
```

Training combines latent dynamics prediction at each stage end, two-view image
self-supervision and SIGReg. Planning optimizes all actions within a tile
together, with movable action windows and coarse-to-fine brush scales.

## Check data generation

The repository includes one CIFAR-10 frog image. These checks use that image
without downloading the dataset:

```bash
python scripts/check_data_generation.py --max_p 100 --output outputs/data-generation
python scripts/compare_samples.py --output outputs/samples-comparison
```

The first script shows generated strokes and compares brush powers. The second
compares 8 and 16 curve samples at powers 10 and 100. These are data-generation
examples, not results from a trained model.

## Development

```bash
pip install -e ".[test]"
pytest -q
```

The `painting/` package contains the configuration, renderer, stroke generator,
model, training loop and planner. Diagnostic scripts are in `scripts/`, and tests
are in `tests/`.

Licensed under Apache-2.0. See [LICENSE](LICENSE).
