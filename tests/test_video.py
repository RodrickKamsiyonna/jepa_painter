import pytest
import torch

from painting.config import ADIM, Config
from painting.planning import join_tiles
from painting.rendering import render_stage
from painting.strokes import to_strokes
from painting.video import replay_frames


@pytest.mark.parametrize("group", [1, 7, 32])
def test_replay_starts_white_and_matches_tiled_render(group):
    cfg = Config(grids=(1,))
    torch.manual_seed(7)
    strokes = to_strokes(torch.rand(4, 1, 16, ADIM), 1, cfg)
    layout = (45, 53, 2, 2)
    frames = list(replay_frames(strokes, layout, cfg, group))
    assert torch.equal(frames[0], torch.ones(3, 45, 53))
    expected = join_tiles(
        render_stage(torch.ones(4, 3, 32, 32), strokes.flatten(1, 2), cfg), layout
    )
    torch.testing.assert_close(frames[-1], expected, atol=1e-6, rtol=1e-6)
    assert len(frames) == 1 + (16 + group - 1) // group


def test_replay_rejects_zero_group():
    with pytest.raises(ValueError, match="positive"):
        list(replay_frames(torch.zeros(1, 1, 16, ADIM), (32, 32, 1, 1), Config(), 0))


def test_mp4_odd_dimensions_decode_as_yuv420p(tmp_path):
    imageio = pytest.importorskip("imageio.v2")
    ffmpeg = pytest.importorskip("imageio_ffmpeg")
    import subprocess

    from painting.video import save_stroke_video

    cfg = Config(grids=(1,))
    strokes = to_strokes(torch.rand(1, 1, 16, ADIM), 1, cfg)
    path = tmp_path / "odd.mp4"
    save_stroke_video(path, strokes, (29, 31, 1, 1), cfg, fps=2, strokes_per_frame=16)
    reader = imageio.get_reader(path)
    try:
        assert reader.get_data(0).shape == (30, 32, 3)
        assert reader.count_frames() == 4
    finally:
        reader.close()
    probe = subprocess.run(
        [ffmpeg.get_ffmpeg_exe(), "-i", str(path), "-f", "null", "-"],
        capture_output=True, text=True, check=True,
    )
    assert "yuv420p" in probe.stderr
