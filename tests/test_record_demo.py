import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dev"))

import record_demo  # noqa: E402

BASE = 100


def test_the_demo_keeps_each_stretch_at_its_speed_and_cuts_the_stretches_at_speed_zero():
    marks = [(0.0, 0), (1.5, 1), (4.0, 10), (24.0, 1), (30.0, 0), (31.0, 1), (33.0, 0)]
    assert record_demo.cuts(marks) == [(1.5, 4.0, 1), (4.0, 24.0, 10), (24.0, 30.0, 1), (31.0, 33.0, 1)]
    assert record_demo.cuts([(0.0, 0)]) == []


def test_the_gif_keeps_a_pixel_until_it_or_its_whole_block_moves_more_than_the_codec_noise():
    still, block, faint = record_demo.GIF_STILL, record_demo.GIF_BLOCK, record_demo.GIF_BLOCK_MEAN
    first = np.full((block, block, 3), BASE, np.uint8)
    noise = first.copy()
    noise[0, 0] = BASE + still
    jump = noise.copy()
    jump[0, 1] = BASE + still + 1
    fade = np.full_like(first, BASE + faint + 1)
    shown = [f.copy() for f in record_demo.steady([first, noise, jump, fade])]
    assert shown[1][0, 0, 0] == BASE
    assert (shown[2][0, 1, 0], shown[2][0, 0, 0], shown[2][1, 1, 0]) == (BASE + still + 1, BASE, BASE)
    assert (shown[3] == BASE + faint + 1).all()
