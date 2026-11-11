import os

import numpy as np
import pytest
from conftest import make_world, world_scene
from fake_sampler import FakeSampler
from PIL import Image
from test_search import CORE, run

from rwg2migrator import appearance, area, maps, scan

HEAT_STEP = 0.25


def test_heat_ramp_gets_darker_from_green_to_red():
    steps = np.arange(0, maps.HEAT_TICKS[-1] + HEAT_STEP, HEAT_STEP)
    lightness = [appearance.lab(maps.heat_colour(s))[0] for s in steps]
    assert all(a > b for a, b in zip(lightness, lightness[1:], strict=False))
    r, g, _ = maps.heat_colour(0)
    assert g > r and maps.heat_colour(maps.HEAT_TICKS[-1])[0] > maps.heat_colour(maps.HEAT_TICKS[-1])[1]


def test_bpp_and_window_arg():
    assert maps.choose_bpp((0, 0, 1000, 800)) == 1 and maps.choose_bpp((0, 0, 3000, 100)) == 4
    assert maps.window_arg("10,-20,300") == (10, -20, 310, 280)
    assert maps.render_window((8, -24, 600, 300), 2) == (0, -32, 608)


def test_map_of_the_world_and_of_a_candidate(tmp_path, cache):
    w = make_world(str(tmp_path / "s"), chunk_specs={(x, z): {} for x in range(8) for z in range(8)})
    a = area.build({}, CORE, set(), 1)
    sc = scan.scan(w, cache, workers=1)
    out = tmp_path / "now.png"
    facts = maps.draw(world_scene(sc, a, "2.9.0"), None, str(out))
    assert Image.open(out).size == tuple(facts["size"]) and len(facts["windows"]) == 1
    job, _ = run(str(tmp_path / "data"), w, a, cache)
    chosen = maps.pick(job, 1)
    renderer = maps.Renderer(chosen, FakeSampler)
    scene = maps.Scene(sc, a, "2.9.0", chosen, [tuple(x) for x in a["windows"]])
    facts = maps.draw(scene, renderer, str(tmp_path / "cand.png"))
    win = facts["windows"][0]
    assert win["seam_pairs"] > 0 and win["config"] == chosen["candidate"]["config"]
    assert maps.cached(chosen, scene.windows, None)
    img = np.asarray(Image.open(tmp_path / "cand.png"))
    assert (img.reshape(-1, 3) == maps.TAINT).all(1).any()


def test_far_apart_parts_get_two_windows(tmp_path, cache):
    w = make_world(str(tmp_path / "s"), chunk_specs={(x, z): {} for x in range(8) for z in range(8)})
    a = area.build({}, {(3, 3), (3 + 330, 3)}, set(), 1)
    facts = maps.draw(world_scene(scan.scan(w, cache, workers=1), a, "2.9.0"), None, str(tmp_path / "two.png"))
    assert len(facts["windows"]) == 2 and facts["windows"][0]["window"][2] < facts["windows"][1]["window"][0]


@pytest.mark.parametrize("text", ["8,8,256", "1000,-500,2048", "0,0,2049", "-7,5,100"])
def test_a_window_off_the_chunk_grid_is_drawn_on_the_chunks_that_cover_it(tmp_path, cache, text):
    w = make_world(str(tmp_path / "s"), chunk_specs={(x, z): {} for x in range(8) for z in range(8)})
    a = area.build({}, CORE, set(), 1)
    job, _ = run(str(tmp_path / "data"), w, a, cache)
    chosen, box = maps.pick(job, 1), maps.window_arg(text)
    scene = maps.Scene(scan.scan(w, cache, workers=1), a, "2.9.0", chosen, [box])
    drawn = maps.draw(scene, maps.Renderer(chosen, FakeSampler), str(tmp_path / "w.png"))["windows"][0]
    x0, z0, x1, z1 = drawn["window"]
    assert x0 <= box[0] and z0 <= box[1] and x1 >= box[2] and z1 >= box[3] and x0 % maps.CHUNK == z0 % maps.CHUNK == 0
    assert (x1 - x0) % maps.CHUNK == (z1 - z0) % maps.CHUNK == 0 and maps.cached(chosen, [box], None)


class Counting(FakeSampler):
    def render(self, config, window, bpp, continental=True):
        self.calls += 1
        return super().render(config, window, bpp, continental)


def test_a_torn_render_is_drawn_again_not_trusted(tmp_path, cache):
    w = make_world(str(tmp_path / "s"), chunk_specs={(x, z): {} for x in range(8) for z in range(8)})
    job, _ = run(str(tmp_path / "data"), w, area.build({}, CORE, set(), 1), cache)
    sampler = Counting()
    renderer = maps.Renderer(maps.pick(job, 1), lambda: sampler)
    win, box = (0, 0, 256), [(0, 0, 256, 256)]
    good, _ = renderer.render(win, 2)
    step, _ = renderer.step(win, 2)
    path = os.path.join(job.folder, step, maps.RENDER_FILE)
    manifest = os.path.join(job.folder, step, "manifest.json")
    assert maps.cached(renderer.pick, box, 2) and renderer.render(win, 2)[0].tobytes() == good.tobytes() and sampler.calls == 1
    for torn_after_manifest in (True, False):
        with open(path, "r+b") as f:
            f.truncate(os.path.getsize(path) // 2)
        if not torn_after_manifest:
            os.remove(manifest)
        assert not maps.cached(renderer.pick, box, 2)
        assert renderer.render(win, 2)[0].tobytes() == good.tobytes() and maps.cached(renderer.pick, box, 2)
    assert sampler.calls == 3
