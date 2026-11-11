import csv

import numpy as np
import pytest
from conftest import golden

from rwg2migrator import area, score

L_SHAPE = {(1, 1), (2, 1), (1, 2)}
WINDOW = 64


def poc_rows():
    with open(golden("search-green-rows.csv")) as f:
        return {tuple(int(r[k]) for k in ("bx", "bz", "lx", "lz")): r for r in csv.DictReader(f)}


def mask_walk(core):
    """Every column step that leaves the core chunks, found by rolling their mask one column each way (as the POC did)."""
    inside = np.zeros((WINDOW, WINDOW), bool)
    for cx, cz in core:
        inside[cz * 16 : cz * 16 + 16, cx * 16 : cx * 16 + 16] = True
    pairs = set()
    for dx, dz in score.DIRS:
        zz, xx = np.nonzero(inside & ~np.roll(inside, (-dz, -dx), (0, 1)))
        pairs |= set(zip(xx.tolist(), zz.tolist(), (xx + dx).tolist(), (zz + dz).tolist(), strict=True))
    return pairs


def test_edge_pairs_match_a_mask_walk():
    inner, outer = score.edge_pairs(L_SHAPE)
    assert set(map(tuple, np.c_[inner, outer].tolist())) == mask_walk(L_SHAPE)
    assert len(inner) == 8 * 16 and len(np.unique(outer, axis=0)) == 8 * 16 - 1


@pytest.mark.fixture
def test_edge_pairs_match_the_poc_mask_walk():
    z = np.load(golden("poc-seam.npz"))
    inner, outer = score.edge_pairs(set(map(tuple, z["core"].tolist())))
    mine = set(map(tuple, np.c_[inner, outer].tolist()))
    assert len(inner) == 2784 and mine == set(map(tuple, np.c_[z["inner"], z["outer"]].tolist()))
    assert len(np.unique(outer, axis=0)) == 2776


def test_seam_counts_the_steps_between_the_ground_inside_and_the_top_outside():
    steps = np.array([0, 0, 1, 1, 2, 5, 12, 12, 31, 40], np.float64)
    s = score.seam(np.full(len(steps), 70.0), score.surface(69.5 + steps))
    assert (s["pairs"], s["median"], s["max"], s["over10"], s["over30"]) == (10, 3.5, 40.0, 0.4, 0.2)
    assert s["p90"] == pytest.approx(31.9) and s["seam"] == pytest.approx(3.5 + 10 * 0.4 + 30 * 0.2)


@pytest.mark.fixture
def test_seam_reproduces_five_poc_rows_on_the_stale_ground():
    z, rows = np.load(golden("poc-seam.npz")), poc_rows()
    for config, h in zip(z["configs"], z["h_out"], strict=True):
        s, r = score.seam(z["h_in"].astype(np.float64), score.surface(h)), rows[tuple(config.tolist())]
        assert [s[k] for k in ("median", "p90", "over10", "over30")] == [float(r[k]) for k in ("median", "p90", "over10", "over30")]
        assert s["seam"] == pytest.approx(float(r["score"]), abs=1e-12)
    green1 = score.seam(z["h_in"].astype(np.float64), score.surface(z["h_out"][0]))
    assert round(green1["seam"], 3) == 1.783 and round(100 * green1["over10"], 2) == 7.83


@pytest.mark.fixture
def test_penalty_and_total_reproduce_every_poc_row():
    for r in poc_rows().values():
        inc = {b: float(r["inc_" + b]) for b in score.BAND_NAMES}
        assert score.penalty(inc) == pytest.approx(float(r["penalty"]), abs=1e-9)
        assert float(r["score"]) + score.penalty(inc) == pytest.approx(float(r["total"]), abs=1e-9)


def test_tolerance_scale_and_estimate_records():
    at = dict(score.TOLERANCE)
    assert score.penalty(at) == 0 and score.penalty(at | {"edge": at["edge"] + 0.1}) == pytest.approx(score.EXCESS * 4 * 0.1)
    assert score.penalty(at | {"edge": 0.2}, scale=2) == 0
    edge = {"level": "edge", "seam": 1.0, "inc_edge": 0.25} | {f"est_{b}": 0.0 for b in score.BAND_NAMES[1:]}
    assert score.total(edge) == pytest.approx(1.0 + score.EXCESS * 4 * 0.1)
    main = {"level": "main", "seam": 1.0} | {f"inc_{b}": v for b, v in zip(score.BAND_NAMES, (0.25, 0.15, 0.35, 0.0, 0.5), strict=True)}
    assert score.total(main) == pytest.approx(1.0 + score.EXCESS * (4 * 0.1 + 2 * 0.2 + 0.5 * 0.15))


def test_surface_is_floor_plus_one():
    assert score.surface([63.0, 63.99, -0.5]).tolist() == [64.0, 64.0, 0.0]


def test_grid_and_distance():
    core = {(0, 0), (1, 0)}
    pts, d = score.grid_points(core)
    assert not set(map(tuple, (pts // 16).tolist())) & core and d.max() <= score.FAR
    beyond = (d > score.GRID_FULL) & np.isfinite(d)
    assert ((pts[beyond] // 16) % score.FAR_STRIDE == 0).all()
    assert area.distance([[32, 3], [40, 3], [-17, 0], [5, 5]], core).tolist() == [1.0, 9.0, 17.0, 0.0]
    assert area.nearest_point([40, -20], core).tolist() == [31.0, 0.0]
    five = {(x, z) for x in range(-2, 3) for z in range(-2, 3)}
    assert area.near({(0, 0)}, 16) == five - {(2, 2), (2, -2), (-2, 2), (-2, -2)}
