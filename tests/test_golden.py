"""Green #1 (Smooth) on the fixture: the generator's records from the direct evaluation are pinned."""

import json

import numpy as np
import pytest
from conftest import golden

from rwg2migrator import appearance, area, score, search, verdicts
from rwg2migrator.generator import RECORD

pytestmark = pytest.mark.fixture
PIN = json.loads(golden("green1.json").read_text())
FACTS = json.loads(golden("world-facts.json").read_text())
TAINT_POC, TAINT_TOLERANCE = 330, 16


@pytest.fixture(scope="module")
def green1():
    site = score.Site.load(golden("green1-site.npz"))
    z = np.load(golden("green1-sample.npz"))
    rec, names = z["rec"].view(RECORD), [str(n) for n in z["names"]]
    ref = appearance.Reference.from_dict(PIN["reference"])
    judge = score.Judge(ref, frozenset(PIN["bans"]), appearance.table("2.9.0"))
    e, n = len(site.edge), len(site.edge) + len(site.grid)
    r = score.evaluate_edge(site, (rec[:e], names), judge)
    r |= score.evaluate_grid(site, (rec[e:n], names), judge, r) | score.evaluate_land(rec[n:])
    return site, rec[e:n], names, judge, r


def test_direct_evaluation_matches_the_pin(green1):
    *_, r = green1
    assert {k: r[k] for k in r} == {k: v for k, v in PIN["record"].items() if k not in ("stage", "bx", "bz", "lx", "lz")}
    assert (r["pairs"], r["median"], r["p90"], r["over30"], r["feasible"]) == (2784, 1.0, 9.0, 0.0, True)
    assert round(r["seam"], 3) == 1.772 and round(100 * r["over10"], 2) == 7.72
    assert score.penalty(score.incs(r)) == pytest.approx(0.6624) and score.total(r) == pytest.approx(2.43467, abs=1e-5)


def test_reference_and_bans_of_the_pin():
    ref = PIN["reference"]
    assert (ref["name"], ref["biome_id"], ref["median_y"], ref["columns"]) == ("Temperate Forest", 236, 70, 459317)
    assert np.abs(np.array(ref["colour"]) - appearance.rgb("#4C6E31")).max() <= 1 and abs(ref["surface"] - 0.95) <= 0.005
    assert PIN["bans"] == ["other", "sand", "snow", "water", "wet"]


def test_tainted_land_330_blocks_north(green1):
    core = set(map(tuple, green1[0].core.tolist()))
    d = float(area.distance([PIN["taint"]["point"]], core)[0])
    assert d == pytest.approx(PIN["taint"]["blocks"], abs=0.05) and abs(d - TAINT_POC) <= TAINT_TOLERANCE
    assert search.with_direction({"point": PIN["taint"]["point"]}, core)["direction"] == "north"


def test_one_render_window_holds_the_pinned_tainted_land_and_ocean(green1):
    wins = search.find_windows(set(map(tuple, green1[0].kept.tolist())))
    assert wins == [tuple(w) for w in FACTS["render_windows"]]
    assert search._inside(np.array([PIN["taint"]["point"], PIN["ocean_point"]]), wins).all()
    assert max(PIN["taint"]["blocks"], PIN["ocean"]) < verdicts.REACH


def test_cards_lines(green1):
    site, grid, names, judge, r = green1
    near = site.grid_d <= score.NEAR
    look, _, same = judge.columns(grid[near], names)
    kept = set(map(tuple, site.kept.tolist()))
    off = search.off_side(site.grid[near], look, same, search.part_centres(site.grid[near], kept))
    assert (off["side"], off["look"], off["columns"]) == ("west", "dull", 262)
    metrics = r | {"base": "Temperate Forest", "off": off, "ocean": PIN["ocean"], "taint": PIN["taint"]}
    card = verdicts.card(metrics, 1, True)
    assert card["badges"] == [verdicts.RECOMMENDED, verdicts.EXCELLENT]
    assert card["lines"] == [
        "Border: very close (most steps 1 block; smoothing evens the rest)",
        "Looks: 73% the same as your base (Temperate Forest) right at the edge; duller grass on the west side",
        "Water: no ocean within 500 blocks; some pools near the border",
        "Tainted Land: 330 blocks north." + verdicts.SPREAD,
    ]
    assert PIN["edge_biomes"] == {"Temperate Forest": 73.1, "Jade Cliffs": 20.3, "Cold Forest": 6.6}
