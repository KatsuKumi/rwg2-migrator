import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
from fake_sampler import NAMES

from rwg2migrator.web import server

WALK = Path(__file__).parent.parent / "dev" / "ui_walkthrough.py"
LIMIT = 900
TILE_LIMIT_S = 5.0
ZOOM_LEVELS = 5


def chromium():
    playwright = pytest.importorskip("playwright.sync_api")
    try:
        with playwright.sync_playwright() as p:
            p.chromium.launch().close()
    except Exception as e:
        pytest.skip(f"Playwright has no Chromium here ({type(e).__name__})")


def test_the_wizard_runs_from_world_to_verify_on_screen_text_alone(tmp_path):
    chromium()
    out = tmp_path / "walk"
    subprocess.run([sys.executable, str(WALK), "--out", str(out), "--fake", "--apply"], check=True, timeout=LIMIT, cwd=WALK.parent,
                   env=os.environ | {"PYTHONPATH": str(WALK.parent.parent)})  # fmt: skip
    r = json.loads((out / "report.json").read_text())
    assert "stopped" not in r and r["stuck"] == [], r.get("stopped")
    assert {"home", "world", "area", "matches-swipe", "preview", "apply", "after", "verify"} <= set(r["width"])
    assert all(px <= 0 for sizes in r["width"].values() for px in sizes.values()), r["width"]
    for m in r["maps"].values():
        assert m["zoom_levels"] == ZOOM_LEVELS and m["revisit"]["redrawn_total"] == 0 and m["overlay_tiles_zoomed_out"]
    assert r["maps"]["#area-map"]["revisit"]["drawn"] == 0
    tiles = r["candidate_tiles"]
    assert tiles["drawn_while_waiting"] and tiles["max_age_s"] < TILE_LIMIT_S, tiles
    assert len(r["search"]["card_updates"]) >= 2 and r["search"]["reload"]["reattached"]
    assert r["search_job"]["prefs"] == {"base": None, "banned": None, "tolerance": 1.0}
    titles = r["search"]["titles"]
    assert any(t.startswith("Searching: ") and "m:ss" in t for t in titles) and "Done – rwg2-migrator" in titles, titles
    assert r["area_suggest"]["suggested"]["core"] == 3 and r["area_suggest"]["cleared"]["core"] == 0
    removed = r["area"]["removed"]
    assert "(GregTech machines 1)" in removed and "BaseMetaTileEntity" not in removed, removed
    assert "also hold 1 more, of kinds villages and dungeons have too (chests 1…)" in removed, removed
    add = r["area"]["add_these"]
    assert add["core_after"] == add["core_before"] + add["listed"]
    e = r["area"]["edits"]
    assert e["painted"]["core"] == e["erased"]["core"] + 1 and e["margin2"]["kept"] > e["margin1"]["kept"] == r["area"]["claims"]["kept"]
    assert r["apply"]["enabled"] and "Your world now uses RWG2 Smooth" in r["after"] and "No problem found." in r["verify"]
    assert r["import_ticked"] == {"ticks": [True, False], "core": 3, "unticked_core": 0}
    vm = r["verify_map"]
    assert vm["choices"] == ["PREVIEW", "BEFORE APPLY", "NOW"] and {"PREVIEW", "NOW"} <= set(vm["labels"]), vm
    assert vm["tiles"] > 0 and vm["heat"] and all(vm["heat"]), vm
    for legends in (r["legend"], vm["legend"]):
        assert all(lg["clear"] for views in legends.values() for lg in views), legends
        assert not legends["side 1366x768"][0]["open"] and legends["swipe 1920x1080"][0]["open"], legends
    m = r["metrics"]
    assert m["1366x768"] == m["1920x1080"] == [True, True] and "the numbers" in m["text"]
    b = r["search"]["badges"]
    recommended = [c for c in b["cards"] if "Recommended" in c]
    assert b["explained"] and len(recommended) == 1
    assert "Excellent match" in recommended[0] or not any("Excellent match" in c for c in b["cards"])
    assert "worldgen migration" in r["after"]
    assert r["console"]["stdout"][1:] == [server.KEEP_OPEN] and r["console"]["stderr"] == []
    new_terrain = re.compile(rf"^({'|'.join(NAMES)}), ground at y \d+$", re.M)
    assert new_terrain.search(r["readout"]["candidate"]) and not new_terrain.search(r["readout"]["now"]), r["readout"]
