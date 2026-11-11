import glob
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
from conftest import golden, world_scene

from rwg2migrator import appearance, area, chunks, cli, maps, nbt, region, scan, score, world

FIXTURE = os.environ.get("RWG2_FIXTURE")
FACTS = "world-facts.json"
ROUND_TRIP_MIN = 1000
pytestmark = [pytest.mark.fixture, pytest.mark.skipif(not FIXTURE, reason="set RWG2_FIXTURE to a COPY of the pre-RWG2 server tree")]


@pytest.fixture(scope="module")
def world_dir():
    return os.path.join(FIXTURE, "World")


@pytest.fixture(scope="module")
def world_facts():
    """The test world's seed, graves, places and extra mods, kept with its golden files outside the repo."""
    return json.loads(golden(FACTS).read_text(encoding="utf-8"))


def only_team(world_dir):
    (core,) = world.claims(world_dir).values()
    return core


@pytest.fixture(scope="module")
def scanned(world_dir):
    scan.scan(world_dir, scan.cache_root())
    return scan.scan(world_dir, scan.cache_root())


def test_inspect(capsys, scanned, world_facts):
    assert cli.main(["inspect", FIXTURE, "--json"]) == 0
    r = json.loads(capsys.readouterr().out)
    assert (r["gtnh"], r["dreamcraft"], r["generator"], r["seed"]) == ("2.9.0", "2.9.99", "RWG", world_facts["seed"])
    assert r["rwg_data"] == {"biomeListVersion": 1, "noiseImplementation": "UNKNOWN"}
    assert r["mods"] == 297 and sorted(r["mod_diff"]["beyond_pack"]) == world_facts["mods_beyond_pack"]
    assert list(r["claims"].values()) == [908]
    s = r["scan"]
    assert s["chunks"] == 72486 and s["formats"] == {"EndlessIDs": 3957, "NEID": 68529}
    assert s["unpopulated"] == 1696 and s["unreadable"] == 0
    assert s["regions_read"] == 0 and s["regions_cached"] == s["regions"]


def test_area_matches_golden_kept(capsys, tmp_path, scanned):
    out = tmp_path / "area.json"
    assert cli.main(["area", FIXTURE, "--claims", "all", "--margin", "1", "--out", str(out)]) == 0
    a = json.loads(out.read_text())
    pin = json.loads(golden("kept-green1.json").read_text())
    assert len(a["core"]) == pin["claims"] == 908 and len(a["kept"]) == pin["kept_chunks"] == 1090
    assert sorted(map(tuple, a["kept"])) == sorted(map(tuple, pin["kept"]))
    st = a["status"]
    assert st["never_generated"] == [] and st["unpopulated_kept"] == [] and st["unreadable_kept"] == []
    assert st["dropped"] == pin["dropped_chunks"] == 71396
    assert "will be removed" in capsys.readouterr().out.lower()


def _reencoded(level):
    fmt = chunks.chunk_format(level)
    if "Sections" in level:
        level["Sections"][1][1] = [chunks.encode_section(fmt, s["Y"][1], chunks.decode_section(s), like=s) for s in chunks.sections(level)]
    key = chunks.BIOME_KEY[fmt]
    level[key][1] = chunks.encode_biomes(fmt, chunks.biomes(level))
    return fmt


def _regions_by_interest(world_dir):
    kept = json.loads(golden("kept-green1.json").read_text())["kept"]
    first = {f"r.{x >> 5}.{z >> 5}.mca" for x, z in kept}
    paths = sorted(glob.glob(os.path.join(world_dir, "region", "*.mca")))
    return sorted(paths, key=lambda p: os.path.basename(p) not in first)


def test_sections_round_trip_byte_identical(world_dir):
    counts = Counter()
    for path in _regions_by_interest(world_dir):
        if min(counts[f] for f in chunks.FORMATS) >= ROUND_TRIP_MIN:
            break
        for rec, _ in region.records(Path(path).read_bytes()).values():
            raw = region.inflate(rec)
            name, root = nbt.parse(raw)
            assert nbt.write(name, root) == raw
            counts[_reencoded(root["Level"][1])] += 1
            assert nbt.write(name, root) == raw
    assert min(counts[f] for f in chunks.FORMATS) >= ROUND_TRIP_MIN, counts
    print("round trip byte-identical:", dict(counts))


def test_region_build_and_check_on_real_files(world_dir):
    for path in _regions_by_interest(world_dir)[:3]:
        data = Path(path).read_bytes()
        kept = region.records(data)
        assert region.check(data, kept) is None
        built = region.build(kept)
        assert region.check(built, kept) is None and region.build(region.records(built)) == built


def test_ground_equals_poc_on_the_zone(scanned):
    g = np.load(golden("poc-zone-ground.npz"))
    x0, z0, poc = int(g["x0"]), int(g["z0"]), g["ground"].astype(np.float32)
    poc[poc == 0] = np.nan
    mine = scanned.grid("ground", (x0, z0, x0 + max(poc.shape), z0 + max(poc.shape)))[: poc.shape[0], : poc.shape[1]]
    zone = np.zeros(poc.shape, bool)
    for cx, cz in g["zone"]:
        zone[cz * 16 - z0 : cz * 16 - z0 + 16, cx * 16 - x0 : cx * 16 - x0 + 16] = True
    a, b = mine[zone], poc[zone]
    assert zone.sum() == 4266 * 256 and np.array_equal(a, b, equal_nan=True)


POC_BANS = {"sand", "snow", "wet", "water", "other"}
POC_COLOUR = "#4C6E31"
SURFACE_TOLERANCE = 0.005


def test_reference_look(world_dir, scanned):
    core = only_team(world_dir)
    ref = appearance.reference(scanned, core, "2.9.0")
    assert (ref.name, ref.biome_id, ref.look) == ("Temperate Forest", 236, "green")
    assert np.abs(np.array(ref.colour) - appearance.rgb(POC_COLOUR)).max() <= 1
    assert abs(ref.surface - 0.950) <= SURFACE_TOLERANCE
    assert appearance.default_bans(ref) == POC_BANS


def test_far_apart_parts_get_two_windows(tmp_path, scanned, world_facts):
    far, out = world_facts["far_part"], tmp_path / "far.json"
    assert cli.main(["area", FIXTURE, "--claims", "all", "--add", far["add"], "--margin", "1", "--out", str(out)]) == 0
    a = json.loads(out.read_text())
    a["core"], a["kept"] = [tuple(c) for c in a["core"]], [tuple(c) for c in a["kept"]]
    assert len(area.parts(set(a["kept"]))) == 2
    facts = maps.draw(world_scene(scanned, a, "2.9.0"), None, str(tmp_path / "far.png"))
    assert [w["window"] for w in facts["windows"]] == far["windows"]


YOURS_MAX = 180
YOURS_KINDS = {"microblocks", "crucibles"}


def test_probably_yours_is_small_graves_are_named_and_village_workshops_are_world_generation(world_dir, scanned, world_facts):
    claims, level, t = world.claims(world_dir), world.read_level(world_dir), scanned.chunk_table()
    found, _ = world.players(world_dir, FIXTURE, level)
    a, scene = area.build(claims, set(), set(), 1), area.scene(scanned, claims, found, world.spawn(level))
    rm, cells = area.removed(a, t, scene), area.removed_cells(a, t, scene)
    graves, crucible, west_village = world_facts["graves"], tuple(world_facts["crucible"]), tuple(world_facts["west_village"])
    worldgen = {tuple(c) for c in world_facts["workshops"] + world_facts["dungeons"]}
    assert rm["graves"] == graves and len(cells["graves"]) == len(graves)
    assert rm["yours"]["count"] <= YOURS_MAX and set(rm["yours"]["named"]) <= YOURS_KINDS
    assert not worldgen & set(cells["yours"]) and crucible in cells["yours"]
    outside = [(cx, cz, t, n) for cx, cz, t, n in scene["tile_entities"] if (cx, cz) not in set(a["kept"])]
    natural = sum(n for *_, t, n in outside if t in area.NATURAL_TES)
    assert rm["yours"]["count"] + rm["beside"]["count"] + rm["worldgen"]["count"] + len(graves) + natural == sum(n for *_, n in outside)
    village = sum(n for cx, cz, t, n in outside if (cx, cz) == west_village and t == "Chest")
    assert village and rm["beside"]["types"]["Chest"] >= village


def test_site_equals_the_pinned_green1_site(world_dir, scanned):
    core = only_team(world_dir)
    site = score.build_site(lambda p: scanned.at("ground", p), core, area.dilate(core, 1))
    pinned = score.Site.load(golden("green1-site.npz"))
    for f in ("h_in", "inner", "edge", "edge_idx", "grid", "grid_d", "land", "core", "kept"):
        assert np.array_equal(getattr(site, f), getattr(pinned, f)), f
    assert {k: len(v) for k, v in site.bands().items()} == {"edge": 2776, "1-32": 352, "33-128": 1083, "129-256": 1737, "257-512": 1160}
