import json
from dataclasses import asdict

import numpy as np
import pytest
from conftest import REGISTRY, make_world, terrain

from rwg2migrator import Refusal, appearance, ground, maps, scan
from rwg2migrator.provision import data

POC_SHARES = {"grass": 0.9223, "dirt": 0.0285, "rock": 0.0274, "sand": 0.0005, "mesa": 0.0003, "snow": 0.017}
POC_SHARES |= {"mud": 0.0003, "marsh": 0.0025, "lake": 0.0002, "other": 0.0012}
POC_REF = appearance.Reference("Temperate Forest", 236, (0x4C, 0x6E, 0x31), POC_SHARES, 70, 459317)
POC_BANS = {"sand", "snow", "wet", "water", "other"}
TAB = appearance.table("2.9.0")
TUNDRA = 115


def test_table_covers_every_registered_biome_and_every_poc_palette():
    registered = json.loads(data("biomes-2.9.0.json").read_text())["biomes"]
    assert {appearance.fold(b["name"]) for b in registered} <= set(TAB)
    for b in registered:
        e = TAB[appearance.fold(b["name"])]
        assert any(b["id"] in v["ids"] for v in e["variants"]) and all(v["grass"][0][0] == 0 for v in e["variants"])
    assert sum("sampled" in e for e in TAB.values()) == 75
    assert len(TAB["oasis"]["variants"]) == 2


def test_temperate_forest_colour_at_the_fixture_median_height():
    got = appearance.colour_at(TAB["temperate forest"], 70, 236)
    assert np.abs(got - appearance.rgb("#4C6E31")).max() <= 1


def test_poc_categories_and_incompatibility():
    want = {
        "Garden": "green",
        "Hot Desert": "sand",
        "Snow Forest": "snow",
        "Land Of Lakes Marsh": "wet",
        "Deep Ocean": "water",
        "Tainted Land": "other",
        "Deciduous Forest": "dull",
    }
    bans = appearance.default_bans(POC_REF)
    assert bans == POC_BANS
    assert {n: appearance.judge(n, POC_REF, bans, TAB).look for n in want} == want
    assert appearance.judge("Hot Desert", POC_REF, bans, TAB).incompat == 1
    assert appearance.incompat("green", 0.7, bans) == 0 and appearance.incompat("dull", 0.3, bans) == pytest.approx(0.5)
    tf = appearance.judge("Temperate Forest", POC_REF, bans, TAB)
    assert tf.same and tf.incompat == 0
    assert not appearance.judge("Jade Cliffs", POC_REF, bans, TAB).same
    unknown = appearance.judge("Some Swamp Nobody Registered", POC_REF, bans, TAB)
    assert (unknown.look, unknown.incompat) == ("wet", 1.0)


def test_reference_scores_one_against_itself():
    assert appearance.score(POC_SHARES, np.array(POC_REF.colour), POC_REF) == 1.0
    assert POC_REF.surface == pytest.approx(0.95025) and POC_REF.bad == pytest.approx(0.0206)


def test_tundra_looks_like_snow_on_the_now_map_as_rwg2s_tundra_does_on_the_candidate():
    tops = np.array([ground.TOPS.index("grass")] * 14 + [ground.TOPS.index("rock"), ground.TOPS.index("snow")], np.uint8)
    cols = {"biome": np.full(len(tops), TUNDRA), "ground": np.full(len(tops), 70.0), "top": tops, "water": np.zeros(len(tops), np.uint8)}
    ((_, _, now),) = maps.biome_looks(cols, POC_REF, "2.9.0")
    assert appearance.LOOKS[now] == appearance.judge("Tundra", POC_REF, POC_BANS, TAB).look == "snow"


def test_snowy_base_does_not_ban_snow():
    snowy = {k: 0.0 for k in appearance.SURFACES} | {"snow": 0.6, "grass": 0.3, "rock": 0.1}
    entry = TAB["snow forest"]
    ref = appearance.Reference("Snow Forest", 313, tuple(appearance.colour_at(entry, 70)), snowy, 70, 1000, "snow")
    assert appearance.look_of("Snow Forest", snowy, False, 1.0) == "snow"
    bans = appearance.default_bans(ref)
    assert "snow" not in bans and bans == {"sand", "wet", "water", "other"}
    assert appearance.judge("Snow Forest", ref, bans, TAB).look == "snow"
    assert appearance.judge("Snow Forest", ref, bans, TAB).incompat < 1


def test_shares_split_water_into_marsh_and_lake():
    top = np.array([ground.TOPS.index("grass")] * 6 + [0, 0], np.uint8)
    water = np.array([0, 0, 0, 0, 1, 3, 0, 0], np.uint8)
    sh = appearance.shares(top, water)
    assert sh["grass"] == 0.6667 and sh["marsh"] == 0.1667 and sh["lake"] == 0.1667


def test_prefs(tmp_path):
    assert appearance.load_prefs(None) == {"base": None, "banned": None, "tolerance": 1.0}
    p = tmp_path / "prefs.json"
    p.write_text(json.dumps({"banned": ["sand", "dull"]}))
    prefs = appearance.load_prefs(str(p))
    assert appearance.bans_for(POC_REF, prefs) == {"sand"}
    p.write_text(json.dumps({"colour": "green"}))
    with pytest.raises(Refusal, match="unknown entries: colour"):
        appearance.load_prefs(str(p))
    p.write_text(json.dumps({"banned": ["lava"]}))
    with pytest.raises(Refusal, match="lava"):
        appearance.load_prefs(str(p))


def test_reference_on_a_small_world(tmp_path, cache):
    w = make_world(str(tmp_path / "s"), chunk_specs={(x, z): {} for x in range(8) for z in range(8)})
    ref = appearance.reference(scan.scan(w, cache, workers=1), {(3, 3)}, "2.9.0")
    assert (ref.name, ref.biome_id, ref.look, ref.median_y) == ("Forest", 4, "green", 63)
    assert ref.shares["grass"] == 1.0 and ref.columns == 63 * 256
    assert ref.colour == tuple(appearance.colour_at(TAB["forest"], 63, 4))
    with pytest.raises(Refusal, match="not a biome"):
        appearance.reference(scan.scan(w, cache, workers=1), {(3, 3)}, "2.9.0", "Moon Base")


def _base(tmp_path, cache, ids):
    w = make_world(str(tmp_path / "s"), chunk_specs={(x, z): {"ids": ids} for x in range(8) for z in range(8)})
    return appearance.reference(scan.scan(w, cache, workers=1), {(3, 3)}, "2.9.0")


def test_a_sand_base_is_measured_and_does_not_ban_sand(tmp_path, cache):
    ids = terrain()
    ids[63], ids[64] = next(i for i, n in REGISTRY.items() if n == "minecraft:sand"), 0
    ref = _base(tmp_path, cache, ids)
    assert (ref.look, ref.shares["sand"], ref.surface) == ("sand", 1.0, 0.0)
    assert appearance.default_bans(ref) == {"other", "snow", "water", "wet"}
    assert appearance.judge("Hot Desert", ref, appearance.default_bans(ref), TAB).incompat < 1


def test_a_snow_covered_base_is_measured_and_does_not_ban_snow(tmp_path, cache):
    ids = terrain()
    ids[64] = next(i for i, n in REGISTRY.items() if n == "minecraft:snow_layer")
    ref = _base(tmp_path, cache, ids)
    assert (ref.look, ref.shares["snow"], ref.surface) == ("snow", 1.0, 0.0)
    assert "snow" not in appearance.default_bans(ref)


def test_a_reference_reads_back_from_its_saved_json():
    ref = appearance.Reference("Plains", 1, (1, 2, 3), {k: 0.0 for k in appearance.SURFACES} | {"grass": 1.0}, 64, 10)
    again = appearance.Reference.from_dict(json.loads(json.dumps(asdict(ref))))
    assert again == ref and again.colour == (1, 2, 3)
