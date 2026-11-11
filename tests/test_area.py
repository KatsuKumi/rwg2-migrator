import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from rwg2migrator import Refusal, area, provision

DEV = Path(__file__).resolve().parents[1] / "dev"


def test_dilate_is_chebyshev():
    assert area.dilate({(0, 0)}, 1) == {(x, z) for x in (-1, 0, 1) for z in (-1, 0, 1)}
    assert area.dilate({(3, 4)}, 0) == {(3, 4)}


def test_parts_use_eight_neighbours():
    cells = {(0, 0), (1, 1), (5, 5), (5, 6), (5, 7)}
    assert area.parts(cells) == [{(5, 5), (5, 6), (5, 7)}, {(0, 0), (1, 1)}]


def test_windows_one_per_part_and_merge():
    near = area.windows({(0, 0), (40, 0)})
    assert near == [(-512, -512, 1168, 528)]
    far = area.windows({(0, 0), (400, 0)})
    assert far == [(-512, -512, 528, 528), (5888, -512, 6928, 528)]
    assert all(v % 16 == 0 for w in far for v in w)


def test_build_core_kept_and_hash():
    claims = {"a": {(0, 0), (1, 0)}, "b": {(10, 10)}}
    a = area.build(area.chosen_claims(claims, "a"), painted={(5, 5)}, erased={(1, 0)}, margin=1)
    assert a["core"] == [(0, 0), (5, 5)] and len(a["kept"]) == 18 and a["teams"] == ["a"] and a["erased_claims"] == [(1, 0)]
    assert a["claim_hash"] == area.claim_hash({(1, 0), (0, 0)})
    assert len(area.build(area.chosen_claims(claims, "all"), set(), set(), 2)["kept"]) == 25 + 6 * 5


def test_margin_zero_and_empty_refused():
    with pytest.raises(Refusal, match="margin of 0") as e:
        area.build({"a": {(0, 0)}}, set(), set(), 0)
    assert "--margin 1" in e.value.action
    with pytest.raises(Refusal, match="empty"):
        area.build({}, set(), set(), 1)
    with pytest.raises(Refusal, match="no claims for team zz"):
        area.chosen_claims({"a": set()}, "zz")


def test_rect_and_chunk_file(tmp_path):
    assert area.rect("2,3:1,4") == {(1, 3), (1, 4), (2, 3), (2, 4)}
    with pytest.raises(Refusal, match="not a chunk rectangle"):
        area.rect("1,2")
    lines = tmp_path / "c.txt"
    lines.write_text("1,2\n-3,4\n")
    js = tmp_path / "c.json"
    js.write_text(json.dumps({"core": [[1, 2]]}))
    assert area.chunk_file(lines) == {(1, 2), (-3, 4)} and area.chunk_file(js) == {(1, 2)}
    bad = tmp_path / "bad.txt"
    bad.write_text("hello")
    with pytest.raises(Refusal, match="does not hold chunk coordinates"):
        area.chunk_file(bad)


def _table(rows):
    cols = list(zip(*rows, strict=True))
    names = ("cx", "cz", "status", "populated", "inhabited", "portals")
    return {k: np.array(v) for k, v in zip(names, cols, strict=True)}


def test_status_and_removed():
    table = _table([(0, 0, 1, 1, 0, 0), (1, 0, 1, 0, 0, 0), (5, 5, 1, 1, 90000, 6), (9, 9, 2, 0, 0, 0)])
    a = area.build({"a": {(0, 0)}}, set(), set(), 1)
    st = area.status(a, table)
    assert st["core"] == 1 and st["kept"] == 9 and st["unpopulated_kept"] == [(1, 0)] and st["unpopulated_core"] == []
    assert len(st["never_generated"]) == 7 and st["dropped"] == 2 and st["parts"] == 1
    scene = {
        "tile_entities": [(0, 0, "Chest", 2), (5, 5, "Chest", 1), (5, 5, "GT_TileEntity_Ores", 40)],
        "owners": {},
        "graves": [],
        "claims": {"a": {(0, 0)}, "b": {(5, 5), (0, 1)}},
        "players": [
            {"name": "In", "dim": 0, "chunk": (0, 0), "bed_chunk": (5, 5)},
            {"name": "Out", "dim": 0, "chunk": (5, 5), "bed_chunk": None},
        ],
        "spawn": (8, 8),
    }
    rm = area.removed(a, table, scene)
    assert rm["yours"] == {"count": 0, "types": {}, "named": {}}
    assert rm["worldgen"] == {"count": 1, "types": {"Chest": 1}, "named": {"chests": 1}}
    assert rm["portal_blocks"] == 6 and rm["other_teams_claims"] == {"b": 1} and rm["players"] == ["Out"] and rm["beds"] == ["In"]
    assert rm["spawn_in_kept"]
    assert area.suggest(table) == {(5, 5)}


def _played(rows):
    """rows of (cx, cz, inhabited ticks)."""
    return _table([(x, z, 1, 1, t, 0) for x, z, t in rows])


def test_the_suggestion_keeps_the_part_with_the_claims_and_offers_the_others_with_size_and_distance():
    base = [(x, z, 100_000) for x in range(10, 14) for z in range(0, 3)]
    west = [(x, z, 900_000) for x in range(-30, -28) for z in range(0, 2)]
    gap_of_two = [(16, 0, 80_000)]
    table = _played([*base, *west, *gap_of_two, (0, 9, 10)])
    s = area.suggestion(table, {(12, 1)})
    assert s["main"] == sorted({(x, z) for x, z, _ in base} | {(16, 0)})
    assert s["extra"] == [{"cells": [(-30, 0), (-30, 1), (-29, 0), (-29, 1)], "chunks": 4, "blocks": 38 * 16, "direction": "west"}]
    assert area.suggestion(table, set())["main"] == [(-30, 0), (-30, 1), (-29, 0), (-29, 1)]
    assert area.suggestion(_played([(0, 0, 10)]), set()) == {"main": [], "extra": []}


def test_far_apart_kept_parts_carry_the_warning():
    table = _played([(0, 0, 0), (40, 0, 0)])
    one, two = (area.status(area.build({}, cells, set(), 1), table)["parts"] for cells in ({(0, 0), (2, 0)}, {(0, 0), (40, 0)}))
    assert (one, two) == (1, 2) and area.parts_warning(one) is None
    assert area.parts_warning(two) == f"The area to keep has 2 separate parts. {area.PARTS_WARNING}"
    assert area.PARTS_WARNING.startswith("One RWG2 setting has to fit every part.")


def test_probably_yours_rests_on_owner_tags_and_types_world_generation_never_makes():
    types, owners = area.worldgen()
    assert {"Chest", "MobSpawner", "Skull", "forestry.Swarm", "doorTileEntity", "PartCrafter", "ToolStation", "BlockSkyChest"} <= types
    assert "BaseMetaTileEntity" not in types and "openblocks_grave" not in types and "RCRollingMachineTile" not in types
    rows = [
        (0, 0, "Chest", 3),
        (1, 0, "Chest", 1),
        (1, 0, "BaseMetaTileEntity", 2),
        (1, 0, "GT_TileEntity_Ores", 50),
        (2, 0, "Chest", 1),
        (3, 0, "openblocks_grave", 1),
        (3, 0, "Chest", 1),
        (4, 0, "Chest", 1),
        (4, 0, "RCRollingMachineTile", 1),
        (9, 9, "BaseMetaTileEntity", 7),
    ]
    tags = [(2, 0, "RCCokeOvenTile", "5f0c2f6e-8a3b-4d1e-9c7a-2b6d4e8f1a03"), (4, 0, "RCRollingMachineTile", "[Railcraft]")]
    graves = [{"chunk": (3, 0), "name": "Alice", "x": 50, "y": 70, "z": 5, "stacks": 12}]
    scene = {"tile_entities": rows, "owners": tags, "graves": graves, "claims": {}, "players": [], "spawn": (0, 0)}
    a = area.build({}, {(9, 9)}, set(), 1)
    rm = area.removed(a, _table([(9, 9, 1, 1, 0, 0)]), scene)
    assert rm["yours"] == {"count": 2, "types": {"BaseMetaTileEntity": 2}, "named": {"GregTech machines": 2}}
    assert rm["beside"] == {"count": 3, "types": {"Chest": 3}, "named": {"chests": 3}}
    assert rm["worldgen"] == {"count": 5, "types": {"Chest": 4, "RCRollingMachineTile": 1}, "named": {"chests": 4, "RC rolling machine": 1}}
    assert rm["graves"] == [{"name": "Alice", "x": 50, "y": 70, "z": 5, "stacks": 12}]
    assert area.grave_text(rm["graves"][0]) == "Alice's grave at 50, 70, 5 (12 stacks of items)"
    cells = area.removed_cells(a, _table([(9, 9, 1, 1, 0, 0)]), scene)
    assert cells["yours"] == [(1, 0), (2, 0), (3, 0)] and cells["graves"] == [(3, 0)]


def test_a_chunk_with_a_players_sign_does_not_make_a_villages_chests_yours():
    rows = [(0, 0, "savedMultipart", 3), (0, 0, "Chest", 5), (0, 0, "Skull", 2), (0, 0, "GT_TileEntity_Ores", 9), (1, 0, "Chest", 4)]
    scene = {"tile_entities": rows, "owners": [], "graves": [], "claims": {}, "players": [], "spawn": (0, 0)}
    rm = area.removed(area.build({}, {(9, 9)}, set(), 1), _table([(9, 9, 1, 1, 0, 0)]), scene)
    assert (rm["yours"]["named"], rm["beside"]["named"], rm["worldgen"]["named"]) == (
        {"microblocks": 3},
        {"chests": 5, "skulls": 2},
        {"chests": 4},
    )


@pytest.mark.parametrize(
    ("te", "words"),
    [
        ("Chest", "chests"),
        ("logisticspipes.pipes.basic.LogisticsTileGenericPipe", "logistics generic pipe"),
        ("StorageDrawers:tileDrawersStandard", "drawers standard"),
        ("cropsnh:cropSticksTE", "crop sticks"),
        ("Tile", "Tile"),
    ],
)
def test_a_tile_entity_id_reads_as_plain_words(te, words):
    assert area.label(te) == words


@pytest.mark.parametrize(
    ("name", "is_player"),
    [
        ("Player_A", True),
        ("Gu_2", True),
        ("9e1b7c40-3f5a-4c2d-8e6b-0a9d2f4c7b15", True),
        ("[Railcraft]", False),
        ("Ancient Cultures", False),
        ("ab", False),
        ("a" * 17, False),
        ("Ünïcode", False),
    ],
)
def test_an_owner_is_a_player_when_it_can_be_a_minecraft_name_or_uuid(name, is_player):
    assert area.player(name, frozenset()) is is_player
    assert not area.player(name, frozenset({name}))


def test_structures_a_mods_code_places_count_as_world_generation_and_carry_their_evidence():
    structures = json.loads(provision.data(area.WORLDGEN_FILE).read_text(encoding="utf-8"))["structures"]
    assert {"etfuturum.brewing_stand", "gol_master", "ms_master", "sdk_master"} <= set(structures) <= area.worldgen()[0]
    assert all("(" in why and why.strip() for why in structures.values())
    rows = [(0, 0, "Chest", 7), (0, 0, "etfuturum.brewing_stand", 1), (1, 0, "gol_master", 1), (2, 0, "TileCrucible", 1)]
    scene = {"tile_entities": rows, "owners": [], "graves": [], "claims": {}, "players": [], "spawn": (0, 0)}
    assert area.player_chunks(scene) == {(2, 0)}


def test_the_sampler_keeps_the_structures_when_it_writes_the_file_again(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(DEV))
    import sample_worldgen

    out = tmp_path / area.WORLDGEN_FILE
    out.write_text(provision.data(area.WORLDGEN_FILE).read_text(encoding="utf-8"), encoding="utf-8")
    before = json.loads(out.read_text(encoding="utf-8"))
    monkeypatch.setattr(sample_worldgen, "sample", lambda w, cache: (Counter({"Chest": 2}), Counter(), 9))
    args = SimpleNamespace(worlds=["w"], label=["a square"], cache=str(tmp_path), out=str(out))
    assert sample_worldgen.collect(args) == 0
    after = json.loads(out.read_text(encoding="utf-8"))
    assert after["structures"] == before["structures"] and after["types"] == {"Chest": 2} and after["about"] == before["about"]
    out.unlink()
    assert sample_worldgen.collect(args) == 0 and json.loads(out.read_text(encoding="utf-8"))["structures"] == {}


def test_natural_tile_entities_are_types_the_generator_was_seen_to_make_but_mana_pods():
    assert area.NATURAL_TES - area.worldgen()[0] == {"TileManaPod"}
