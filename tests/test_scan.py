import errno
import os
from pathlib import Path

import numpy as np
import pytest
from conftest import LAZY_POS, SURFACE, chunk_root, lazy_chunk, make_world, string, tag, terrain, write_region

from rwg2migrator import Refusal, chunks, ground, nbt, region, scan, world


def test_ground_rule_names():
    ok = ground.ground_ids({
        1: "minecraft:stone",
        2: "minecraft:grass",
        31: "minecraft:tallgrass",
        78: "minecraft:snow_layer",
        98: "minecraft:stonebrick",
        9: "minecraft:water",
    })
    assert ok[[1, 2]].all() and not ok[[31, 78, 98, 9, 0]].any()


def test_column_ground_skips_plants_and_limits_depth():
    ok = np.zeros(ground.ID_SPACE, bool)
    ok[[1, 2]] = True
    ids = terrain()
    hm = np.full((16, 16), SURFACE + 1)
    g = ground.column_ground(ids, hm, ok)
    assert (g == SURFACE).all()
    ids[:, 0, 0] = 0
    ids[10, 0, 0] = 1
    hm[0, 0] = 100
    g = ground.column_ground(ids, hm, ok)
    assert np.isnan(g[0, 0]) and g[1, 1] == SURFACE


def test_top_look_and_water_depth():
    reg = {1: "minecraft:stone", 2: "minecraft:grass", 9: "minecraft:water", 12: "minecraft:sand", 31: "minecraft:tallgrass"}
    reg[78] = "minecraft:snow_layer"
    rule = scan.rule_for(reg)
    ids = terrain().astype(np.uint32)
    ids[SURFACE, 3, 3] = 78
    ids[SURFACE - 1, 0, 0] = 12
    ids[SURFACE : SURFACE + 3, 0, 0] = 9
    ids[:, 1, 1] = 0
    ids[SURFACE - 1, 2, 2] = 70000
    hm = np.full((16, 16), SURFACE + 3)
    kind, depth = ground.column_top(ids, hm, rule)
    names = np.array(ground.TOPS)
    assert names[kind[0, 0]] == "sand" and depth[0, 0] == 3 and names[kind[3, 3]] == "snow"
    assert names[kind[0, 1]] == "grass" and depth[0, 1] == 0
    assert names[kind[1, 1]] == "none" and names[kind[2, 2]] == "rock"
    assert [ground.TOPS[ground.top_kind(n)] for n in ("minecraft:stained_hardened_clay", "minecraft:snow", "minecraft:dirt")] == [
        "mesa",
        "snow",
        "dirt",
    ]


def test_ids_beyond_16_bits_and_beyond_the_registry():
    ok = ground.ground_ids({1: "minecraft:stone", 70000: "big:stone"})
    assert ok.size == 70001 and ok[[1, 70000]].all()
    ids = terrain().astype(np.uint32)
    ids[SURFACE - 1] = 70000
    hm = np.full((16, 16), SURFACE + 1)
    assert (ground.column_ground(ids, hm, ok) == SURFACE).all()
    ids[SURFACE - 1] = 80000
    assert (ground.column_ground(ids, hm, ok) == SURFACE - 1).all()


def test_chunk_with_lazy_sections_scans(tmp_path, cache):
    w = make_world(str(tmp_path / "s"))
    rx, rz = (c // region.SIDE for c in LAZY_POS)
    write_region(os.path.join(w, "region", f"r.{rx}.{rz}.mca"), {LAZY_POS: lazy_chunk()[1]})
    sc = scan.scan(w, cache, workers=1)
    s = sc.summary()
    assert s["formats"] == {"EndlessIDs": 17} and s["unreadable"] == 0 and s["chunks"] == 17
    t = sc.chunk_table()
    assert t["populated"][(t["cx"] == LAZY_POS[0]) & (t["cz"] == LAZY_POS[1])].all()


def test_unknown_layout_counts_as_other(tmp_path, cache):
    root = chunk_root(0, 0)
    root["Level"][1]["Sections"][1][1][0]["Blocks16"] = [nbt.BYTE_ARRAY, bytes(2 * chunks.VOLUME)]
    w = make_world(str(tmp_path / "s"), chunk_specs={(1, 0): {}})
    write_region(os.path.join(w, "region", "r.0.0.mca"), {(0, 0): root, (1, 0): chunk_root(1, 0)})
    assert scan.scan(w, cache, workers=1).summary()["formats"] == {"EndlessIDs": 1, "other": 1}


def grave_te(name, stacks):
    return ("openblocks_grave", {"perishedUsername": string(name), "Items": tag(nbt.LIST, [nbt.COMPOUND, [{}] * stacks])})


def test_scan_keeps_owner_tags_and_graves(tmp_path, cache):
    machine = ("BaseMetaTileEntity", {"mOwnerName": string("Alice"), "mOwnerUuid": string("u-1"), "mName": string("x")})
    track = ("RailcraftTrackTESRTile", {"owner": string("[Railcraft]"), "ownerId": string("  ")})
    specs = {(0, 0): {"tes": (machine, "Chest")}, (1, 0): {"tes": (track,)}, (2, 0): {"tes": (grave_te("Bob", 3),)}}
    sc = scan.scan(make_world(str(tmp_path / "s"), chunk_specs=specs), cache, workers=1)
    assert sc.owners() == [
        (0, 0, "BaseMetaTileEntity", "Alice"),
        (0, 0, "BaseMetaTileEntity", "u-1"),
        (1, 0, "RailcraftTrackTESRTile", "[Railcraft]"),
    ]
    assert sc.graves() == [{"chunk": (2, 0), "name": "Bob", "x": 32, "y": SURFACE, "z": 0, "stacks": 3}]


def test_scan_counts_cache_and_unreadable(tmp_path, cache):
    specs = {(0, 0): {}, (1, 0): {"fmt": chunks.NEID, "populated": 0}, (2, 0): {"inhabited": 99, "tes": ("Chest", "Chest")}}
    w = make_world(str(tmp_path / "s"), chunk_specs=specs)
    first = scan.scan(w, cache, workers=1)
    s = first.summary()
    assert s["chunks"] == 3 and s["formats"] == {"EndlessIDs": 2, "NEID": 1} and s["unpopulated"] == 1 and s["unreadable"] == 0
    assert (first.read, first.cached) == (1, 0)
    t = first.chunk_table()
    assert t["inhabited"][t["cx"] == 2][0] == 99
    assert first.tile_entities() == [(2, 0, "Chest", 2)]
    g = first.grid("ground", (-8, 0, 56, 64))
    assert np.isnan(g[:, :8]).all() and (g[:16, 8:56] == SURFACE).all() and np.isnan(g[16:]).all()
    assert (first.grid("biome", (0, 0, 16, 16)) == 4).all()
    assert (first.grid("top", (0, 0, 16, 16)) == ground.TOPS.index("grass")).all() and not first.grid("water", (0, 0, 16, 16)).any()
    assert first.at("ground", [[20, 3], [-1, 0], [5000, 5000]]).tolist()[0] == SURFACE
    assert np.isnan(first.at("ground", [[-1, 0], [5000, 5000]])).all()
    assert first.grid("ground", (0, 0, 32, 8)).shape == (8, 32)
    again = scan.scan(w, cache, workers=1)
    assert (again.read, again.cached) == (0, 1) and again.summary()["chunks"] == 3


def test_a_world_moved_aside_takes_its_scan_from_the_cache_of_the_folder_it_left(tmp_path, cache):
    """Apply renames the world to its backup: the backup's map needs no new scan, and a region changed since is read again."""
    w = make_world(str(tmp_path / "s" / "World"), chunk_specs={(0, 0): {}, (40, 0): {}})
    scan.scan(w, cache, workers=1)
    aside = str(tmp_path / "s" / "World-pre")
    os.rename(w, aside)
    write_region(os.path.join(aside, "region", "r.1.0.mca"), {(40, 0): chunk_root(40, 0), (41, 0): chunk_root(41, 0)})
    assert scan.adopt(aside, w, cache) == 1
    moved = scan.scan(aside, cache, workers=1)
    fresh = scan.scan(aside, str(tmp_path / "other-cache"), workers=1)
    assert (moved.read, moved.cached) == (1, 1) and moved.chunk_table()["cx"].tolist() == fresh.chunk_table()["cx"].tolist()
    assert 41 in moved.chunk_table()["cx"]


def test_changed_region_is_rescanned_and_bad_chunk_marked(tmp_path, cache):
    w = make_world(str(tmp_path / "s"), chunk_specs={(0, 0): {}})
    scan.scan(w, cache, workers=1)
    path = os.path.join(w, "region", "r.0.0.mca")
    roots = {(0, 0): chunk_root(0, 0), (3, 3): chunk_root(3, 3)}
    write_region(path, roots)
    with open(path, "rb") as f:
        data = bytearray(f.read())
    loc = int.from_bytes(data[4 * region.slot_index(3, 3) : 4 * region.slot_index(3, 3) + 3], "big")
    data[loc * region.SECTOR + 5 : loc * region.SECTOR + 9] = b"bad!"
    with open(path, "wb") as f:
        f.write(bytes(data))
    s = scan.scan(w, cache, workers=1)
    assert s.read == 1 and s.summary()["unreadable"] == 1 and s.summary()["chunks"] == 2


def test_rule_depends_on_registry():
    a = scan.rule_for({1: "minecraft:stone", 90: "minecraft:portal"})
    b = scan.rule_for({1: "minecraft:stone", 91: "minecraft:portal"})
    assert a.portal == 90 and a.digest != b.digest


def test_portal_blocks_counted(tmp_path, cache):
    w = make_world(str(tmp_path / "s"), chunk_specs={(0, 0): {"ids": terrain(portal=6)}})
    assert scan.scan(w, cache, workers=1).summary()["portal_blocks"] == 6
    assert world.registry(world.read_level(w))[90] == scan.PORTAL


def test_glob_characters_and_region_copies(tmp_path, cache):
    w = make_world(str(tmp_path / ("GTNH [2.9]" if os.name == "nt" else "GTNH [2.9] *")), chunk_specs={(0, 0): {}, (1, 0): {}})
    rd = os.path.join(w, "region")
    with open(os.path.join(rd, "r.0.0.mca"), "rb") as src, open(os.path.join(rd, "r.0.0 (copy).mca"), "wb") as dst:
        dst.write(src.read())
    s = scan.scan(w, cache, workers=1)
    assert s.summary()["chunks"] == 2 and list(s.regions) == [(0, 0)]


def test_damaged_cache_file_is_rescanned(tmp_path, cache):
    w = make_world(str(tmp_path / "s"), chunk_specs={(0, 0): {}, (1, 0): {}})
    npz = Path(scan.scan(w, cache, workers=1).regions[(0, 0)])
    for damage in (lambda b: b[: len(b) // 2], lambda b: b""):
        npz.write_bytes(damage(npz.read_bytes()))
        again = scan.scan(w, cache, workers=1)
        assert (again.read, again.summary()["chunks"]) == (1, 2)


def test_a_cache_drive_filling_up_during_the_scan_is_refused_in_plain_words(tmp_path, monkeypatch):
    make_world(str(tmp_path / "server"))
    w = str(tmp_path / "server" / "World")
    monkeypatch.setattr(scan, "_scan_all", lambda *a: (_ for _ in ()).throw(OSError(errno.ENOSPC, "No space left on device")))
    with pytest.raises(Refusal, match="filled up while scanning"):
        scan.scan(w, str(tmp_path / "cache"), workers=1)
