"""M5 on the synthetic world with the fake generator: gen-world, gen boot gating, harvest guards, smoothing, post-check, the
refresh that reuses the gen-world, and the far-apart and noise variants."""

import json
import os
import re
import shutil
import struct
import zipfile
from pathlib import Path

import fake_gen
import pytest
from conftest import OFFSETS, RWG2_CFG, Migration, add_migration_parts, age, level_dat, make_world, tag, tree, write_nbt

from rwg2migrator import (
    Refusal,
    apply,
    area,
    chunks,
    cli,
    generator,
    genworld,
    jobs,
    nbt,
    postcheck,
    region,
    smooth,
    staging,
    transplant,
    world,
)

RING = 3
CHEST_CHUNK = (2, 2)
COW_CHUNK = (1, 1)
COW = (11, 22)


def chunks_of(world_dir):
    out = {}
    for path in transplant.region_files(world_dir):
        rx, rz = region.coords(path)
        out.update({region.slot_pos(rx, rz, i): r for i, r in region.records(Path(path).read_bytes()).items()})
    return out


def level_of(rec):
    return nbt.parse(region.inflate(rec[0]))[1]["Level"][1]


def uuids(level):
    return {(e["UUIDMost"][1], e["UUIDLeast"][1]) for e in level.get("Entities", [0, [0, []]])[1][1] if "UUIDMost" in e}


def _with_cow(world_dir):
    """A cow with a known UUID in a protected chunk of the world."""
    path = os.path.join(world_dir, "region", "r.0.0.mca")
    recs = region.records(Path(path).read_bytes())
    i = region.slot_index(*COW_CHUNK)
    name, root = nbt.parse(region.inflate(recs[i][0]))
    pos = [struct.pack(">d", v) for v in (24.5, 65.0, 24.5)]
    cow = {"id": tag(nbt.STRING, "Cow"), "Pos": tag(nbt.LIST, [nbt.DOUBLE, pos]), "UUIDMost": tag(nbt.LONG, COW[0])}
    cow["UUIDLeast"] = tag(nbt.LONG, COW[1])
    root["Level"][1]["Entities"] = tag(nbt.LIST, [nbt.COMPOUND, [cow]])
    recs[i] = (region.deflate(nbt.write(name, root)), recs[i][1])
    Path(path).write_bytes(region.build(recs))


def setup(base, monkeypatch, **world_kw):
    root = base / "server"
    specs = {(x, z): ({"tes": ("Chest",)} if (x, z) == CHEST_CHUNK else {}) for x in range(4) for z in range(4)}
    make_world(str(root), chunk_specs=specs, **world_kw)
    add_migration_parts(str(root))
    _with_cow(str(root / "World"))
    age(str(root / "World"))
    fake_gen.use(monkeypatch)
    m = Migration(root, base)
    fake_gen.settings(m.cache)
    return m


@pytest.fixture
def gen(tmp_path, monkeypatch):
    return setup(tmp_path, monkeypatch)


def report(job):
    return jobs.read_json(os.path.join(job.folder, staging.POSTCHECK, staging.REPORT))


def area_of(m):
    a = json.loads(Path(m.area).read_text())
    return {tuple(c) for c in a["core"]}, {tuple(c) for c in a["kept"]}


def test_preview_generates_harvests_smooths_and_checks_the_working_copy(gen, capsys):
    code, job = gen.generate("--ring", RING)
    out = capsys.readouterr()
    assert code == 0, out.err
    core, kept = area_of(gen)
    st, src = job.data["args"]["staging"], chunks_of(gen.world)
    have, made = chunks_of(st), chunks_of(genworld.generated(job)["folder"])
    ring = area.dilate(kept, RING) - kept
    assert all(have[c] == src[c] for c in core)
    assert set(have) == kept | set(made) and ring <= set(have)
    assert all(level_of(have[c])["TerrainPopulated"][1] for c in area.dilate(kept, RING - 1) - kept)
    harvested = set(have) - kept
    assert uuids(level_of(have[COW_CHUNK])) == {COW} and not any(COW in uuids(level_of(have[c])) for c in harvested)
    w, s = tree(gen.world), tree(st)
    assert sorted(rel for rel in s if not rel.startswith("region/") and w.get(rel) != s[rel]) == ["level.dat"]
    assert staging.mismatches(st, staging.final(job)) == []
    assert fake_gen.FakeGens.boots == ["gen", "sampler"]
    p = report(job)
    assert p["new_terrain"]["median"] == 0 and p["new_terrain"]["within1_pct"] == 100
    assert p["seams"]["before"]["kept_border"]["max"] > 2 and p["seams"]["after"]["kept_border"]["max"] <= 2
    assert p["seams"]["after"]["kept_border"]["pairs"] == p["seams"]["before"]["kept_border"]["pairs"]
    assert p["core_columns_changed"] == 0 and p["core_identical"] == len(core) and p["columns_reshaped"] > 0
    smoothed = jobs.read_json(os.path.join(job.folder, smooth.SMOOTH, staging.RECORD))["removed"]
    assert p["smoothing_removed"] == {"entities": smoothed.get("Entities", 0), "tile_entities": smoothed.get("TileEntities", 0)}
    assert p["outside_overworld"]["changed_by_transplant"] == ["data/Stronghold.dat", "data/Village.dat", "level.dat"]
    h = jobs.read_json(os.path.join(job.folder, genworld.HARVEST, staging.RECORD))
    assert h["trees_clipped"]["logs"] > 0 and h["registry"]["entries"] == h["registry"]["shared"]
    assert "Seam at the kept border after smoothing" in out.out and "Trees cut at the kept border" in out.out
    assert re.search(r"Removed on the reshaped columns: \d+ animals or other entities, \d+ crops, flower pots or similar blocks", out.out)


def spilled_log(level):
    return fake_gen.LOG in chunks.block_ids(level)[:, 0, 0]


def decorated(rec):
    return bool(level_of(rec)["TerrainPopulated"][1])


def nothing_decorates_twice_or_loses_spill(job, kept):
    """Every decorated chunk of the gen-world outside kept is in staging, so the user's game never decorates it again, and so is
    every chunk it decorated into (+x, +z, +xz), so none of that decoration is lost. Returns (staging chunks, decorated ones)."""
    have, made = chunks_of(job.data["args"]["staging"]), chunks_of(genworld.generated(job)["folder"])
    deco = {c for c, rec in made.items() if c not in kept and decorated(rec)}
    assert deco <= set(have)
    fed = {(x + dx, z + dz) for x, z in deco for dx in (0, 1) for dz in (0, 1)} - kept
    assert fed <= set(have)
    smoothed = set(map(tuple, jobs.read_json(os.path.join(job.folder, "smooth", staging.RECORD))["chunks_changed"]))
    assert all(have[c] == made[c] for c in set(made) - kept - smoothed)
    return have, deco


def test_the_undecorated_outer_layer_is_harvested_with_what_its_neighbours_decorated_into_it(gen):
    """The game decorates a chunk's +x/+z neighbours too, and the applied world never decorates the inner chunks again: an outer
    chunk left for the user's server to regenerate would lose that decoration for good, a treeless strip along the ring."""
    _, job = gen.generate("--ring", RING)
    _, kept = area_of(gen)
    have, _ = nothing_decorates_twice_or_loses_spill(job, kept)
    bare = {c for c in set(have) - kept if not decorated(have[c])}
    fed = {c for c in bare if (c[0] - 1, c[1]) in have and decorated(have[(c[0] - 1, c[1])])}
    assert fed and all(spilled_log(level_of(have[c])) for c in fed)
    assert not bare & area.dilate(kept, RING - 1)
    h = jobs.read_json(os.path.join(job.folder, genworld.HARVEST, staging.RECORD))
    assert h["not_decorated"] == len(bare) and h["not_generated"] == 0
    assert h["harvested"] == len(set(have) - kept)


def test_the_spawn_area_past_the_ring_is_harvested_whole(gen, capsys):
    """A kept area under about 9x9 chunks: the gen boot's 25x25 spawn area reaches past the ring and decorates the ring's outer
    layer and chunks beyond it at boot. Left in the gen-world, those chunks would be decorated a second time by the user's game,
    and the outer layer's decoration into them would be lost (an 8-block strip with no trees about 128 blocks out)."""
    code, job = gen.generate("--ring", RING)
    assert code == 0
    _, kept = area_of(gen)
    spawn = genworld.spawn_area(kept)
    assert not spawn <= area.dilate(kept, RING), "the synthetic area must be small enough for the spawn area to reach past the ring"
    have, deco = nothing_decorates_twice_or_loses_spill(job, kept)
    outer = area.dilate(kept, RING) - area.dilate(kept, RING - 1)
    assert outer <= deco and spawn - kept <= set(have)
    h = jobs.read_json(os.path.join(job.folder, genworld.HARVEST, staging.RECORD))
    assert h["beyond_ring"] == len(set(have) - area.dilate(kept, RING)) > 0 and h["beyond_ring_decorated"] > 0
    p = report(job)
    assert p["new_terrain"]["columns"] == 256 * (len(deco)) and p["new_terrain"]["outside_render"] == 0
    assert f"{h['beyond_ring']:,} of them past the ring" in capsys.readouterr().out


def test_the_gen_world_holds_only_stripped_kept_chunks_with_the_spawn_at_their_centre(gen):
    _, job = gen.generate("--ring", RING)
    _, kept = area_of(gen)
    g = genworld.generated(job)
    assert os.path.basename(g["folder"]) == "gen-" + job.id and g["built"]["chunks"] == len(kept)
    assert g["built"]["tile_entity_blocks_cleared"] == 1
    gw = chunks_of(g["folder"])
    for c in kept:
        lv = level_of(gw[c])
        assert lv["TileEntities"][1][1] == [] and lv["Entities"][1][1] == []
    data = nbt.read_file(os.path.join(g["folder"], "level.dat"))["Data"]
    assert (data["SpawnX"], data["SpawnZ"]) == genworld.centre(kept) and "Player" not in data
    assert data["generatorName"] == "RWG2_CONTINENTS"
    assert sorted(os.listdir(os.path.join(g["folder"], "data"))) == [
        "GregTech_OregenPattern.dat",
        "Mineshaft.dat",
        "Temple.dat",
        "rwg_data.dat",
    ]
    assert not os.path.exists(os.path.join(g["folder"], "playerdata")) and not os.path.exists(os.path.join(g["folder"], "DIM-1"))


def test_an_unclean_generator_exit_counts_nothing_and_resume_builds_the_gen_world_again(gen, capsys):
    fake_gen.FakeGens.exit_code = 1
    code, job = gen.generate("--ring", RING)
    assert code == 1 and "exited with code 1" in capsys.readouterr().err
    assert genworld.generated(job) is None and job.state() == jobs.FAILED
    planted = Path(genworld.folder(gen.cache, fake_gen.RELEASE, job.id), "half-saved.txt")
    planted.write_text("left by the crashed boot")
    fake_gen.FakeGens.exit_code = 0
    assert cli.main(["jobs", "resume", job.id, *gen.opts()]) == 0
    assert not planted.exists() and genworld.generated(job)["exit_code"] == 0
    assert fake_gen.FakeGens.boots == ["gen", "gen", "sampler"]


def test_a_finished_preview_resumed_again_redoes_nothing(gen):
    _, job = gen.generate("--ring", RING)
    want = staging.final(job)
    job.update(state=jobs.INTERRUPTED)
    assert cli.main(["jobs", "resume", job.id, *gen.opts()]) == 0
    assert fake_gen.FakeGens.boots == ["gen", "sampler"] and staging.final(job) == want


def test_a_job_moved_with_its_cache_finds_its_gen_world_there(gen):
    """A job folder taken to another machine with its cache (a PC job resumed on Linux) uses the gen-world in the cache it names
    now, not the path recorded at generate time: the harvest and smoothing run again on it and nothing boots."""
    _, job = gen.generate("--ring", RING)
    want = staging.final(job)
    moved = Path(gen.cache).with_name("cache-moved")
    os.rename(gen.cache, moved)
    job.update(state=jobs.INTERRUPTED, args=job.data["args"] | {"cache": str(moved)})
    job.drop_step(genworld.HARVEST)
    assert cli.main(["jobs", "resume", job.id, "--data", gen.data, "--cache", str(moved), *gen.workers]) == 0
    assert fake_gen.FakeGens.boots == ["gen", "sampler"] and staging.final(job) == want


def _play(m):
    """A simulated play session: a protected chunk saved again and a player moved, both inside the kept area."""
    path = os.path.join(m.world, "region", "r.0.0.mca")
    recs = region.records(Path(path).read_bytes())
    i = region.slot_index(1, 1)
    name, root = nbt.parse(region.inflate(recs[i][0]))
    root["Level"][1]["InhabitedTime"] = [nbt.LONG, 4242]
    recs[i] = (region.deflate(nbt.write(name, root)), recs[i][1] + 60)
    Path(path).write_bytes(region.build(recs))
    from conftest import player_dat

    write_nbt(os.path.join(m.world, "playerdata", "u1.dat"), player_dat(30, 20))
    age(m.world)


def test_apply_refresh_after_play_reuses_the_gen_world_and_the_render(gen, capsys):
    _, job = gen.generate("--ring", RING)
    before, gen0 = report(job), genworld.generated(job)
    _play(gen)
    capsys.readouterr()
    assert gen.apply(job) == 0
    err = capsys.readouterr().err
    assert "copied 3 changed files again and rebuilt the kept area, the new terrain and the smoothing in" in err
    assert fake_gen.FakeGens.boots == ["gen", "sampler"] and genworld.generated(job) == gen0
    after = report(job)
    assert {k: v for k, v in after.items() if k != "core_identical"} == {k: v for k, v in before.items() if k != "core_identical"}
    played = level_of(chunks_of(gen.world)[(1, 1)])
    assert played["InhabitedTime"][1] == 4242
    assert not os.path.exists(gen0["folder"])


def test_a_config_changed_before_apply_is_refused_and_put_back_reuses_the_generated_terrain(gen, capsys):
    _, job = gen.generate("--ring", RING)
    extra = Path(gen.root, "config", "extra.cfg")
    extra.write_text("general {\n    B:changed=true\n}\n")
    capsys.readouterr()
    assert gen.apply(job) == 1
    err = capsys.readouterr().err
    assert "changed since preview" in err and "(extra.cfg added)" in err and "generating the new terrain" not in err, err
    extra.unlink()
    assert gen.apply(job) == 0 and "generating the new terrain" not in capsys.readouterr().err
    assert fake_gen.FakeGens.boots == ["gen", "sampler"]


def test_a_newer_preview_deletes_the_older_ones_gen_world(gen):
    _, first = gen.generate("--ring", RING)
    folder = genworld.generated(first)["folder"]
    _, second = gen.generate("--ring", RING)
    assert first.state() == jobs.SUPERSEDED and not os.path.exists(folder) and os.path.isdir(genworld.generated(second)["folder"])


def test_opensimplex_noise_deletes_rwg_data_in_staging_and_the_gen_world(gen):
    noise = {"data": [nbt.COMPOUND, {"noiseImplementation": [nbt.STRING, "OPENSIMPLEX"], "biomeListVersion": [nbt.INT, 1]}]}
    write_nbt(os.path.join(gen.world, "data", "rwg_data.dat"), noise)
    age(gen.world)
    code, job = gen.generate("--ring", RING)
    assert code == 0
    t = jobs.read_json(os.path.join(job.folder, staging.TRANSPLANT, staging.RECORD))
    g = genworld.generated(job)
    assert t["rwg_data"] == "deleted" and not os.path.exists(os.path.join(g["folder"], "data", "rwg_data.dat"))
    assert g["noise"] in world.PERLIN


def test_no_smooth_keeps_the_step_and_reports_no_smoothing(gen, capsys):
    code, job = gen.generate("--ring", RING, "--no-smooth")
    assert code == 0
    p = report(job)
    assert p["seams"]["before"] == p["seams"]["after"] and p["columns_reshaped"] == 0
    assert "after smoothing" not in capsys.readouterr().out


def test_smoothing_refuses_without_the_free_memory_it_needs_and_resume_finishes_it(gen, capsys, monkeypatch):
    monkeypatch.setattr(generator, "free_ram", lambda: 300 << 20)
    code, job = gen.generate("--ring", RING)
    err = capsys.readouterr().err
    assert code == 1 and "Smoothing the border needs 1." in err and "GB free; 0.3 GB is free now" in err, err
    assert f"then continue the preview: rwg2-migrator jobs resume {job.id}" in err
    assert genworld.generated(job) is not None and jobs.read_json(os.path.join(job.folder, "smooth", staging.RECORD)) is None
    monkeypatch.setattr(generator, "free_ram", lambda: 64 * generator.GB)
    assert cli.main(["jobs", "resume", job.id, *gen.opts()]) == 0
    assert report(job)["columns_reshaped"] > 0 and fake_gen.FakeGens.boots == ["gen", "sampler"]


def test_a_ring_too_small_to_decorate_the_border_is_refused(gen, capsys):
    code, job = gen.generate("--ring", 2)
    assert code == 1 and job is None and "--ring 3 or more" in capsys.readouterr().err


def _level(folder, items, fluids=()):
    entries = [{"K": [nbt.STRING, k], "V": [nbt.INT, v]} for k, v in items.items()]
    fml = {"ItemData": [nbt.LIST, [nbt.COMPOUND, entries]]}
    forge = {"DefaultFluidList": [nbt.LIST, [nbt.STRING, list(fluids)]]}
    write_nbt(os.path.join(folder, "level.dat"), {"FML": [nbt.COMPOUND, fml], "Forge": [nbt.COMPOUND, forge]})


def test_the_registry_guard(tmp_path):
    user, gen = tmp_path / "user", tmp_path / "gen"
    items = {"\x01minecraft:stone": 1, "\x02minecraft:stick": 280, "\x01extramod:block": 4000}
    _level(str(user), items, ["water", "lava"])
    _level(str(gen), {k: v for k, v in items.items() if "extramod" not in k}, ["water"])
    assert genworld.registry_check(str(user), str(gen)) == {"entries": 2, "shared": 2, "fluids": 1}
    _level(str(gen), {"\x01minecraft:stone": 2})
    with pytest.raises(Refusal, match=r"differ from your world's in 1 entries and 0 fluids \(the first 'minecraft:stone'\)"):
        genworld.registry_check(str(user), str(gen))
    _level(str(gen), {"\x01minecraft:stone": 1}, ["honey"])
    with pytest.raises(Refusal, match="0 entries and 1 fluids") as e:
        genworld.registry_check(str(user), str(gen))
    assert e.value.action.startswith("Install exactly the GTNH release your world uses")


def test_ids_that_differ_name_the_pack_mods_the_world_was_opened_without(tmp_path):
    user, gen = tmp_path / "user", tmp_path / "gen"
    level = level_dat()
    mod_list = level["FML"][1]["ModList"][1]
    gone = sorted(m["ModId"][1] for m in mod_list[1] if m["ModId"][1] not in ("dreamcraft", "RWG", "rwg2"))[:2]
    mod_list[1] = [m for m in mod_list[1] if m["ModId"][1] not in gone]
    write_nbt(str(user / "level.dat"), level)
    _level(str(gen), {"\x01minecraft:stone": 1, "\x01removedmod:block": 4001})
    with pytest.raises(Refusal, match="differ from your world's in 1 entries") as e:
        genworld.registry_check(str(user), str(gen))
    assert e.value.action.startswith(f"Your world was last opened without these mods of the GTNH pack: {', '.join(gone)}. Open it once")


def test_the_coverage_guard_names_an_undecorated_chunk_next_to_kept():
    a = {"kept": [(0, 0)]}
    taken = dict.fromkeys(area.dilate({(0, 0)}, genworld.COVERAGE) - {(0, 0), (2, 2)})
    with pytest.raises(Refusal, match="1 chunk within 2 chunks of the kept area was not generated and decorated: 2,2"):
        genworld._coverage(a, set(), taken)
    assert genworld._coverage(a, set(), taken | {(2, 2): None}) == len(area.dilate({(0, 0)}, 2)) - 1


def test_seams_and_region_problems(tmp_path):
    inside, outside = postcheck.np.full((16, 16), 64.0), postcheck.np.full((16, 16), 70.0)
    cols = postcheck.Columns({(0, 0): inside} | {(x, z): outside for x in (-1, 0, 1) for z in (-1, 0, 1) if (x, z) != (0, 0)})
    s = postcheck.seam(cols, {(0, 0)})
    assert s == {"pairs": 64, "median": 6.0, "p90": 6.0, "max": 6.0, "over10_pct": 0.0, "over30_pct": 0.0}
    folder = tmp_path / "w" / "region"
    folder.mkdir(parents=True)
    rec = region.deflate(b"\x0a\x00\x00\x00")
    good = region.build({0: (rec, 1), 1: (rec, 1)})
    (folder / "r.0.0.mca").write_bytes(good)
    assert postcheck.region_problems(str(tmp_path / "w")) == []
    bad = bytearray(good)
    bad[4:8] = bad[0:4]
    (folder / "r.0.0.mca").write_bytes(bytes(bad))
    assert postcheck.region_problems(str(tmp_path / "w")) == ["r.0.0.mca: slot 1 has a bad or overlapping header entry"]


def _server_generates(world_dir, config, cells, ts=1800000000):
    """The user's server generates chunks beyond the harvested ring after Apply, with the given offsets."""
    by_region = {}
    for cx, cz in cells:
        by_region.setdefault((cx // region.SIDE, cz // region.SIDE), []).append((cx, cz))
    for (rx, rz), mine in by_region.items():
        path = Path(world_dir, "region", f"r.{rx}.{rz}.mca")
        recs = region.records(path.read_bytes()) if path.exists() else {}
        for cx, cz in mine:
            recs[region.slot_index(cx, cz)] = (region.deflate(nbt.write("", fake_gen.new_chunk(config, cx, cz))), ts)
        path.write_bytes(region.build(recs))


def _verify(m, capsys):
    capsys.readouterr()
    code = cli.main(["verify", m.root, *m.opts()])
    return code, capsys.readouterr().out


def test_verify_compares_the_chunks_generated_since_apply_with_the_preview(gen, capsys):
    """A few correct chunks can each be off the render (lakes, features it does not hold), so verify judges only a patch of
    touching chunks that are all off, and never counts the old terrain of a merged upload as new chunks."""
    _, job = gen.generate("--ring", RING)
    assert gen.apply(job) == 0
    config = job.data["args"]["config"]
    edge = max(c[0] for c in jobs.read_json(os.path.join(job.folder, genworld.HARVEST, staging.RECORD))["chunks"]) + 1
    _server_generates(gen.world, config, [(edge, 0), (edge, 1)])
    code, out = _verify(gen, capsys)
    assert code == 0 and "Chunks generated since Apply: 2" in out and "Too few to judge yet: 2 of them lie on the preview's map" in out
    assert "No problem found so far (the new chunks are too few to judge)." in out
    _server_generates(gen.world, config, [(edge + 1, z) for z in range(3)] + [(edge, 2)])
    code, out = _verify(gen, capsys)
    assert code == 0 and "Chunks generated since Apply: 6" in out, out
    assert "Game vs the preview there: median 0, 100% within 1 block (1,536 columns" in out and "0 of 6 chunks are off" in out
    assert "Too few" not in out and "Seam at the kept border now: median" in out and "No problem found." in out
    _server_generates(gen.world, [0, 0, 0, 0], [(region.SIDE + x, z) for x in range(2) for z in range(2)], ts=1600000000)
    code, out = _verify(gen, capsys)
    assert code == 2 and "Leftover: region/r.1.0.mca holds 4 chunks saved before Apply" in out
    assert "Chunks generated since Apply: 6" in out and "do not follow the preview" not in out
    Path(gen.world, "region", "r.1.0.mca").unlink()
    _server_generates(gen.world, [0, 0, 0, 0], [(edge + 2 + x, z) for x in range(2) for z in range(2)])
    code, out = _verify(gen, capsys)
    assert code == 2 and "4 touching chunks around x" in out and "do not follow the preview" in out and "No problem found." not in out


def standalone(base, monkeypatch):
    """A world folder downloaded alone: no server.properties beside it, no saves/ parent and no config folder."""
    root = base / "download"
    make_world(str(root))
    add_migration_parts(str(root))
    os.remove(root / "server.properties")
    shutil.rmtree(root / "config")
    fake_gen.use(monkeypatch)
    w = str(root / "World")
    m = Migration(w, base, world_dir=w, cand={"config": OFFSETS, "taint": {"blocks": 330.1, "point": [0, 0], "direction": "north"}})
    fake_gen.settings(m.cache)
    return m


def test_a_world_folder_alone_migrates_with_the_lines_to_add_by_hand_verifies_and_rolls_back(tmp_path, monkeypatch, capsys):
    m = standalone(tmp_path, monkeypatch)
    before, by_hand = tree(m.world), tmp_path / "download" / ("World" + apply.BY_HAND)
    code, job = m.generate("--ring", RING)
    assert code == 0 and "Note: no config folder of its own was found next to the world" in capsys.readouterr().err
    assert cli.main(["apply", m.world, "--preview", job.id, *m.opts()]) == 0
    assert "configs written: none, as no config folder was found" in capsys.readouterr().out
    assert m.apply(job) == 0
    out, rec = capsys.readouterr().out, json.loads(Path(m.world, apply.MARKER).read_text())
    assert (rec["config_dir"], rec["configs"], rec["by_hand"]) == (None, {}, {"file": by_hand.name, "taint": True})
    text = by_hand.read_text()
    assert text.startswith(f"Warning: {apply.NO_CONFIG}.") and f"Warning: {apply.NO_CONFIG}." in out and f"saved in {by_hand}." in out
    assert all(line in text and line in out for line in rec["offset_lines"] + [apply.TAINT_LINE, "'expert / worldgen migration'"])
    assert not (tmp_path / "download" / "config").exists()
    code, out = _verify(m, capsys)
    assert code == 0 and "Config: none of its own found next to the world, so the 4 offset lines were not checked" in out
    assert rec["offset_lines"][0] in out and apply.TAINT_LINE in out and "No other problem found" in out
    cfg = tmp_path / "download" / "config"
    cfg.mkdir()
    (cfg / "rwg2.cfg").write_text(RWG2_CFG)
    assert cli.main(["verify", m.world, "--restore-offsets", *m.opts()]) == 0
    assert rec["offset_lines"][0] in (cfg / "rwg2.cfg").read_text()
    capsys.readouterr()
    assert m.rollback() == 0 and "no config to restore (Apply wrote none)" in capsys.readouterr().out
    assert tree(m.world) == before and not by_hand.exists()


def test_an_upload_without_a_config_folder_says_where_the_lines_go(tmp_path, monkeypatch, capsys):
    m = standalone(tmp_path, monkeypatch)
    _, job = m.generate("--ring", RING)
    assert m.apply(job, "--remote", "--zip", "--confirm-stopped") == 0
    by_hand = tmp_path / "download" / ("World" + apply.BY_HAND)
    assert f"4. On the host, set the lines saved in {by_hand} in its config folder" in capsys.readouterr().out
    names = zipfile.ZipFile(Path(staging.work_dir(str(tmp_path / "download")), "World-upload.zip")).namelist()
    assert "World/level.dat" in names and not [n for n in names if n.startswith("config/")]


def test_an_apply_stopped_after_the_swap_finishes_without_the_config_folder_picked_since(tmp_path, monkeypatch, capsys):
    m = standalone(tmp_path, monkeypatch)
    _, job = m.generate("--ring", RING)
    real, calls = apply._swapped_stopped, []

    def started_once(ctx):
        calls.append(ctx)
        if len(calls) == 1:
            raise Refusal("the world was started", "Stop it")
        return real(ctx)

    monkeypatch.setattr(apply, "_swapped_stopped", started_once)
    assert m.apply(job) == 1
    mine = tmp_path / "mine"
    mine.mkdir()
    (mine / "rwg2.cfg").write_text(RWG2_CFG)
    capsys.readouterr()
    assert m.apply(job, "--config-dir", mine) == 1
    err = capsys.readouterr().err
    assert f"started without a config folder; it must finish that way, not with the config folder {mine}" in err
    applying = next(j for j in jobs.all_jobs(m.data) if j.data["kind"] == "apply")
    assert f"rwg2-migrator jobs resume {applying.id}" in err and (mine / "rwg2.cfg").read_text() == RWG2_CFG
    assert cli.main(["jobs", "resume", applying.id, *m.opts()]) == 0
    rec = json.loads(Path(m.world, apply.MARKER).read_text())
    assert (rec["config_dir"], rec["configs"], rec["by_hand"]["file"]) == (None, {}, "World" + apply.BY_HAND)
    assert f"Warning: {apply.NO_CONFIG}." in capsys.readouterr().out
    assert cli.main(["verify", m.world, "--config-dir", str(mine), "--restore-offsets", *m.opts()]) == 0
    assert rec["offset_lines"][0] in (mine / "rwg2.cfg").read_text()


def test_one_chunk_off_the_preview_is_not_a_mismatch():
    s = {"new_chunks": 9, "compared_chunks": 9, "off_chunks": 3, "off_cluster": postcheck.CLUSTER - 1}
    assert not postcheck.mismatch(s) and postcheck.mismatch(s | {"off_cluster": postcheck.CLUSTER}) and not postcheck.mismatch(None)


def test_kept_chunks_the_world_never_had_are_harvested_decorated(gen, tmp_path, capsys):
    out = tmp_path / "edge-area.json"
    assert (
        cli.main(["area", gen.root, "--claims", "all", "--add", "4,1:4,1", "--out", str(out), "--cache", gen.cache, "--workers", "1"]) == 0
    )
    from conftest import search_job

    gen.area, gen.search = str(out), search_job(gen.data, str(out), world.locate(gen.world).config)
    code, job = gen.generate("--ring", RING)
    assert code == 0, capsys.readouterr().err
    t, h = (jobs.read_json(os.path.join(job.folder, s, staging.RECORD)) for s in (staging.TRANSPLANT, genworld.HARVEST))
    never = {tuple(c) for c in t["never_generated"]}
    assert (4, 1) in never and {tuple(c) for c in h["never_generated_harvested"]} == never
    have = chunks_of(job.data["args"]["staging"])
    assert all(level_of(have[c])["TerrainPopulated"][1] for c in never)
    assert report(job)["core_identical"] == 2
