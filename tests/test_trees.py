"""M6 tree replanting on the synthetic world with the fake generator: trees grow back on the reshaped strips, only logs and leaves
and only into air, an entity is dropped only where a log fills it, and a refresh after play in the margin replays the merge list
without a boot, while a border reshaped otherwise plants again."""

import json
import os
from pathlib import Path

import fake_gen
import numpy as np
import pytest
from conftest import REGISTRY, SURFACE, age, chunk_root, terrain
from test_blend import setup, voxels
from test_generate import area_of, report

from rwg2migrator import cli, genworld, jobs, nbt, plant, region, smooth, world

RING = 3
LOG, LEAVES = 17, 18
TREE_STEP = 4
TRUNK = range(64, 68)
CROWN = range(66, 69)
PLAYER_BLOCK = 90
DUG, BORDER_ROWS = 6, 4
FIRST_TREE = (slice(0, 3), slice(0, 3))


def forest():
    """The test terrain with a tree every TREE_STEP columns: a 4-log trunk with a 3x3 crown."""
    ids = terrain()
    for x in range(1, 16, TREE_STEP):
        for z in range(1, 16, TREE_STEP):
            for y in CROWN:
                crown = ids[y, max(z - 1, 0) : z + 2, max(x - 1, 0) : x + 2]
                crown[crown == 0] = LEAVES
            ids[TRUNK.start : TRUNK.stop, z, x] = LOG
    return ids


@pytest.fixture
def woods(tmp_path, monkeypatch):
    return setup(tmp_path, monkeypatch, ids=forest())


def test_trees_grow_back_on_the_strips_as_logs_and_leaves_into_air_only(woods, capsys):
    """The fake trees also turn the grass under them to dirt and put a snow layer on top: neither may reach staging."""
    code, plain = woods.generate("--ring", RING)
    assert code == 0
    before = voxels(plain.data["args"]["staging"])
    code, job = woods.generate("--ring", RING, "--trees")
    out = capsys.readouterr().out
    assert code == 0
    after = voxels(job.data["args"]["staging"])
    core, kept = area_of(woods)
    assert set(after) == set(before)
    k = smooth.classify(world.registry(world.read_level(job.data["args"]["staging"])))
    grown = {"log": 0, "leaves": 0}
    for pos, (ids, meta, biomes) in after.items():
        was = before[pos]
        changed = (ids != was[0]) | (meta != was[1])
        assert np.array_equal(biomes, was[2]), pos
        assert not (changed & (was[0] != 0)).any(), f"a voxel that was not air changed in {pos}"
        cat = smooth.lookup(k.cat, ids[changed])
        assert np.isin(cat, (smooth.LOG, smooth.LEAVES)).all(), f"something other than logs or leaves grew in {pos}"
        assert pos not in core or not changed.any(), f"protected chunk {pos} changed"
        grown["log"] += int(np.sum(cat == smooth.LOG))
        grown["leaves"] += int(np.sum(cat == smooth.LEAVES))
    t = report(job)["trees"]
    assert t["log_removed"] > 0 and t["log_in_strips"] >= 0.5 * t["log_removed"], t
    assert t["log_removed"] == jobs.read_json(os.path.join(job.folder, smooth.SMOOTH, "record.json"))["tree_log_voxels_removed"]
    assert grown == {"log": t["log"], "leaves": t["leaves"]} and t["chunks_changed_in_core"] == 0
    assert t["boot"]["trees"] > 0 and fake_gen.FakeGens.boots == ["gen", "sampler", "gen", "sampler", "gen"]
    assert f"Trees replanted: {t['boot']['trees']:,} trees" in out and "only into air" in out
    assert not os.path.exists(genworld.folder(woods.cache, fake_gen.RELEASE, job.id, genworld.PLANT_PREFIX))
    assert report(job)["core_identical"] == len(core) and kept


def _rewrite(world_dir, pos, ids=None, change=None):
    """Save chunk pos of the world again, as the game does: with the id stack ids (default: the forest it was made with), then
    change(level)."""
    path = Path(world_dir, "region", f"r.{pos[0] // 32}.{pos[1] // 32}.mca")
    recs = region.records(path.read_bytes())
    i = region.slot_index(*pos)
    root = chunk_root(*pos, spec={"ids": forest() if ids is None else ids})
    (change or (lambda level: None))(root["Level"][1])
    recs[i] = (region.deflate(nbt.write("", root)), recs[i][1] + 60)
    path.write_bytes(region.build(recs))
    age(world_dir)


def _merge_voxel_in_air(job, pos):
    """A voxel of chunk pos the merge list writes, which was air in the world before the preview too, as (y, z, x)."""
    idx = plant.load_merge(os.path.join(job.folder, plant.PLANT, plant.MERGE_FILE))[pos][0]
    old = forest().reshape(-1)
    return np.unravel_index(int(next(i for i in idx if old[i] == 0)), (256, 16, 16))


def _played(level):
    level["InhabitedTime"] = [nbt.LONG, 4242]
    level["LastUpdate"] = [nbt.LONG, 987654]
    level["Entities"] = [nbt.LIST, [nbt.COMPOUND, [fake_gen.pig(0, 0, 80)]]]


def test_play_in_the_margin_before_apply_replays_the_merge_list_without_a_boot(woods, capsys):
    """Play in the margin chunks the trees reach: each saved again (InhabitedTime, LastUpdate, an entity), a block placed where
    the list grows a leaf or a log, and a whole tree cut in a reshaped strip, which changes the logs smoothing removes there."""
    _, job = woods.generate("--ring", RING, "--trees")
    want, trees0 = voxels(job.data["args"]["staging"]), report(job)["trees"]
    boots, plant0 = list(fake_gen.FakeGens.boots), jobs.read_json(os.path.join(job.folder, plant.PLANT, jobs.MANIFEST_FILE))
    core, kept = area_of(woods)
    stage, a = job.data["args"]["staging"], jobs.read_json(os.path.join(job.folder, "area.json"))
    strips = plant.load_strips(os.path.join(job.folder, smooth.SMOOTH, plant.STRIPS))
    p = plant.plan(stage, a, strips)
    assert {(0, 0), (1, 0), (3, 0)} <= p.reach(core) & kept and {(0, 0), (3, 0)} <= {tuple(g) for g in p.groups}
    assert strips[(3, 0)][0][FIRST_TREE].all(), "the tree cut in (3, 0) must lie wholly in its reshaped strip"
    y, z, x = _merge_voxel_in_air(job, (0, 0))
    placed = forest()
    placed[y, z, x] = PLAYER_BLOCK
    _rewrite(woods.world, (0, 0), placed, _played)
    _rewrite(woods.world, (1, 0), change=_played)
    cut = forest()
    cut[(slice(TRUNK.start, CROWN.stop), *FIRST_TREE)] = 0
    _rewrite(woods.world, (3, 0), cut, _played)
    capsys.readouterr()
    assert woods.apply(job) == 0
    err = capsys.readouterr().err
    assert fake_gen.FakeGens.boots == boots, "the refresh booted the generator again"
    assert "replanting" not in err and "and replayed the trees' merge list in" in err
    assert jobs.read_json(os.path.join(job.folder, plant.PLANT, jobs.MANIFEST_FILE)) == plant0
    got = voxels(woods.world)
    assert set(got) == set(want)
    want[(0, 0)][0][y, z, x], want[(0, 0)][1][y, z, x] = PLAYER_BLOCK, 0
    for pos in got:
        assert all(np.array_equal(a, b) for a, b in zip(got[pos], want[pos], strict=True)), pos
    t = report(job)["trees"]
    grown = {k: t[k] - trees0[k] for k in ("log", "leaves", "skipped_not_air")}
    assert t["skipped_not_air"] == 1 and sum(grown.values()) == 0 and t["log_removed"] < trees0["log_removed"], t


def test_a_border_reshaped_otherwise_before_apply_plants_the_trees_again(woods, capsys):
    """Ground dug at the kept border moves the ramp outside kept, so the trees there stand on other ground: the list is not replayed."""
    _, job = woods.generate("--ring", RING, "--trees")
    boots = list(fake_gen.FakeGens.boots)
    dug = forest()
    dug[SURFACE - DUG : SURFACE, :BORDER_ROWS, :] = 0
    _rewrite(woods.world, (1, 0), dug)
    capsys.readouterr()
    assert woods.apply(job) == 0
    err = capsys.readouterr().err
    assert fake_gen.FakeGens.boots == [*boots, "gen"]
    assert "replanting trees" in err
    assert "and planted the trees again (one more start of the generator: the ground where they grow changed)" in err


def test_trees_need_the_smoothed_border(woods, capsys):
    for extra in (("--no-smooth", "--trees"), ("--no-generate", "--trees")):
        code, job = woods.generate(*extra)
        assert code == 1 and job is None
        assert "need the new terrain generated and smoothed" in capsys.readouterr().err


def area_of_reach():
    return {(x, z) for x in range(3) for z in range(-1, 2)} - {(0, 0)}


def test_spots_keep_clear_of_core_and_one_per_cell(tmp_path):
    core = {(0, 0)}
    d = plant.clearance((1, 0), core)
    assert d[0, 0] == 1 and d[0, 7] == 8 and d[0, 15] == d.max() == plant.CLEARANCE + 1
    ok = np.ones((16, 16), bool)
    picked = plant.candidates((1, 0), ok)
    assert len(picked) == (16 // plant.SPACING) ** 2 and picked == plant.candidates((1, 0), ok)
    cells = {(x // plant.SPACING, z // plant.SPACING) for x, z in picked}
    assert len(cells) == len(picked)
    strips = {(1, 0): (ok, 7), (2, 0): (ok, 0)}
    assert plant.Plan([], [], [[1, 0]]).reach(core) == area_of_reach()
    spot = [[20, 3, 4, 0]]
    assert plant.Plan(spot, [7], [[1, 0]], [65.0]).key() == plant.Plan(spot, [2], [[1, 0]], [65.0]).key()
    assert plant.Plan(spot, [7], [[1, 0]], [65.0]).key() != plant.Plan(spot, [7], [[1, 0]], [64.0]).key()
    merge = {(1, 0): (np.array([5, 9]), np.array([17, 18]), np.array([0, 4]))}
    plant.save_merge(str(tmp_path / "m.npz"), merge)
    back = plant.load_merge(str(tmp_path / "m.npz"))
    assert set(back) == {(1, 0)} and all(np.array_equal(a, b) for a, b in zip(back[(1, 0)], merge[(1, 0)], strict=True))
    plant.save_strips(str(tmp_path / "s.npz"), strips)
    assert {k: v[1] for k, v in plant.load_strips(str(tmp_path / "s.npz")).items()} == {(1, 0): 7, (2, 0): 0}


def test_the_merge_drops_only_an_entity_a_new_log_fills(tmp_path):
    root = chunk_root(2, 3)
    pigs = [fake_gen.pig(2, 3, ground) for ground in (69, 89)]
    pigs[1]["UUIDLeast"] = [nbt.LONG, 7]
    root["Level"][1]["Entities"] = [nbt.LIST, [nbt.COMPOUND, pigs]]
    ch = smooth.Chunk((2, 3), region.deflate(nbt.write("", root)), 1)
    reg = {**REGISTRY}
    k = smooth.classify(reg)
    trunk, crown = 70 * 256 + 8 * 16 + 8, 72 * 256 + 8 * 16 + 8
    rec, n = plant._apply(ch, (np.array([trunk, crown]), np.array([LOG, LEAVES]), np.array([0, 0])), None, k)
    level = nbt.parse(region.inflate(rec[0]))[1]["Level"][1]
    left = [e["UUIDLeast"][1] for e in level["Entities"][1][1]]
    assert left == [7] and n["log"] == 1 and n["leaves"] == 1 and n["dropped_entities"] == 1


def test_jobs_resume_after_a_crashed_plant_boot_plants_again(woods, capsys):
    fake_gen.FakeGens.exit_code = 0
    real = fake_gen.FakeGen.op_plant

    def crash(self, spots, budgets):
        fake_gen.FakeGens.exit_code = 1
        return real(self, spots, budgets)

    fake_gen.FakeGen.op_plant = crash
    try:
        code, job = woods.generate("--ring", RING, "--trees")
    finally:
        fake_gen.FakeGen.op_plant = real
    assert code == 1 and "exited with code 1 after planting the trees" in capsys.readouterr().err
    assert jobs.read_json(os.path.join(job.folder, plant.PLANT, jobs.MANIFEST_FILE)) is None
    fake_gen.FakeGens.exit_code = 0
    assert cli.main(["jobs", "resume", job.id, *woods.opts()]) == 0
    assert report(job)["trees"]["log"] > 0 and fake_gen.FakeGens.boots[-1] == "gen"
    assert json.loads(Path(job.folder, plant.PLANT, plant.RECORD).read_text())["trees"] > 0
