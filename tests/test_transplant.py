import gzip
import json
import os
import struct
from pathlib import Path

from conftest import Migration, add_migration_parts, compound, level_dat, make_world, player_dat, string, tag, terrain, tree, write_nbt

from rwg2migrator import cli, nbt, region, staging, transplant, world


def _err(capsys):
    return capsys.readouterr().err


def _level(path):
    return nbt.parse(gzip.decompress(Path(path).read_bytes()))[1]


def _chunk(world_dir, pos):
    path = os.path.join(world_dir, "region", f"r.{pos[0] >> 5}.{pos[1] >> 5}.mca")
    rec, ts = region.records(Path(path).read_bytes())[region.slot_index(*pos)]
    return transplant.decode(rec, pos)[1]["Level"][1], ts


def test_staging_holds_only_kept_chunks_and_changes_only_the_named_files(mig):
    code, job = mig.preview()
    assert code == 0
    st = job.data["args"]["staging"]
    src, dst = tree(mig.world), tree(st)
    record = staging.final(job)
    assert "region/r.3.3.mca" in src and "region/r.3.3.mca" not in dst and dst == record
    m = staging.source(job)
    far = m["files"]["region/r.3.3.mca"]
    assert (far["copied"], far["sha256"]) == (False, src["region/r.3.3.mca"])
    changed = {rel for rel in src if src[rel] != dst.get(rel)}
    assert changed == {"level.dat", "region/r.0.0.mca", "region/r.3.3.mca", "data/Village.dat", "data/Stronghold.dat"}
    assert "data/Mineshaft.dat" in dst and "DIM-1/region/r.0.0.mca" in dst
    kept = {tuple(c) for c in json.loads(Path(mig.area).read_text())["kept"]}
    recs = region.records(Path(st, "region", "r.0.0.mca").read_bytes())
    assert {region.slot_pos(0, 0, i) for i in recs} == kept
    old = region.records(Path(mig.world, "region", "r.0.0.mca").read_bytes())
    assert all(recs[i] == old[i] for i in recs)


def test_level_dat_splice_changes_the_three_generator_fields_only(tmp_path):
    w = make_world(str(tmp_path / "s"))
    level = level_dat()
    level["Data"][1]["generatorOptions"] = string("junk")
    level["Data"][1]["generatorVersion"] = tag(nbt.INT, 1)
    write_nbt(os.path.join(w, "level.dat"), level)
    write_nbt(os.path.join(w, "level.dat_old"), level)
    before = _level(os.path.join(w, "level.dat"))
    a = {"core": [[1, 1]], "kept": [[x, z] for x in range(3) for z in range(3)], "teams": []}
    r = transplant.run(w, a, transplant.Options())
    for name in transplant.LEVEL_FILES:
        after = _level(os.path.join(w, name))
        data = after["Data"][1]
        assert (data["generatorName"][1], data["generatorVersion"][1], data["generatorOptions"][1]) == ("RWG2_CONTINENTS", 0, "")
        for k in transplant.FIELDS:
            data[k] = before["Data"][1][k]
        assert after == before and r["level"][name] == {"generatorName": "RWG", "generatorVersion": 1, "generatorOptions": "junk"}


def test_claims_made_after_the_area_step_are_refused_by_name(mig, capsys):
    claims = os.path.join(mig.world, world.CLAIMS, "alpha.dat")
    entries = [{"x": tag(nbt.INT, x), "z": tag(nbt.INT, z)} for x, z in [(1, 1), (2, 1), (3, 3)]]
    write_nbt(claims, {"ClaimedChunks": compound(**{"0": tag(nbt.LIST, [nbt.COMPOUND, entries])})})
    capsys.readouterr()
    code, _ = mig.preview()
    err = _err(capsys)
    assert code == 1 and "1 chunk claimed since the area was made lies outside it: 3,3" in err and "--add" in err


def test_edge_area_sets_margin_flags_and_refuses_core_without_set_populated(tmp_path, capsys):
    root = str(tmp_path / "edge")
    specs = {(x, z): {"populated": 0 if (x, z) in ((0, 0), (1, 1)) else 1} for x in range(3) for z in range(3) if (x, z) != (2, 2)}
    make_world(root, claims={"alpha": [(1, 1)]}, chunk_specs=specs)
    add_migration_parts(root)
    m = Migration(Path(root), tmp_path)
    other = tmp_path / "other-area.json"
    assert cli.main(["area", root, "--add", "0,0:0,0", "--out", str(other), "--cache", m.cache]) == 0
    capsys.readouterr()
    assert cli.main(["preview", root, "--area", str(other), "--search", m.search.id, "--no-generate", *m.opts()]) == 1
    assert "was made for another area" in _err(capsys)
    code, job = m.preview()
    err = _err(capsys)
    assert code == 1 and "1 chunk of the area to keep was never decorated" in err and ": 1,1." in err and "--set-populated" in err
    code, job = m.preview("--set-populated")
    assert code == 0, _err(capsys)
    r = json.loads(Path(job.folder, "transplant", "record.json").read_text())
    assert r["populated_set"] == {"core": [[1, 1]], "margin": [[0, 0]]} and r["never_generated"] == [[2, 2]]
    st = job.data["args"]["staging"]
    for pos in ((0, 0), (1, 1)):
        level, ts = _chunk(st, pos)
        old, old_ts = _chunk(m.world, pos)
        old["TerrainPopulated"] = [nbt.BYTE, 1]
        assert level == old and ts == old_ts
    assert "never explored kept chunks: 1" in capsys.readouterr().out.lower()


def test_data_files_follow_rwg2_structure_generators_and_noise(tmp_path):
    w = make_world(str(tmp_path / "s"))
    add_migration_parts(str(tmp_path / "s"))
    write_nbt(os.path.join(w, world.RWG_DATA), {"data": compound(noiseImplementation=string("OPENSIMPLEX"))})
    a = {"core": [[1, 1]], "kept": [[1, 1]], "teams": []}
    r = transplant.run(w, a, transplant.Options())
    assert transplant.DROP_DATA == ("Village.dat", "Stronghold.dat")
    assert r["data_deleted"] == ["data/Village.dat", "data/Stronghold.dat", "data/rwg_data.dat"] and r["rwg_data"] == "deleted"
    assert r["data_reported"] == ["Temple"] and os.path.isfile(os.path.join(w, "data", "Mineshaft.dat"))
    assert sorted(f for f in os.listdir(os.path.join(w, "data"))) == ["GregTech_OregenPattern.dat", "Mineshaft.dat", "Temple.dat"]


def test_players_outside_kept_move_to_the_top_of_the_nearest_core_chunk(tmp_path):
    ids = terrain(top=70)
    specs = {(x, z): {"ids": ids} for x in range(4) for z in range(4)}
    w = make_world(str(tmp_path / "s"), claims={}, players={"u1": ("A", 24, 24), "u2": ("B", 60, 60)}, chunk_specs=specs)
    write_nbt(os.path.join(w, "playerdata", "u3.dat"), player_dat(60, 60, dim=-1))
    level = level_dat(player=player_dat(-40, 5))
    write_nbt(os.path.join(w, "level.dat"), level)
    a = {"core": [[1, 1]], "kept": [[x, z] for x in range(3) for z in range(3)], "teams": []}
    before = tree(w)
    r = transplant.run(w, a, transplant.Options())
    moved = {m["name"]: (m["from"], m["to"]) for m in r["players_moved"]}
    assert moved == {"u2": ((3, 3), (24.5, 71.0, 24.5)), "single-player (level.dat)": ((-3, 0), (24.5, 71.0, 24.5))}
    pos = nbt.parse(gzip.decompress(Path(w, "playerdata", "u2.dat").read_bytes()))[1]["Pos"][1][1]
    assert [struct.unpack(">d", b)[0] for b in pos] == [24.5, 71.0, 24.5]
    after = tree(w)
    assert after["playerdata/u1.dat"] == before["playerdata/u1.dat"] and after["playerdata/u3.dat"] == before["playerdata/u3.dat"]


def test_an_unreadable_kept_chunk_is_refused(tmp_path):
    w = make_world(str(tmp_path / "s"))
    path = os.path.join(w, "region", "r.0.0.mca")
    good = region.records(Path(path).read_bytes())
    good[region.slot_index(1, 1)] = (region.deflate(b"\x0a\x00\x00broken"), 1)
    Path(path).write_bytes(region.build(good))
    a = {"core": [[1, 1]], "kept": [[1, 1], [2, 1]], "teams": []}
    try:
        transplant.run(w, a, transplant.Options())
        raise AssertionError("accepted")
    except transplant.Refusal as e:
        assert "1 kept chunk does not read in the working copy: 1,1" in e.reason
