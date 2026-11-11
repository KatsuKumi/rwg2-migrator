"""M4 on a COPY of the real fixture server tree (RWG2_FIXTURE): transplant, apply/rollback round trip and the variants.
Every test leaves the copy as it found it."""

import gzip
import json
import os
import shutil
import struct
import threading
import time
import zipfile
import zlib
from pathlib import Path

import numpy as np
import pytest
from conftest import Migration, age, golden, level_dat, player_dat, search_job, tree, write_nbt

from rwg2migrator import apply, cli, forgecfg, jobs, nbt, region, scan, staging, transplant, world

FIXTURE = os.environ.get("RWG2_FIXTURE")
pytestmark = [pytest.mark.fixture, pytest.mark.skipif(not FIXTURE, reason="set RWG2_FIXTURE to a COPY of the pre-RWG2 server tree")]
POC_REGIONS = ["r.2.0.mca", "r.3.0.mca", "r.3.1.mca", "r.4.0.mca", "r.4.1.mca"]
NOT_OURS = ("session.lock", apply.MARKER)
OFFSETS = "expert / worldgen migration"


def chunks_of(path):
    """{(cx, cz): (compression byte + payload, timestamp)} read straight from a region file's header, independent of region.py."""
    data = Path(path).read_bytes()
    rx, rz = (int(p) for p in os.path.basename(path).split(".")[1:3])
    out = {}
    for i in range(1024):
        loc = int.from_bytes(data[4 * i : 4 * i + 3], "big")
        if loc:
            start = loc * 4096
            length = int.from_bytes(data[start : start + 4], "big")
            out[(rx * 32 + i % 32, rz * 32 + i // 32)] = (
                data[start + 4 : start + 4 + length],
                int.from_bytes(data[4096 + 4 * i : 4100 + 4 * i], "big"),
            )
    return out


def all_chunks(world_dir):
    out = {}
    for p in sorted(Path(world_dir, "region").glob("r.*.*.mca")):
        out.update(chunks_of(p))
    return out


@pytest.fixture(scope="module")
def fx(tmp_path_factory, green1_pick):
    work = Path(FIXTURE, staging.WORK)
    existed = work.exists()
    m = Migration(Path(FIXTURE), tmp_path_factory.mktemp("m4"), cand=green1_pick, cache=scan.cache_root())
    m.world0 = tree(m.world)
    m.config0 = tree(os.path.join(m.root, "config"))
    yield m
    assert tree(m.world) == m.world0 and tree(os.path.join(m.root, "config")) == m.config0
    if not existed:
        shutil.rmtree(work)


def test_transplant_keeps_core_identical_and_changes_only_the_named_files(fx, capsys):
    t0 = time.monotonic()
    code, job = fx.preview()
    seconds = time.monotonic() - t0
    assert code == 0, capsys.readouterr().err
    st = job.data["args"]["staging"]
    a = json.loads(Path(fx.area).read_text())
    core, kept = {tuple(c) for c in a["core"]}, {tuple(c) for c in a["kept"]}
    pin = json.loads(golden("kept-green1.json").read_text())
    assert len(core) == 908 and kept == {tuple(c) for c in pin["kept"]} and len(kept) == 1090
    before, after = all_chunks(fx.world), all_chunks(st)
    assert set(after) == kept and sum(after[c] == before[c] for c in core) == 908 and all(after[c] == before[c] for c in kept)
    assert sorted(p.name for p in Path(st, "region").iterdir()) == POC_REGIONS
    src, dst = fx.world0, tree(st)
    outside = {rel for rel in src if not rel.startswith("region/")}
    differ = sorted(rel for rel in outside if src[rel] != dst.get(rel))
    assert differ == ["data/Stronghold.dat", "data/Village.dat", "level.dat", "level.dat_old"]
    assert not {"data/Stronghold.dat", "data/Village.dat"} & set(dst) and set(dst) - set(src) == set()
    for name in ("level.dat", "level.dat_old"):
        old, new = (gzip.decompress(Path(d, name).read_bytes()) for d in (fx.world, st))
        cut = next(i for i in range(len(old)) if old[i] != new[i])
        assert old[cut : cut + 4] == b"\x03RWG" and new[cut : cut + 16] == b"\x0fRWG2_CONTINENTS"
        assert old[:cut] == new[:cut] and old[cut + 4 :] == new[cut + 16 :]
    r = json.loads(Path(job.folder, "transplant", "record.json").read_text())
    assert r["players_moved"] == [] and r["never_generated"] == [] and r["populated_set"] == {"core": [], "margin": []}
    assert r["data_reported"] == [] and r["rwg_data"] == "kept" and r["core_identical"] == 908
    assert tree(fx.world) == fx.world0
    print(f"preview --no-generate on the fixture: {seconds:.1f} s; {len(dst)} staging files; {len(src) - len(dst)} fewer than the world")


def _other_key(path):
    """A key outside the offsets section, to stand for an edit the user makes after Apply."""
    for _, key, m in forgecfg._walk(forgecfg._lines(path)):
        if key and key[0] != OFFSETS and m.group(2) == "B":
            return key, m.group(5)
    raise AssertionError("no boolean key in rwg2.cfg")


def test_apply_then_rollback_round_trip(fx, capsys):
    _, job = fx.preview()
    vp0 = tree(os.path.join(fx.root, "visualprospecting"))
    t0 = time.monotonic()
    assert fx.apply(job) == 0, capsys.readouterr().err
    seconds = time.monotonic() - t0
    out = capsys.readouterr().out
    assert "908 of 908 protected chunks are identical" in out
    applied = all_chunks(fx.world)
    before = {}
    for p in POC_REGIONS:
        before.update(chunks_of(os.path.join(apply.resolve(apply.record_of(fx.world)["backup"], fx.root), "region", p)))
    core = {tuple(c) for c in json.loads(Path(fx.area).read_text())["core"]}
    assert sum(applied[c] == before[c] for c in core) == 908
    assert cli.main(["verify", fx.root, *fx.opts()]) == 0
    cfg = os.path.join(fx.root, "config", "rwg2.cfg")
    key, value = _other_key(cfg)
    flipped = "false" if value == "true" else "true"
    forgecfg.set_keys(cfg, {key: ("B", flipped)})
    t1 = time.monotonic()
    assert fx.rollback() == 0, capsys.readouterr().err
    back = time.monotonic() - t1
    assert tree(fx.world) == fx.world0 and tree(os.path.join(fx.root, "visualprospecting")) == vp0
    assert forgecfg.get(cfg, key) == flipped and forgecfg.get(cfg, (OFFSETS, "Biome Placement Offset X")) == "0"
    forgecfg.set_keys(cfg, {key: ("B", value)})
    assert tree(os.path.join(fx.root, "config")) == fx.config0
    print(f"apply (refresh + swap + record + verify) {seconds:.1f} s; rollback {back:.1f} s")


def _team_file(world_dir):
    (path,) = Path(world_dir, world.CLAIMS).glob("*.dat")
    raw = path.read_bytes()
    return path, raw, nbt.parse(gzip.decompress(raw))


def test_a_claim_made_after_the_area_step_is_refused_by_name(fx, capsys):
    path, raw, (name, root) = _team_file(fx.world)
    root["ClaimedChunks"][1]["0"][1][1].append({"x": [nbt.INT, 2200], "z": [nbt.INT, 400]})
    try:
        path.write_bytes(gzip.compress(nbt.write(name, root)))
        capsys.readouterr()
        code, _ = fx.preview()
        err = capsys.readouterr().err
        assert code == 1 and "1 chunk claimed since the area was made lies outside it: 2200,400" in err
    finally:
        path.write_bytes(raw)


def _edge_spot(world_dir):
    """An unpopulated chunk with an unpopulated and a never-generated neighbour: the edge of the explored world."""
    t = scan.scan(world_dir, scan.cache_root()).chunk_table()
    cells = set(zip(t["cx"].tolist(), t["cz"].tolist(), strict=True))
    flat = {c for c, p in zip(zip(t["cx"].tolist(), t["cz"].tolist(), strict=True), t["populated"].tolist(), strict=True) if not p}
    ring = [(dx, dz) for dx in (-1, 0, 1) for dz in (-1, 0, 1) if dx or dz]
    for c in sorted(flat):
        near = {(c[0] + dx, c[1] + dz) for dx, dz in ring}
        if near & flat and near - cells:
            return c, sorted(near & flat), sorted(near - cells)
    raise AssertionError("no edge spot")


def test_edge_area_sets_flags_and_refuses_core_without_set_populated(fx, capsys, tmp_path):
    spot, flat, never = _edge_spot(fx.world)
    edge = tmp_path / "edge.json"
    rect = "{0},{1}:{0},{1}".format(*spot)
    assert cli.main(["area", fx.root, "--add", rect, "--out", str(edge), "--cache", fx.cache]) == 0
    s = search_job(fx.data, str(edge), os.path.join(fx.root, "config"))
    args = ["preview", fx.root, "--area", str(edge), "--search", s.id, "--no-generate", *fx.opts()]
    capsys.readouterr()
    assert cli.main(args) == 1
    assert f"never decorated by the game: {spot[0]},{spot[1]}." in capsys.readouterr().err
    assert cli.main([*args, "--set-populated"]) == 0
    job = next(j for j in reversed(jobs.all_jobs(fx.data)) if j.data["kind"] == "preview")
    r = json.loads(Path(job.folder, "transplant", "record.json").read_text())
    assert r["populated_set"] == {"core": [list(spot)], "margin": [list(c) for c in flat]}
    assert r["never_generated"] == [list(c) for c in never]
    st = all_chunks(job.data["args"]["staging"])
    old = all_chunks(fx.world)
    for c in [spot, *flat]:
        a, b = (nbt.parse(zlib.decompress(rec[0][1:]))[1]["Level"][1] for rec in (st[c], old[c]))
        assert a["TerrainPopulated"][1] == 1 and b["TerrainPopulated"][1] == 0
        b["TerrainPopulated"] = a["TerrainPopulated"]
        assert a == b and st[c][1] == old[c][1]
    print(f"edge: core {spot} flagged, margin {flat} flagged, never generated {never}")


def test_remote_apply_zip_checklist_and_verify(fx, capsys, tmp_path, green1_pick):
    """Verify runs where the user runs it: on the host's world, unpacked from the upload zip and later merged with an old file."""
    _, job = fx.preview()
    assert fx.apply(job, "--remote", "--zip", "--confirm-stopped") == 0
    out = capsys.readouterr().out
    assert "the migration removed 102 overworld region files" in out
    rec = apply.record_of(fx.world)
    work = Path(fx.root, staging.WORK)
    host = tmp_path / "host"
    with zipfile.ZipFile(work / "World-upload.zip") as z:
        assert {"World/level.dat", "World/region/r.2.0.mca", "config/rwg2.cfg", "config/Thaumcraft.cfg"} <= set(z.namelist())
        z.extractall(host)
    shutil.copy(Path(fx.root, "server.properties"), host)
    assert apply.record_of(str(host / "World")) == rec and rec["state"] == apply.APPLIED
    assert tree(str(host / "World")) == tree(fx.world)
    assert (work / "World-upload.txt").read_text().startswith("Upload checklist")
    assert cli.main(["verify", str(host), *fx.opts()]) == 0 and "No problem found." in capsys.readouterr().out
    backup = apply.resolve(rec["backup"], fx.root)
    leftover = host / "World" / "region" / "r.0.0.mca"
    leftover.write_bytes(Path(backup, "region", "r.0.0.mca").read_bytes())
    cfg = host / "config" / "rwg2.cfg"
    applied = cfg.read_text()
    forgecfg.set_keys(str(cfg), {(OFFSETS, k): ("I", "0") for k in rec["offsets"]})
    reset = cfg.read_text()
    assert cli.main(["verify", str(host), "--restore-offsets", *fx.opts()]) == 2
    out = capsys.readouterr().out
    wrote = f"Biome Placement Offset X is 0, the migration wrote {green1_pick['config'][0]} (restored)"
    assert "Leftover: region/r.0.0.mca holds" in out and wrote in out
    assert cfg.read_text() == applied and sum(a != b for a, b in zip(reset.splitlines(), applied.splitlines(), strict=True)) == 4
    leftover.unlink()
    assert cli.main(["verify", str(host), *fx.opts()]) == 0
    assert fx.rollback() == 0


def test_single_player_copy_of_the_fixture(fx, capsys, tmp_path, green1_pick):
    game = tmp_path / "inst" / ".minecraft"
    save = game / "saves" / "World"
    shutil.copytree(fx.world, save)
    shutil.copytree(os.path.join(fx.root, "config"), game / "config")
    level = nbt.parse(gzip.decompress((save / "level.dat").read_bytes()))
    data = level[1]["Data"][1]
    data["Player"] = [nbt.COMPOUND, player_dat(data["SpawnX"][1], data["SpawnZ"][1])]
    (save / "level.dat").write_bytes(gzip.compress(nbt.write(*level)))
    age(str(save))
    write_nbt(str(game / "saves" / "Test" / "level.dat"), level_dat(generator="RWG2_CONTINENTS"))
    m = Migration(save, tmp_path, cand=green1_pick, world_dir=save, cache=scan.cache_root())
    _, job = m.preview()
    capsys.readouterr()
    base = ["apply", str(save), "--preview", job.id, "--yes", *m.opts()]
    assert cli.main(base) == 1 and "other saves of this instance already use RWG2 (Test)" in capsys.readouterr().err
    assert cli.main([*base, "--allow-shared-instance"]) == 0
    rec = apply.record_of(str(save))
    assert rec["affected_saves"] == ["Test"]
    pos = nbt.parse(gzip.decompress((save / "level.dat").read_bytes()))[1]["Data"][1]["Player"][1]["Pos"][1][1]
    x, y, z = (struct.unpack(">d", b)[0] for b in pos)
    core = np.array(json.loads(Path(m.area).read_text())["core"])
    assert ((core[:, 0] * 16 + 8.5 == x) & (core[:, 1] * 16 + 8.5 == z)).any() and y > 60
    print(f"single player: Data.Player moved from the world spawn to {x}, {y}, {z}")


class GameLikeWriter(threading.Thread):
    """Saves two kept chunks of a region file the way the game does, while the copy runs: one rewritten in place in its own
    sectors, one appended at the end of the file with the header pointed at it afterwards; each record goes out in two halves."""

    BURSTS = 3
    PAUSE = 0.002
    GAP = 0.05

    def __init__(self, path, cells, start):
        super().__init__(daemon=True)
        self.path, self.cells, self.start_at, self.bursts = path, cells, start, 0

    def _new(self, data, pos):
        i = region.slot_index(*pos)
        loc, count = int.from_bytes(data[4 * i : 4 * i + 3], "big"), data[4 * i + 3]
        name, root = transplant.decode(region.record(data, loc, count), pos)
        root["Level"][1]["InhabitedTime"] = [nbt.LONG, root["Level"][1]["InhabitedTime"][1] + 1]
        return i, loc, count, region.deflate(nbt.write(name, root))

    def _halves(self, f, offset, rec):
        half = len(rec) // 2
        for at, part in ((offset, rec[:half]), (offset + half, rec[half:])):
            f.seek(at)
            f.write(part)
            f.flush()
            time.sleep(self.PAUSE)

    def burst(self):
        data = Path(self.path).read_bytes()
        stamp = int(time.time()).to_bytes(4, "big")
        with open(self.path, "r+b") as f:
            i, loc, count, rec = self._new(data, self.cells[0])
            assert len(rec) <= count * region.SECTOR
            self._halves(f, loc * region.SECTOR, rec)
            f.seek(region.SECTOR + 4 * i)
            f.write(stamp)
            i, _, _, rec = self._new(data, self.cells[1])
            end, n = len(data), -(-len(rec) // region.SECTOR)
            self._halves(f, end, rec + bytes(n * region.SECTOR - len(rec)))
            f.seek(4 * i)
            f.write((end // region.SECTOR << 8 | n).to_bytes(4, "big"))
            f.seek(region.SECTOR + 4 * i)
            f.write(stamp)
        self.bursts += 1

    def run(self):
        self.start_at.wait()
        for _ in range(self.BURSTS):
            self.burst()
            time.sleep(self.GAP)


def test_a_game_like_writer_during_the_copy_then_the_refresh(fx, capsys, monkeypatch):
    path = os.path.join(fx.world, "region", "r.3.0.mca")
    original = Path(path).read_bytes()
    core = sorted({tuple(c) for c in json.loads(Path(fx.area).read_text())["core"]} & set(chunks_of(path)))
    started, copies, real = threading.Event(), [], staging._read_through

    def watched(a, b):
        copies.append(a)
        if a != path or copies.count(path) > 1:
            return real(a, b)
        _, loc, _, rec = writer._new(Path(path).read_bytes(), core[2])
        with open(path, "r+b") as f:
            f.seek(loc * region.SECTOR)
            f.write(rec[: len(rec) // 2])
            f.flush()
            out = real(a, b)
            f.write(rec[len(rec) // 2 :])
        started.set()
        return out

    writer = GameLikeWriter(path, core[:2], started)
    writer.start()
    torn, check = [], staging.torn_chunk
    monkeypatch.setattr(staging, "torn_chunk", lambda p, *a: torn.append((p, check(p, *a))) or torn[-1][1])
    monkeypatch.setattr(staging, "_read_through", watched)
    try:
        code, job = fx.preview()
        writer.join()
        found, in_preview = [c for p, c in torn if p.endswith("r.3.0.mca")], copies.count(path)
        assert code == 0 and writer.bursts == GameLikeWriter.BURSTS and in_preview >= 2 and found[0] == core[2]
        st = os.path.join(job.data["args"]["staging"], "region", "r.3.0.mca")
        staged = chunks_of(st)
        assert all(transplant.decode(region.record(Path(st).read_bytes(), *_loc(st, c)), c) for c in staged)
        writer.burst()
        capsys.readouterr()
        assert fx.apply(job) == 0
        refresh = next(line for line in capsys.readouterr().err.splitlines() if "refresh re-copied" in line)
        backup = apply.resolve(apply.record_of(fx.world)["backup"], fx.root)
        world_now, source_now = chunks_of(path), chunks_of(os.path.join(backup, "region", "r.3.0.mca"))
        assert all(world_now[c] == source_now[c] for c in core) and world_now[core[0]] != staged[core[0]]
        assert staging.mismatches(fx.world, staging.final(job), ignore=NOT_OURS) == []
        print(
            f"writer: 1 torn save, {GameLikeWriter.BURSTS} saves during the copy, 1 after it; r.3.0.mca copied {in_preview} times "
            f"by the preview, torn kept chunk per copy: {found}; {refresh}"
        )
        assert fx.rollback() == 0
    finally:
        Path(path).write_bytes(original)


def _loc(path, pos):
    data = Path(path).read_bytes()
    i = region.slot_index(*pos)
    return int.from_bytes(data[4 * i : 4 * i + 3], "big"), data[4 * i + 3]
