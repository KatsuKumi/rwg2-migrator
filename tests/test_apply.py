import gzip
import json
import os
import shutil
import struct
import time
import zipfile
from pathlib import Path

import pytest
from conftest import (
    OFFSETS,
    RWG2_CFG,
    Migration,
    Unreadable,
    add_migration_parts,
    age,
    level_dat,
    make_world,
    player_dat,
    search_job,
    tree,
    write_nbt,
)

from rwg2migrator import apply, cli, cli_apply, forgecfg, jobs, nbt, procs, staging, world

STAMP = "20990101-000000"
NOT_OURS = ("session.lock", apply.MARKER)


class PowerCut(BaseException):
    """Stands in for the power going out: nothing runs after it, not even the clean-up an ordinary error gets."""


def _power_cut(*args):
    raise PowerCut


@pytest.fixture
def fixed_stamp(monkeypatch):
    monkeypatch.setattr(apply, "stamp", lambda: STAMP)


def _snapshot(m):
    return tree(m.world), tree(os.path.join(m.root, "config")), tree(os.path.join(m.root, "visualprospecting"))


def _err(capsys):
    return capsys.readouterr().err


def _record(m):
    return json.loads(Path(m.world, apply.MARKER).read_text())


def _journal(job_id, m):
    return [e["phase"] for e in jobs.read_lines(os.path.join(m.data, "jobs", job_id, apply.JOURNAL))]


def _apply_job(m):
    return [j for j in jobs.all_jobs(m.data) if j.data["kind"] == "apply"][-1]


def test_apply_then_rollback_gives_back_the_world_and_the_configs(mig, capsys):
    world0, config0, vp0 = _snapshot(mig)
    code, job = mig.preview()
    assert code == 0 and mig.apply(job) == 0
    rec = _record(mig)
    assert rec["state"] == apply.APPLIED and rec["offsets"]["Biome Placement Offset X"] == OFFSETS[0]
    assert tree(mig.world, skip=NOT_OURS) == staging.final(job)
    cfg = os.path.join(mig.root, "config", "rwg2.cfg")
    assert forgecfg.get(cfg, ("expert / worldgen migration", "Landmass Placement Offset Z")) == str(OFFSETS[3])
    assert forgecfg.get(os.path.join(mig.root, "config", "Thaumcraft.cfg"), ("general", "biome_taint_spread")) == "0"
    lock = struct.unpack(">q", Path(mig.world, "session.lock").read_bytes())[0]
    assert abs(lock / 1000 - time.time()) < 60
    out = capsys.readouterr().out
    assert f'I:"Biome Placement Offset X"={OFFSETS[0]}' in out and "First rule: After every GTNH update" in out
    assert f"Config files written: {os.path.join(mig.root, 'config', 'Thaumcraft.cfg')}, {cfg}." in out
    forgecfg.set_keys(cfg, {("general", "Some Other Key"): ("B", "false")})
    pregen = Path(mig.root, "saves", "World", "pregenerationFiles")
    pregen.mkdir(parents=True)
    (pregen / "DIM0.txt").write_text("ring")
    Path(mig.root, "visualprospecting", "cache.bin").write_bytes(b"rwg2 veins")
    assert mig.rollback() == 0
    assert tree(mig.world) == world0 and tree(os.path.join(mig.root, "visualprospecting")) == vp0
    rolled = next(Path(staging.work_dir(mig.root)).glob("World-rwg2-rolled-back-*-pregenerationFiles"))
    assert not pregen.exists() and (rolled / "DIM0.txt").read_text() == "ring"
    assert (
        next(Path(staging.work_dir(mig.root)).glob("World-rwg2-rolled-back-*-visualprospecting")).joinpath("cache.bin").read_bytes()
        == b"rwg2 veins"
    )
    config = tree(os.path.join(mig.root, "config"))
    assert config["Thaumcraft.cfg"] == config0["Thaumcraft.cfg"] and config["rwg2.cfg"] != config0["rwg2.cfg"]
    assert Path(cfg).read_text() == RWG2_CFG.replace('"Some Other Key"=true', '"Some Other Key"=false')
    forgecfg.set_keys(cfg, {("general", "Some Other Key"): ("B", "true")})
    assert tree(os.path.join(mig.root, "config")) == config0
    capsys.readouterr()
    assert mig.rollback() == 1 and "carries no migration record" in _err(capsys)
    assert mig.apply(job) == 1
    err = _err(capsys)
    assert "was applied, and that Apply was rolled back" in err and f"--search {mig.search.id} --pick 1" in err, err
    assert "already applied" not in err


def test_preview_and_apply_take_the_config_folder_named(mig, tmp_path, capsys):
    other = tmp_path / "copy-of-config"
    shutil.copytree(os.path.join(mig.root, "config"), other)
    code, job = mig.preview("--config-dir", other)
    assert code == 0 and job.data["args"]["config_dir"] == job.data["args"]["place"]["config"] == str(other)
    (other / "extra.cfg").write_text("x\n")
    capsys.readouterr()
    assert mig.apply(job) == 1 and f"the config folder {other} changed since preview {job.id} ran (extra.cfg added)" in _err(capsys)
    config = os.path.join(mig.root, "config")
    assert cli.main(["apply", mig.root, "--preview", job.id, "--auto-restart-off", "--config-dir", config, *mig.opts()]) == 0
    out = capsys.readouterr().out
    assert f"the 4 offset lines in {os.path.join(config, 'rwg2.cfg')}" in out
    assert out.rstrip().endswith(f"--preview {job.id} --yes --auto-restart-off --config-dir {config}")
    assert mig.apply(job, "--config-dir", config) == 0
    assert _record(mig)["config_dir"] == "config"


def test_a_preview_runs_on_the_config_folder_its_search_ran_on(mig, tmp_path, capsys):
    other = tmp_path / "copy-of-config"
    shutil.copytree(os.path.join(mig.root, "config"), other)
    (other / "extra.cfg").write_text("x\n")
    mig.search.update(args={"config_dir": str(other)})
    Path(mig.search.folder, "search", "inputs.json").write_text(json.dumps({"lmap": {"config": jobs.tree_key(str(other))}}))
    code, job = mig.preview()
    assert code == 0 and job.data["args"]["config_dir"] == str(other)
    shutil.rmtree(other)
    capsys.readouterr()
    assert mig.preview()[0] == 1
    err = _err(capsys)
    config = os.path.join(mig.root, "config")
    assert f"the config folder {config} changed since search job {mig.search.id} ran" in err and f"--config-dir {config}." in err


def test_configs_are_written_before_the_first_rename(mig, monkeypatch):
    code, job = mig.preview()
    order, real_set, real_rename = [], forgecfg.set_keys, apply.rename
    monkeypatch.setattr(forgecfg, "set_keys", lambda p, e: order.append(("config", os.path.basename(p))) or real_set(p, e))
    monkeypatch.setattr(apply, "rename", lambda a, b: order.append(("rename", os.path.basename(a))) or real_rename(a, b))
    assert code == 0 and mig.apply(job) == 0
    first = next(i for i, (kind, _) in enumerate(order) if kind == "rename")
    assert {n for k, n in order[:first] if k == "config"} == {"rwg2.cfg", "Thaumcraft.cfg"}
    plan = apply.Journal(os.path.join(_apply_job(mig).folder, apply.JOURNAL)).first("plan")
    vp = os.path.basename(plan["backup"]) + "-visualprospecting.partial"
    assert [n for k, n in order if k == "rename"][:5] == [vp, "staging", "staging" + apply.TRIAL, "World", "staging"]


def test_the_preview_next_line_is_an_apply_that_runs(mig, capsys):
    code, job = mig.preview()
    nxt = capsys.readouterr().out.rstrip().splitlines()[-1]
    assert code == 0 and nxt.startswith("Next (keep the server stopped, and turn off its auto-restart): rwg2-migrator apply")
    assert cli.main([*nxt.split("): ", 1)[1].split()[1:], *mig.opts()]) == 0 and _record(mig)["state"] == apply.APPLIED


def test_the_backup_is_named_after_the_apply_that_ran_not_an_earlier_refused_one(mig, monkeypatch, capsys):
    _, job = mig.preview()
    monkeypatch.setattr(apply, "stamp", lambda: "20990101-000000")
    assert cli.main(["apply", mig.root, "--preview", job.id, *mig.opts()]) == 1 and "--auto-restart-off" in _err(capsys)
    assert cli.main(["apply", mig.root, "--preview", job.id, "--auto-restart-off", *mig.opts()]) == 0
    monkeypatch.setattr(apply, "stamp", lambda: "20990102-000000")
    assert mig.apply(job) == 0
    backup = apply.Journal(os.path.join(_apply_job(mig).folder, apply.JOURNAL)).first("plan")["backup"]
    assert os.path.basename(backup) == "World-pre-rwg2-20990102-000000" and os.path.isdir(backup)


def test_a_supervisor_that_recreates_the_world_between_the_renames(mig, monkeypatch, capsys, fixed_stamp):
    world0, config0, _ = _snapshot(mig)
    code, job = mig.preview()
    real = apply.rename

    def supervisor(a, b):
        real(a, b)
        if a == mig.world and "-pre-rwg2-" in b:
            write_nbt(os.path.join(mig.world, "level.dat"), level_dat(generator="RWG2_CONTINENTS"))

    monkeypatch.setattr(apply, "rename", supervisor)
    capsys.readouterr()
    assert code == 0 and mig.apply(job) == 1
    err = _err(capsys)
    assert "between the two renames" in err and "put your world back" in err
    assert tree(mig.world) == world0 and tree(os.path.join(mig.root, "config")) == config0
    aside = Path(staging.work_dir(mig.root), f"World-recreated-{STAMP}")
    assert (aside / "level.dat").is_file() and os.path.isdir(job.data["args"]["staging"])
    applying = _apply_job(mig)
    assert _journal(applying.id, mig)[-3:] == ["reverse", "aside", "reversed"]
    monkeypatch.setattr(apply, "rename", real)
    assert mig.apply(job) == 0 and _record(mig)["apply_job"] == applying.id


def test_a_running_server_that_recreated_the_world_is_left_alone(mig, monkeypatch, capsys):
    code, job = mig.preview()
    real = apply.rename

    def supervisor(a, b):
        real(a, b)
        if a == mig.world and "-pre-rwg2-" in b:
            os.makedirs(mig.world)
            monkeypatch.setattr(procs, "in_use", lambda root, w, now=None: procs.Usage(procs.RUNNING, "java (pid 7) is running"))

    monkeypatch.setattr(apply, "rename", supervisor)
    capsys.readouterr()
    assert code == 0 and mig.apply(job) == 1
    err = _err(capsys)
    assert "it is running (java (pid 7)" in err and "the config changes were undone" in err
    assert (
        os.listdir(mig.world) == []
        and forgecfg.get(os.path.join(mig.root, "config", "rwg2.cfg"), ("expert / worldgen migration", "Biome Placement Offset X")) == "0"
    )
    monkeypatch.undo()
    os.rmdir(mig.world)
    assert mig.apply(job) == 1
    assert "put your world back" in _err(capsys) and tree(mig.world).keys() >= {"level.dat", "region/r.3.3.mca"}
    assert mig.apply(job) == 0


def test_an_apply_refused_before_the_swap_writes_the_offsets_where_its_next_run_points(mig, monkeypatch, capsys, tmp_path):
    _, config0, _ = _snapshot(mig)
    code, job = mig.preview()
    other = tmp_path / "other-config"
    shutil.copytree(os.path.join(mig.root, "config"), other)
    real = apply._swap

    def changed(*_):
        raise apply.Refusal("the world changed while Apply was running", "Run apply again")

    monkeypatch.setattr(apply, "_swap", changed)
    assert code == 0 and mig.apply(job) == 1 and tree(os.path.join(mig.root, "config")) == config0
    monkeypatch.setattr(apply, "_swap", real)
    assert mig.apply(job, "--config-dir", other) == 0
    rec = _record(mig)
    assert rec["config_dir"] == str(other) and sorted(rec["configs"]) == [str(other / "Thaumcraft.cfg"), str(other / "rwg2.cfg")]
    assert rec["offset_lines"][0] in (other / "rwg2.cfg").read_text() and tree(os.path.join(mig.root, "config")) == config0


def test_a_flipped_byte_in_the_ready_copy_stops_apply_before_any_rename(mig, capsys):
    world0, config0, _ = _snapshot(mig)
    code, job = mig.preview()
    path = Path(job.data["args"]["staging"], "data", "Mineshaft.dat")
    raw = bytearray(path.read_bytes())
    raw[-1] ^= 1
    path.write_bytes(bytes(raw))
    capsys.readouterr()
    assert code == 0 and mig.apply(job) == 1
    assert "changed since it was made: 1 file, the first data/Mineshaft.dat (changed)" in _err(capsys)
    assert tree(mig.world) == world0 and tree(os.path.join(mig.root, "config")) == config0
    assert sorted(os.listdir(staging.work_dir(mig.root))) == [job.id]


def test_a_second_preview_supersedes_the_first(mig, capsys, monkeypatch):
    _, first = mig.preview()
    _, second = mig.preview()
    assert first.state() == jobs.SUPERSEDED and not os.path.exists(os.path.dirname(first.data["args"]["staging"]))
    capsys.readouterr()
    assert mig.apply(first) == 1 and "replaced by a newer preview" in _err(capsys)
    monkeypatch.setattr(apply, "rename", _power_cut)
    with pytest.raises(PowerCut):
        mig.apply(second)
    capsys.readouterr()
    assert mig.preview()[0] == 1 and f"preview {second.id} is not finished" in _err(capsys)
    monkeypatch.undo()
    assert mig.apply(second) == 0 and second.state() == jobs.DONE and second.data["applied"]


def _refused(capsys, m, job, *extra):
    capsys.readouterr()
    world0 = tree(m.world)
    assert m.apply(job, *extra) == 1
    assert tree(m.world) == world0 and not os.path.exists(os.path.join(m.root, "rwg2-work", f"World-pre-rwg2-{STAMP}"))
    lines = [line for line in _err(capsys).splitlines() if line.startswith("Refused: ")]
    assert len(lines) == 1 and ". Next: " in lines[0]
    return lines[0]


def test_apply_refusals(mig, capsys, monkeypatch, fixed_stamp):
    _, job = mig.preview()
    job.update(state=jobs.INTERRUPTED)
    assert "is not ready to apply (interrupted)" in _refused(capsys, mig, job)
    job.update(state=jobs.DONE)
    os.utime(os.path.join(mig.world, "level.dat"))
    assert "level.dat was saved" in _refused(capsys, mig, job)
    os.utime(os.path.join(mig.world, "level.dat"), (0, 0))
    monkeypatch.setattr(procs.psutil, "process_iter", lambda attrs: iter([Unreadable()]))
    assert "the world may be in use: " in _refused(capsys, mig, job)
    monkeypatch.undo()
    monkeypatch.setattr(apply, "stamp", lambda: STAMP)
    capsys.readouterr()
    assert cli.main(["apply", mig.root, "--preview", job.id, "--yes", *mig.opts()]) == 1
    assert "--auto-restart-off" in _err(capsys)
    monkeypatch.setattr(apply, "free_bytes", lambda path: 1)
    assert "MB free on the drive" in _refused(capsys, mig, job)
    monkeypatch.setattr(apply, "free_bytes", lambda path: 1 << 50)
    monkeypatch.setattr(staging, "device", lambda path: hash(path))
    assert "is on another drive" in _refused(capsys, mig, job)
    monkeypatch.setattr(staging, "device", lambda path: 1)
    monkeypatch.setattr(apply, "WINDOWS", True)
    monkeypatch.setattr(apply, "WINDOWS_PATH_MAX", len(mig.root) + 20)
    assert "more than Windows allows" in _refused(capsys, mig, job)
    monkeypatch.setattr(apply, "WINDOWS", False)
    dest = Path(staging.work_dir(mig.root), f"World-pre-rwg2-{STAMP}-source.json")
    dest.write_text("{}")
    assert f"{dest} already exists" in _refused(capsys, mig, job)
    dest.unlink()
    capsys.readouterr()
    world0 = tree(mig.world)
    assert cli.main(["apply", mig.root, "--preview", job.id, "--auto-restart-off", *mig.opts()]) == 0
    out = capsys.readouterr().out
    assert "removed for good" in out and cli_apply.NOTHING_CHANGED in out and tree(mig.world) == world0
    assert f"configs written: the 4 offset lines in {os.path.join(mig.root, 'config', 'rwg2.cfg')} and biome_taint_spread=0" in out
    assert out.rstrip().endswith(f"Next: rwg2-migrator apply {mig.world} --preview {job.id} --yes --auto-restart-off")
    assert apply.Journal(os.path.join(_apply_job(mig).folder, apply.JOURNAL)).first("plan") is None
    config = os.path.join(mig.root, "config")
    os.rename(config, config + "-away")
    assert "the config folder" in _refused(capsys, mig, job) and not os.path.exists(config)
    os.rename(config + "-away", config)
    assert mig.apply(job) == 0


def _untouched(m, world0, config0):
    return tree(m.world) == world0 and tree(os.path.join(m.root, "config")) == config0


def test_a_folder_found_at_the_backup_path_is_refused_never_swapped_in(mig, capsys, fixed_stamp):
    world0, config0, _ = _snapshot(mig)
    _, job = mig.preview()
    assert cli.main(["apply", mig.root, "--preview", job.id, "--auto-restart-off", *mig.opts()]) == 0
    found = Path(staging.work_dir(mig.root), f"World-pre-rwg2-{STAMP}")
    (found / "mine").mkdir(parents=True)
    for _ in range(2):
        capsys.readouterr()
        assert mig.apply(job) == 1 and f"Refused: {found} already exists" in _err(capsys)
        assert _untouched(mig, world0, config0) and os.listdir(found) == ["mine"]
    assert "rename1" not in _journal(_apply_job(mig).id, mig)
    os.rmdir(found / "mine")
    os.rmdir(found)
    assert mig.apply(job) == 0 and tree(str(found)) == world0


def test_backup_copy_onto_the_backup_path_or_into_the_world_is_refused(mig, capsys, fixed_stamp):
    world0, config0, _ = _snapshot(mig)
    _, job = mig.preview()
    os.makedirs(staging.work_dir(mig.root), exist_ok=True)
    for target in (staging.work_dir(mig.root), os.path.join(mig.world, "data")):
        for _ in range(2):
            capsys.readouterr()
            assert mig.apply(job, "--backup-copy", target) == 1
            assert f"the extra backup copy, {target}, is inside the world or inside {staging.work_dir(mig.root)}" in _err(capsys)
            assert _untouched(mig, world0, config0)
    assert sorted(os.listdir(staging.work_dir(mig.root))) == [job.id]
    assert mig.apply(job) == 0


def test_a_refused_first_rename_undoes_the_configs_and_leaves_the_found_folder(mig, monkeypatch, capsys, fixed_stamp):
    world0, config0, _ = _snapshot(mig)
    _, job = mig.preview()
    found = Path(staging.work_dir(mig.root), f"World-pre-rwg2-{STAMP}")
    real = apply.write_configs

    def late_folder(files, which):
        real(files, which)
        if which == "new":
            found.mkdir()

    monkeypatch.setattr(apply, "write_configs", late_folder)
    capsys.readouterr()
    assert mig.apply(job) == 1 and f"{found} already exists" in _err(capsys)
    assert _untouched(mig, world0, config0) and found.is_dir() and os.listdir(found) == []
    phases = _journal(_apply_job(mig).id, mig)
    assert phases[-3:] == ["trial", "unmoved", "reversed"] and "rename1" not in phases
    assert apply.unfinished(mig.data, job.id) is None
    monkeypatch.undo()
    found.rmdir()
    assert mig.apply(job) == 0 and _record(mig)["apply_job"] == _apply_job(mig).id


def test_rolling_back_an_apply_that_failed_its_last_check_ends_that_apply(mig, monkeypatch, capsys):
    world0, config0, _ = _snapshot(mig)
    _, job = mig.preview()
    real = apply._record

    def damaged(ctx, j, plan):
        out = real(ctx, j, plan)
        Path(ctx.place.world, "data", "Mineshaft.dat").write_bytes(b"disk error")
        return out

    monkeypatch.setattr(apply, "_record", damaged)
    capsys.readouterr()
    assert mig.apply(job) == 1 and "Roll back: rwg2-migrator rollback" in _err(capsys)
    applying = _apply_job(mig)
    assert apply.unfinished(mig.data, job.id).id == applying.id
    assert mig.rollback() == 0 and _untouched(mig, world0, config0)
    assert applying.state() == jobs.DONE and _journal(applying.id, mig)[-1] == "rolled_back"
    capsys.readouterr()
    assert cli.main(["jobs", "resume", applying.id, *mig.opts()]) == 1 and f"job {applying.id} is done" in _err(capsys)
    assert mig.preview()[0] == 0


def test_a_process_holding_the_world_refuses_apply(mig, capsys, java_child):
    _, job = mig.preview()
    assert f"java (pid {java_child.pid}) is running in {mig.root}" in _refused(capsys, mig, job)


def test_backup_copy_goes_to_another_folder_and_is_verified(mig, tmp_path):
    world0 = tree(mig.world)
    _, job = mig.preview()
    other = tmp_path / "other-drive"
    other.mkdir()
    assert mig.apply(job, "--backup-copy", other) == 0
    copies = list(other.iterdir())
    assert len(copies) == 1 and tree(str(copies[0])) == world0 and _record(mig)["backup_copy"] == str(copies[0])


def _instance(tmp_path):
    """A launcher instance: saves/Base (the world, with Data.Player outside the area), saves/Other (RWG2), config/."""
    game = tmp_path / "inst" / ".minecraft"
    build = tmp_path / "build"
    make_world(str(build), player=player_dat(-40, 5))
    add_migration_parts(str(build))
    (game / "saves").mkdir(parents=True)
    os.rename(build / "World", game / "saves" / "Base")
    os.rename(build / "config", game / "config")
    write_nbt(str(game / "saves" / "Other" / "level.dat"), level_dat(generator="RWG2_CONTINENTS"))
    (game / "journeymap" / "data" / "sp" / "Base" / "DIM0").mkdir(parents=True)
    save = game / "saves" / "Base"
    return Migration(save, tmp_path, world_dir=save), game


def test_single_player_is_blocked_by_other_rwg2_saves_unless_allowed(tmp_path, capsys):
    m, game = _instance(tmp_path)
    code, job = m.preview()
    assert code == 0 and job.data["args"]["staging"].startswith(str(game / "rwg2-work"))
    capsys.readouterr()
    assert cli.main(["apply", m.root, "--preview", job.id, "--yes", *m.opts()]) == 1
    err = _err(capsys)
    assert f"other worlds that run on the config folder {game / 'config'} already use RWG2 (Other)" in err and "Copy the instance" in err
    args = ["apply", m.root, "--preview", job.id, "--yes", "--allow-shared-instance", "--reset-journeymap", *m.opts()]
    assert cli.main(args) == 0
    rec = _record(m)
    assert rec["affected_saves"] == ["Other"] and rec["journeymap"].endswith(os.path.join("Base", "DIM0-legacy-rwg"))
    player = nbt.parse(gzip.decompress(Path(m.world, "level.dat").read_bytes()))[1]["Data"][1]["Player"][1]
    assert [struct.unpack(">d", b)[0] for b in player["Pos"][1][1]] == [24.5, 65.0, 24.5]
    backup = apply.resolve(rec["backup"], str(game))
    assert backup.startswith(str(game / "rwg2-work")) and os.path.isdir(backup)
    assert not (game / "journeymap" / "data" / "sp" / "Base" / "DIM0").exists()


def _copy_in_rwg2_server(tmp_path):
    """A legacy world copied as OldWorld into a server folder whose own World already uses RWG2."""
    root, legacy = tmp_path / "server", tmp_path / "legacy"
    make_world(str(root), generator="RWG2_CONTINENTS")
    add_migration_parts(str(root))
    make_world(str(legacy))
    add_migration_parts(str(legacy))
    shutil.copytree(legacy / "World", root / "OldWorld")
    w = root / "OldWorld"
    return Migration(w, tmp_path, world_dir=w, cand={"config": OFFSETS, "taint": None}), root / "config"


def test_a_copy_in_a_server_folder_leaves_that_server_s_config_alone(tmp_path, capsys):
    m, config = _copy_in_rwg2_server(tmp_path)
    config0 = tree(str(config))
    code, job = m.preview()
    assert code == 0 and job.data["args"]["config_dir"] is None
    assert m.apply(job) == 0 and tree(str(config)) == config0
    assert _record(m)["by_hand"] and f"Warning: {apply.NO_CONFIG}." in capsys.readouterr().out


def test_a_config_folder_another_rwg2_world_runs_on_is_refused_unless_allowed(tmp_path, capsys):
    m, config = _copy_in_rwg2_server(tmp_path)
    m.search = search_job(m.data, m.area, str(config), {"config": OFFSETS, "taint": None})
    code, job = m.preview("--config-dir", config)
    assert code == 0 and job.data["args"]["config_dir"] == str(config)
    config0 = tree(str(config))
    capsys.readouterr()
    assert m.apply(job) == 1 and tree(str(config)) == config0
    err = _err(capsys)
    assert f"other worlds that run on the config folder {config} already use RWG2 (World)" in err and "folder of its own" in err
    assert m.apply(job, "--allow-shared-instance") == 0 and forgecfg.get(str(config / "rwg2.cfg"), OFFSET_X) == str(OFFSETS[0])
    assert _record(m)["affected_saves"] == ["World"]


@pytest.mark.parametrize("kind", ["server", "singleplayer"])
def test_a_server_or_instance_with_no_config_folder_migrates_with_the_lines_to_add_by_hand(tmp_path, capsys, kind):
    root = tmp_path / ("server" if kind == "server" else os.path.join("instance", "saves"))
    make_world(str(root))
    add_migration_parts(str(root))
    shutil.rmtree(root / "config")
    if kind == "singleplayer":
        os.remove(root / "server.properties")
    w = root / "World"
    place = world.locate(str(w))
    assert (place.kind, place.config) == (kind, None)
    m = Migration(w, tmp_path, world_dir=w, cand={"config": OFFSETS, "taint": None})
    code, job = m.preview()
    assert code == 0 and f"Note: {world.NO_CONFIG}." in _err(capsys)
    assert m.apply(job) == 0 and f"Warning: {apply.NO_CONFIG}." in capsys.readouterr().out
    assert _record(m)["by_hand"] and not os.path.exists(os.path.join(place.root, "config"))
    assert cli.main(["verify", str(w), *m.opts()]) == 0 and "Config: none of its own" in capsys.readouterr().out


def test_remote_apply_writes_the_upload_zip_and_verify_finds_leftovers(mig, capsys):
    _, job = mig.preview()
    assert mig.apply(job, "--remote", "--zip", "--confirm-stopped") == 0
    out = capsys.readouterr().out
    assert "2. Delete the old World folder on the host first: the migration removed 1 overworld region file" in out
    zip_path = Path(staging.work_dir(mig.root), "World-upload.zip")
    names = zipfile.ZipFile(zip_path).namelist()
    assert "World/level.dat" in names and "World/rwg2-migration.json" in names and "config/rwg2.cfg" in names
    assert "config/Thaumcraft.cfg" in names and "World/region/r.3.3.mca" not in names
    assert Path(staging.work_dir(mig.root), "World-upload.txt").read_text().startswith("Upload checklist")
    rec = _record(mig)
    backup = apply.resolve(rec["backup"], mig.root)
    Path(mig.world, "region", "r.3.3.mca").write_bytes(Path(backup, "region", "r.3.3.mca").read_bytes())
    cfg = Path(mig.root, "config", "rwg2.cfg")
    applied = cfg.read_text().splitlines()
    cfg.write_text(RWG2_CFG)
    assert cli.main(["verify", mig.root, *mig.opts()]) == 2
    out = capsys.readouterr().out
    assert "Leftover: region/r.3.3.mca holds 1 chunk saved before Apply" in out and "Biome Placement Offset X is 0" in out
    assert cli.main(["verify", mig.root, "--restore-offsets", *mig.opts()]) == 2
    now = cfg.read_text().splitlines()
    assert sum(a != b for a, b in zip(RWG2_CFG.splitlines(), now, strict=True)) == 4 and now == applied
    os.remove(os.path.join(mig.world, "region", "r.3.3.mca"))
    assert cli.main(["verify", mig.root, *mig.opts()]) == 0 and "No problem found." in capsys.readouterr().out


def test_the_upload_zip_carries_the_applied_record_so_verify_works_on_the_host(mig, tmp_path, capsys):
    _, job = mig.preview()
    assert mig.apply(job, "--remote", "--zip", "--confirm-stopped") == 0
    host = tmp_path / "host"
    host.mkdir()
    (host / "server.properties").write_text("level-name=World\n")
    zipfile.ZipFile(Path(staging.work_dir(mig.root), "World-upload.zip")).extractall(host)
    assert json.loads((host / "World" / apply.MARKER).read_text()) == _record(mig)
    capsys.readouterr()
    assert cli.main(["verify", str(host), *mig.opts()]) == 0 and "No problem found." in capsys.readouterr().out
    backup = apply.resolve(_record(mig)["backup"], mig.root)
    Path(host, "World", "region", "r.3.3.mca").write_bytes(Path(backup, "region", "r.3.3.mca").read_bytes())
    assert cli.main(["verify", str(host), *mig.opts()]) == 2 and "Leftover: region/r.3.3.mca" in capsys.readouterr().out


def test_a_world_applied_with_a_config_folder_then_moved_alone_verifies_without_one(mig, tmp_path, capsys):
    _, job = mig.preview()
    assert mig.apply(job) == 0
    alone = tmp_path / "alone" / "World"
    shutil.copytree(mig.world, alone)
    capsys.readouterr()
    assert cli.main(["verify", str(alone), "--restore-offsets", *mig.opts()]) == 0
    out = capsys.readouterr().out
    assert "Config: none of its own found next to the world" in out and _record(mig)["offset_lines"][0] in out and "missing" not in out
    assert apply.TAINT_LINE in out
    assert not (tmp_path / "alone" / "config").exists()
    assert apply.config_in_use(world.locate(str(alone)), _record(mig)) is None


def test_verify_never_restores_the_lines_into_a_config_folder_that_lost_rwg2_cfg(mig, capsys):
    _, job = mig.preview()
    assert mig.apply(job) == 0
    cfg = Path(mig.root, "config", "rwg2.cfg")
    cfg.unlink()
    capsys.readouterr()
    assert cli.main(["verify", mig.root, "--restore-offsets", *mig.opts()]) == 0
    assert "Config: none of its own" in capsys.readouterr().out and not cfg.exists()


def test_rollback_refuses_a_world_without_a_record_and_checks_the_backup(mig, capsys):
    capsys.readouterr()
    assert mig.rollback() == 1 and "carries no migration record" in _err(capsys)
    _, job = mig.preview()
    assert mig.apply(job) == 0
    backup = apply.resolve(_record(mig)["backup"], mig.root)
    Path(backup, "data", "Mineshaft.dat").write_bytes(b"changed")
    capsys.readouterr()
    assert mig.rollback() == 1 and "changed since Apply (data/Mineshaft.dat: changed)" in _err(capsys)
    assert _record(mig)["state"] == apply.APPLIED


@pytest.mark.parametrize(
    "case",
    [
        ("plain", None, True, apply.NOT_SWAPPED),
        ("plain", "found", True, apply.NOT_SWAPPED),
        (None, "moved", True, apply.BETWEEN),
        ("ours", "moved", False, apply.SWAPPED),
        ("plain", "moved", True, apply.RECREATED),
        (None, None, True, apply.UNKNOWN),
        (None, "found", True, apply.UNKNOWN),
        (None, "moved", False, apply.UNKNOWN),
        ("ours", "moved", True, apply.UNKNOWN),
        ("ours", "found", False, apply.UNKNOWN),
        ("ours", None, False, apply.UNKNOWN),
        ("other", None, True, apply.UNKNOWN),
        ("plain", None, False, apply.UNKNOWN),
    ],
)
def test_state_table_never_guesses(tmp_path, case):
    """backup: None (no folder), "found" (a folder the journal never moved the world to), "moved" (the journal says it did)."""
    world, backup, stage, state = case
    w, b, s = tmp_path / "World", tmp_path / "backup", tmp_path / "staging"
    for path, there in ((b, backup), (s, stage)):
        if there:
            path.mkdir()
    if world:
        w.mkdir()
        job = {"ours": "apply-1", "other": "apply-2"}.get(world)
        if job:
            (w / apply.MARKER).write_text(json.dumps({"tool": apply.TOOL, "apply_job": job}))
    j = apply.Journal(str(tmp_path / "journal.jsonl"))
    j.note("plan", backup=str(b))
    if backup == "moved":
        j.note("rename1", backup=str(b))
    assert apply.swap_state(str(w), apply.moved(j, str(b)), str(s), "apply-1") == state


OFFSET_X = ("expert / worldgen migration", "Biome Placement Offset X")


def _offset(m):
    return forgecfg.get(os.path.join(m.root, "config", "rwg2.cfg"), OFFSET_X)


def _locked_trial(real):
    def rename(a, b):
        if b.endswith(apply.TRIAL):
            raise apply.Refusal(f"{a} cannot be renamed to {b} ([WinError 32] used by another process)", "Close any window")
        return real(a, b)

    return rename


def _late_save(m):
    real = apply._configs

    def configs(ctx, j):
        out = real(ctx, j)
        os.utime(os.path.join(m.world, "data", "Mineshaft.dat"), (1, 1))
        return out

    return "_configs", configs


def _server_started(m):
    real = apply._configs

    def configs(ctx, j):
        out = real(ctx, j)
        procs.in_use = lambda root, w, now=None: procs.Usage(procs.RUNNING, "java (pid 7) is running")
        return out

    return "_configs", configs


REFUSED = {
    "trial rename locked": "[WinError 32]",
    "world saved after the refresh": "the world changed while Apply was running (1 file, the first data/Mineshaft.dat)",
    "server started after the refresh": "java (pid 7) is running",
}


@pytest.mark.parametrize("cause", sorted(REFUSED))
def test_a_refusal_after_the_config_writes_puts_the_configs_back(mig, monkeypatch, capsys, cause):
    world0, config0, _ = _snapshot(mig)
    _, job = mig.preview()
    if cause == "trial rename locked":
        monkeypatch.setattr(apply, "rename", _locked_trial(apply.rename))
    else:
        monkeypatch.setattr(procs, "in_use", procs.in_use)
        monkeypatch.setattr(apply, *(_late_save if cause.startswith("world") else _server_started)(mig))
    capsys.readouterr()
    assert mig.apply(job) == 1
    err = _err(capsys)
    assert err.count("Refused: ") == 1 and REFUSED[cause] in err
    applying = _apply_job(mig)
    assert tree(os.path.join(mig.root, "config")) == config0 and _journal(applying.id, mig)[-1] == "reversed"
    assert "rename1" not in _journal(applying.id, mig) and apply.unfinished(mig.data, job.id) is None
    assert tree(mig.world).keys() == world0.keys()
    monkeypatch.undo()
    age(mig.world)
    assert mig.apply(job) == 0 and _offset(mig) == str(OFFSETS[0])


def _claim(m, cells):
    entries = [{"x": [nbt.INT, x], "z": [nbt.INT, z]} for x, z in cells]
    write_nbt(
        os.path.join(m.world, world.CLAIMS, "alpha.dat"), {"ClaimedChunks": [nbt.COMPOUND, {"0": [nbt.LIST, [nbt.COMPOUND, entries]]}]}
    )
    age(m.world)


def test_a_power_cut_after_the_config_writes_then_a_new_claim_still_gets_the_user_out(mig, monkeypatch, capsys):
    """The run after a power cut puts the configs back before anything can refuse; a claim made meanwhile then only asks for a new area."""
    world0, config0, _ = _snapshot(mig)
    _, job = mig.preview()
    monkeypatch.setattr(apply, "rename", _power_cut)
    with pytest.raises(PowerCut):
        mig.apply(job)
    monkeypatch.undo()
    applying = _apply_job(mig)
    assert _offset(mig) == str(OFFSETS[0]) and apply.is_open(applying)
    capsys.readouterr()
    assert mig.preview()[0] == 1 and f"Resume it first: rwg2-migrator jobs resume {applying.id}" in _err(capsys)
    for held in (applying.id, job.id):
        assert cli.main(["jobs", "discard", held, *mig.opts()]) == 1
        assert f"Resume it: rwg2-migrator jobs resume {applying.id}" in _err(capsys)
    _claim(mig, [(1, 1), (2, 1), (30, 30)])
    assert cli.main(["jobs", "resume", applying.id, *mig.opts()]) == 1
    assert "lies outside it: 30,30" in _err(capsys)
    assert tree(os.path.join(mig.root, "config")) == config0 and not apply.is_open(applying)
    assert {k: v for k, v in tree(mig.world).items() if "alpha" not in k} == {k: v for k, v in world0.items() if "alpha" not in k}
    assert cli.main(["jobs", "resume", applying.id, *mig.opts()]) == 1 and "Make the area again" in _err(capsys)
    assert cli.main(["area", mig.root, "--claims", "all", "--out", mig.area, "--cache", mig.cache, *mig.workers]) == 0
    mig.search = search_job(mig.data, mig.area, os.path.join(mig.root, "config"))
    code, second = mig.preview()
    assert code == 0 and mig.apply(second) == 0 and _offset(mig) == str(OFFSETS[0])
    assert cli.main(["jobs", "resume", applying.id, *mig.opts()]) == 1 and "replaced by a newer preview" in _err(capsys)
    assert cli.main(["jobs", "discard", applying.id, *mig.opts()]) == 0
    assert mig.rollback() == 0 and tree(os.path.join(mig.root, "config")) == config0


def test_discard_refuses_an_apply_between_the_renames_and_its_preview(mig, monkeypatch, capsys):
    world0, _, _ = _snapshot(mig)
    _, job = mig.preview()
    real, calls = apply.rename, []

    def cut_before_rename2(a, b):
        calls.append(a)
        if a == job.data["args"]["staging"] and len(calls) > 2:
            raise PowerCut
        return real(a, b)

    monkeypatch.setattr(apply, "rename", cut_before_rename2)
    with pytest.raises(PowerCut):
        mig.apply(job)
    monkeypatch.undo()
    assert not os.path.exists(mig.world)
    applying = _apply_job(mig)
    capsys.readouterr()
    for held in (applying.id, job.id):
        assert cli.main(["jobs", "discard", held, *mig.opts()]) == 1 and "may have changed your config or moved your world" in _err(capsys)
    assert os.path.isdir(applying.folder) and os.path.isdir(job.folder)
    assert cli.main(["jobs", "resume", applying.id, *mig.opts()]) == 0 and _record(mig)["state"] == apply.APPLIED
    assert cli.main(["jobs", "discard", applying.id, *mig.opts()]) == 0
    assert mig.rollback() == 0 and tree(mig.world) == world0


def test_a_rollback_stopped_before_its_first_rename_checks_everything_again(mig, monkeypatch, capsys):
    world0, config0, _ = _snapshot(mig)
    _, job = mig.preview()
    assert mig.apply(job) == 0
    real, calls = apply.rename, []

    def locked_once(a, b):
        calls.append(a)
        if len(calls) == 1:
            raise apply.Refusal(f"{a} cannot be renamed (locked)", "Close Explorer")
        return real(a, b)

    monkeypatch.setattr(apply, "rename", locked_once)
    capsys.readouterr()
    assert mig.rollback() == 1 and "locked" in _err(capsys)
    monkeypatch.undo()
    Path(mig.world, "region", "played.txt").write_bytes(b"weeks of play")
    age(mig.world)
    plain = ["rollback", mig.root, *mig.opts()]
    assert cli.main([*plain, "--auto-restart-off"]) == 0
    out = capsys.readouterr().out
    assert "all play since Apply is lost" in out and cli_apply.NOTHING_CHANGED in out and Path(mig.world, "region", "played.txt").exists()
    assert out.rstrip().endswith(f"Next: rwg2-migrator rollback {mig.world} --yes --auto-restart-off")
    assert cli.main([*plain, "--yes"]) == 1 and "--auto-restart-off" in _err(capsys)
    backup = Path(apply.resolve(_record(mig)["backup"], mig.root), "data", "Mineshaft.dat")
    raw = backup.read_bytes()
    backup.write_bytes(b"changed")
    assert mig.rollback() == 1 and "changed since Apply (data/Mineshaft.dat: changed)" in _err(capsys)
    assert _record(mig)["state"] == apply.APPLIED and Path(mig.world, "region", "played.txt").exists()
    backup.write_bytes(raw)
    assert mig.rollback() == 0 and tree(mig.world) == world0 and tree(os.path.join(mig.root, "config")) == config0


def _resume(m, job):
    return cli.main(["jobs", "resume", job.id, *m.opts()])


def _cut_after_the_swap(m, monkeypatch, job):
    monkeypatch.setattr(apply, "_record", _power_cut)
    with pytest.raises(PowerCut):
        m.apply(job)
    monkeypatch.undo()
    assert _record(m)["state"] == apply.STAGED
    return _apply_job(m)


def test_a_resume_after_the_swap_waits_for_a_stopped_world_and_keeps_the_play_since(mig, monkeypatch, capsys):
    _, job = mig.preview()
    applying = _cut_after_the_swap(mig, monkeypatch, job)
    Path(mig.root, "visualprospecting", "cache.bin").write_bytes(b"rwg2 veins")
    monkeypatch.setattr(procs, "in_use", lambda root, w, now=None: procs.Usage(procs.RUNNING, "java (pid 7) is running"))
    capsys.readouterr()
    assert _resume(mig, applying) == 1
    assert "Apply swapped the migrated world in, but before it wrote its record the world is in use: java (pid 7)" in _err(capsys)
    assert _record(mig)["state"] == apply.STAGED
    monkeypatch.undo()
    level, played = Path(mig.world, "level.dat"), Path(mig.world, "data", "Mineshaft.dat")
    level.write_bytes(gzip.compress(gzip.decompress(level.read_bytes()), mtime=1))
    played.write_bytes(b"saved by the game")
    assert _resume(mig, applying) == 1 and "level.dat was saved" in _err(capsys)
    monkeypatch.setattr(procs, "RECENT_SAVE_SECONDS", 0)
    os.utime(played, (1, 1))
    assert _resume(mig, applying) == 1
    err = _err(capsys)
    assert "1 file of" in err and "data/Mineshaft.dat (changed)" in err and "Roll back (this loses the play since Apply)" in err, err
    os.utime(played)
    assert _resume(mig, applying) == 0
    rec, out = _record(mig), capsys.readouterr().out
    assert rec["state"] == apply.APPLIED and "the 2 files your game saved since then are not compared" in out, out
    assert Path(apply.resolve(rec["visualprospecting"], mig.root), "cache.bin").read_bytes() == b"veins"
    assert job.data["applied"] == applying.id and applying.state() == jobs.DONE


def _level_with(m, **level):
    write_nbt(os.path.join(m.world, "level.dat"), level_dat(**level))
    age(m.world)


def test_apply_refuses_a_world_mods_or_config_changed_since_the_preview(mig, capsys, fixed_stamp):
    _, job = mig.preview()
    cfg, level = Path(mig.root, "config", "rwg2.cfg"), Path(mig.world, "level.dat").read_bytes()
    cfg.write_text(RWG2_CFG.replace('"Some Other Key"=true', '"Some Other Key"=false'))
    line = _refused(capsys, mig, job)
    assert "config folder" in line and "(rwg2.cfg changed)" in line and "Put those settings back" in line, line
    assert f"--area {os.path.join(job.folder, 'area.json')}" in line
    cfg.write_text(RWG2_CFG)
    mods = Path(mig.root, "mods")
    mods.mkdir()
    (mods / "rwg2-2.1.0.jar").write_bytes(b"jar")
    assert "the mods folder changed since preview" in (line := _refused(capsys, mig, job)) and "rwg2-2.1.0.jar added" in line
    (mods / "rwg2-2.1.0.jar").unlink()
    _level_with(mig, extra_mods={"rwg2": "2.1.0"})
    line = _refused(capsys, mig, job)
    assert "another GTNH or RWG2 version" in line and "rwg2 2.0.6 -> 2.1.0" in line, line
    other = level_dat()
    other["Data"][1]["RandomSeed"] = [nbt.LONG, 42]
    write_nbt(os.path.join(mig.world, "level.dat"), other)
    age(mig.world)
    line = _refused(capsys, mig, job)
    assert "has the seed 42, but preview" in line and "seed 1234567890123" in line and "Put the previewed world back" in line, line
    Path(mig.world, "level.dat").write_bytes(level)
    age(mig.world)
    assert mig.apply(job) == 0 and _offset(mig) == str(OFFSETS[0])


def test_rollback_finishes_when_the_config_folder_apply_wrote_into_is_gone(mig, capsys):
    world0, _, _ = _snapshot(mig)
    code, job = mig.preview()
    assert code == 0 and mig.apply(job) == 0
    shutil.move(os.path.join(mig.root, "config"), os.path.join(mig.root, "config-uploaded"))
    capsys.readouterr()
    assert mig.rollback() == 0
    assert tree(mig.world) == world0
    out = capsys.readouterr().out
    assert f"Not restored, its folder is gone: {os.path.join(mig.root, 'config', 'rwg2.cfg')}" in out
    assert '    I:"Biome Placement Offset X"=0' in out
    assert not os.path.exists(apply.rollback_journal(world.locate(mig.root)))


def test_verify_reads_a_recorded_config_folder_only_while_it_is_this_worlds(tmp_path):
    server = tmp_path / "server"
    (server / "config").mkdir(parents=True)
    (server / "config" / "rwg2.cfg").write_text(RWG2_CFG)
    copy = world.Place(str(server / "World-migrated"), str(server), None, "folder")
    assert apply.config_in_use(copy, {"config_dir": "config"}) is None
    own = world.Place(str(server / "World"), str(server), str(server / "config"), "server")
    assert apply.config_in_use(own, {"config_dir": "config"}) == str(server / "config")
    picked = tmp_path / "picked"
    picked.mkdir()
    (picked / "rwg2.cfg").write_text(RWG2_CFG)
    assert apply.config_in_use(copy, {"config_dir": str(picked)}) == str(picked)


def test_a_config_with_a_setting_twice_is_refused_in_plain_words(tmp_path):
    cfg = tmp_path / "rwg2.cfg"
    line = '    I:"Biome Placement Offset X"=0\n'
    cfg.write_text(RWG2_CFG.replace(line, line + line))
    key = ["expert / worldgen migration", "Biome Placement Offset X"]
    files = {str(cfg): {"created": False, "keys": [{"key": key, "type": "I", "old": "0", "new": "-16000"}]}}
    with pytest.raises(apply.Refusal, match="has a setting twice, or is missing one the tool writes"):
        apply.write_configs(files, "new")


def test_windows_counts_a_character_outside_the_basic_plane_twice():
    assert apply._windows_len("D:\\a") == 4 and apply._windows_len("D:\\\U0001d11e") == 5


def test_the_backup_copy_is_made_again_when_apply_runs_a_second_time(mig, tmp_path, monkeypatch, capsys):
    _, job = mig.preview()
    other = tmp_path / "other-drive"
    other.mkdir()
    real, late = apply._configs, Path(mig.world, "data", "late.dat")

    def save_late_once(ctx, j):
        real(ctx, j)
        if not late.exists():
            late.write_bytes(b"saved after the first copy")

    monkeypatch.setattr(apply, "_configs", save_late_once)
    capsys.readouterr()
    assert mig.apply(job, "--backup-copy", other) == 1 and "the world changed while Apply was running" in _err(capsys)
    assert mig.apply(job, "--backup-copy", other) == 0
    (copy,) = other.iterdir()
    backup = apply.resolve(_record(mig)["backup"], mig.root)
    assert "data/late.dat" in tree(backup) and tree(str(copy)) == tree(backup)


def test_an_error_after_the_config_writes_puts_the_configs_back(mig, monkeypatch, capsys):
    _, config0, _ = _snapshot(mig)
    _, job = mig.preview()
    monkeypatch.setattr(apply, "_save_vp", lambda *a: (_ for _ in ()).throw(OSError("the drive stopped answering")))
    capsys.readouterr()
    assert mig.apply(job) == cli.EXIT_REFUSED and "the drive stopped answering" in _err(capsys)
    applying = _apply_job(mig)
    assert tree(os.path.join(mig.root, "config")) == config0 and _journal(applying.id, mig)[-1] == "reversed"
    assert not apply.is_open(applying)
    monkeypatch.undo()
    assert mig.apply(job) == 0 and _offset(mig) == str(OFFSETS[0])


def test_discarding_a_preview_deletes_its_working_copy(mig, capsys):
    _, job = mig.preview()
    work = os.path.dirname(job.data["args"]["staging"])
    assert os.path.isdir(job.data["args"]["staging"])
    assert cli.main(["jobs", "discard", job.id, *mig.opts()]) == 0
    assert not os.path.exists(work) and not os.path.exists(job.folder) and os.path.isdir(mig.world)
    _, second = mig.preview()
    assert mig.apply(second) == 0
    world_now = tree(mig.world)
    assert cli.main(["jobs", "discard", second.id, *mig.opts()]) == 0
    assert tree(mig.world) == world_now and os.path.isdir(apply.resolve(_record(mig)["backup"], mig.root))


def test_a_rollback_cut_between_its_renames_says_how_to_finish_it(mig, monkeypatch, capsys):
    world0 = tree(mig.world)
    _, job = mig.preview()
    assert mig.apply(job) == 0
    real, calls = apply.rename, []

    def cut_before_rename2(a, b):
        calls.append(a)
        if len(calls) == 2:
            raise PowerCut
        return real(a, b)

    monkeypatch.setattr(apply, "rename", cut_before_rename2)
    with pytest.raises(PowerCut):
        mig.rollback()
    monkeypatch.undo()
    assert not os.path.exists(mig.world)
    capsys.readouterr()
    assert cli.main(["rollback", mig.root, "--auto-restart-off", *mig.opts()]) == 1
    err = _err(capsys)
    assert "stopped before it finished" in err and f"rwg2-migrator rollback {mig.world} --yes --auto-restart-off" in err
    assert "no migration record" not in err and not os.path.exists(mig.world)
    assert mig.rollback() == 0 and tree(mig.world) == world0


def test_single_player_roll_back_keeps_the_pregeneration_files_of_the_restored_world(tmp_path):
    m, _ = _instance(tmp_path)
    pregen = Path(m.world, "pregenerationFiles")
    pregen.mkdir()
    (pregen / "DIM0.txt").write_text("an unfinished pregeneration of the old world")
    age(m.world)
    world0 = tree(m.world)
    _, job = m.preview()
    assert cli.main(["apply", m.root, "--preview", job.id, "--yes", "--allow-shared-instance", *m.opts()]) == 0
    assert cli.main(["rollback", m.root, "--yes", *m.opts()]) == 0
    assert tree(m.world) == world0
