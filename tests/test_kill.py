"""Power cuts: kill -9 at random points inside the copy, the transplant and the swap, then resume; every run must end where an
uninterrupted run ends, and after every kill the folders must be in a state of the table."""

import os
import random
import shutil
import signal
import subprocess
import sys
from pathlib import Path

import fake_gen
import pytest
from conftest import Migration, add_migration_parts, level_dat, make_world, tree, write_nbt
from run_fake_cli import BATCH
from test_trees import forest

from rwg2migrator import apply, generator, genworld, jobs, plant, staging

POINTS = 5
SEED = 20261102
KILLED = (-getattr(signal, "SIGKILL", signal.SIGTERM), signal.SIGTERM)
NOT_OURS = ("session.lock", apply.MARKER)
TIMEOUT = 300
RUNNER = Path(__file__).parent / "run_fake_cli.py"
RING = "3"


def fresh(base, alone=False, **world_kw):
    """A server folder ready for preview; alone: its World folder only, with no server.properties and no config folder."""
    root = base / "server"
    make_world(str(root), players={"u1": ("Alice", 24, 24), "u2": ("Bob", 60, 60)}, **world_kw)
    write_nbt(str(root / "World" / "level.dat_old"), level_dat())
    add_migration_parts(str(root))
    if not alone:
        return Migration(root, base)
    os.remove(root / "server.properties")
    shutil.rmtree(root / "config")
    return Migration(root / "World", base, world_dir=root / "World")


def beside(m):
    """What Apply writes next to the world: the files of the config folder, and the offset lines left to set by hand."""
    parent = os.path.dirname(m.world)
    config, by_hand = os.path.join(parent, "config"), Path(parent, os.path.basename(m.world) + apply.BY_HAND)
    return tree(config) if os.path.isdir(config) else None, by_hand.read_text() if by_hand.is_file() else None


def cli(m, *argv, kill=None, fake=False):
    env = {k: v for k, v in os.environ.items() if k != jobs.KILL_ENV} | ({jobs.KILL_ENV: kill} if kill else {})
    args = [sys.executable, *([str(RUNNER)] if fake else ["-m", "rwg2migrator"]), *map(str, argv), *m.opts()]
    return subprocess.run(args, env=env, capture_output=True, text=True, timeout=TIMEOUT)


def counted(monkeypatch, step, run):
    """How many kill points the step passes in an uninterrupted run, and the run's result."""
    monkeypatch.setenv(jobs.KILL_ENV, f"{step}:0")
    jobs._passed.clear()
    out = run()
    monkeypatch.delenv(jobs.KILL_ENV)
    return jobs._passed.get(step, 0), out


def points(step, n):
    assert n >= POINTS, f"{step} has only {n} kill points"
    return sorted(random.Random(f"{SEED}-{step}").sample(range(1, n + 1), POINTS))


def preview_job(m):
    return next(j for j in jobs.all_jobs(m.data) if j.data["kind"] == "preview")


@pytest.mark.parametrize("step", ["copy", "transplant"])
def test_kill_9_inside_a_preview_step_then_resume(tmp_path, monkeypatch, step):
    ref = fresh(tmp_path / "ref")
    n, (code, job) = counted(monkeypatch, step, ref.preview)
    assert code == 0
    want = staging.final(job)
    for point in points(step, n):
        m = fresh(tmp_path / f"{step}-{point}")
        r = cli(m, "preview", m.root, "--area", m.area, "--search", m.search.id, "--no-generate", kill=f"{step}:{point}")
        assert r.returncode in KILLED, (point, r.returncode, r.stderr[-2000:])
        job = preview_job(m)
        r = cli(m, "jobs", "resume", job.id)
        assert r.returncode == 0, (point, r.stderr[-2000:])
        assert staging.final(job) == want and tree(job.data["args"]["staging"]) == want, point
    print(f"{step}: {n} kill points, killed at {points(step, n)}")


@pytest.mark.parametrize("alone", [False, True], ids=["server", "world-alone"])
def test_kill_9_inside_the_swap_then_apply_again(tmp_path, monkeypatch, alone):
    ref = fresh(tmp_path / "ref", alone)
    _, job = ref.preview()
    n, code = counted(monkeypatch, apply.SWAP, lambda: ref.apply(job))
    assert code == 0
    want, want_beside = tree(ref.world, skip=NOT_OURS), beside(ref)
    assert (want_beside[0] is None, want_beside[1] is None) == (alone, not alone)
    seen = set()
    for point in points(apply.SWAP, n):
        m = fresh(tmp_path / f"swap-{point}", alone)
        _, job = m.preview()
        world0 = tree(m.world)
        r = cli(m, "apply", m.root, "--preview", job.id, "--yes", "--auto-restart-off", kill=f"{apply.SWAP}:{point}")
        assert r.returncode in KILLED, (point, r.returncode, r.stderr[-2000:])
        applying = next(j for j in jobs.all_jobs(m.data) if j.data["kind"] == "apply")
        journal = apply.Journal(os.path.join(applying.folder, apply.JOURNAL))
        plan = journal.first("plan")
        stage = job.data["args"]["staging"]
        if os.path.isdir(stage + apply.TRIAL) and not os.path.exists(stage):
            stage += apply.TRIAL
        state = apply.swap_state(m.world, apply.moved(journal, plan["backup"]), stage, applying.id)
        assert state in (apply.NOT_SWAPPED, apply.BETWEEN, apply.SWAPPED), (point, state)
        seen.add(state)
        r = cli(m, "apply", m.root, "--preview", job.id, "--yes", "--auto-restart-off")
        assert r.returncode == 0, (point, r.stderr[-2000:])
        assert tree(m.world, skip=NOT_OURS) == want and tree(plan["backup"]) == world0, point
        assert beside(m) == want_beside, point
        assert apply.record_of(m.world)["state"] == apply.APPLIED
    print(f"swap: {n} kill points, killed at {points(apply.SWAP, n)}, states after the kills {sorted(seen)}")
    assert len(seen) >= 2, seen


def test_rollback_after_a_kill_once_the_record_is_written_frees_the_world(tmp_path, monkeypatch):
    ref = fresh(tmp_path / "ref")
    _, job = ref.preview()
    n, _ = counted(monkeypatch, apply.SWAP, lambda: ref.apply(job))
    m = fresh(tmp_path / "killed")
    _, job = m.preview()
    world0 = tree(m.world)
    r = cli(m, "apply", m.root, "--preview", job.id, "--yes", "--auto-restart-off", kill=f"{apply.SWAP}:{n}")
    assert r.returncode in KILLED and apply.record_of(m.world)["state"] == apply.APPLIED
    r = cli(m, "rollback", m.root, "--yes", "--auto-restart-off")
    assert r.returncode == 0, r.stderr[-2000:]
    assert tree(m.world) == world0
    applying = next(j for j in jobs.all_jobs(m.data) if j.data["kind"] == "apply")
    assert applying.state() == jobs.DONE and applying.data["rolled_back"]
    r = cli(m, "preview", m.root, "--area", m.area, "--search", m.search.id, "--no-generate")
    assert r.returncode == 0, r.stderr[-2000:]


def test_kill_9_inside_rollback_then_rollback_again(tmp_path):
    states = set()
    for point in (1, 2):
        m = fresh(tmp_path / f"rollback-{point}")
        _, job = m.preview()
        world0, beside0 = tree(m.world), beside(m)
        assert m.apply(job) == 0
        r = cli(m, "rollback", m.root, "--yes", "--auto-restart-off", kill=f"{apply.ROLLBACK}:{point}")
        assert r.returncode in KILLED, (point, r.returncode, r.stderr[-2000:])
        states.add(os.path.isdir(m.world))
        r = cli(m, "rollback", m.root, "--yes", "--auto-restart-off")
        assert r.returncode == 0, (point, r.stderr[-2000:])
        assert tree(m.world) == world0 and beside(m) == beside0, point
        applying = next(j for j in jobs.all_jobs(m.data) if j.data["kind"] == "apply")
        assert applying.data["rolled_back"]
    assert states == {False, True}, "the two kill points are between the renames and after them"


def generating(base, **world_kw):
    m = fresh(base, **world_kw)
    fake_gen.settings(m.cache)
    return m


@pytest.mark.parametrize("step", [genworld.GENERATE, "smooth"])
def test_kill_9_inside_generate_or_smooth_then_resume(tmp_path, monkeypatch, step):
    """A kill inside generate leaves no gen-world manifest, so resume builds the gen-world again; a kill inside smooth leaves
    staging half smoothed, so resume resets it to the copy. Both end where an uninterrupted run ends."""
    fake_gen.use(monkeypatch)
    monkeypatch.setattr(generator, "GEN_BATCH", BATCH)
    ref = generating(tmp_path / "ref")
    n, (code, job) = counted(monkeypatch, step, lambda: ref.generate("--ring", RING))
    assert code == 0
    want = staging.final(job)
    for point in points(step, n):
        m = generating(tmp_path / f"{step}-{point}")
        r = cli(m, "preview", m.root, "--area", m.area, "--search", m.search.id, "--ring", RING, kill=f"{step}:{point}", fake=True)
        assert r.returncode in KILLED, (point, r.returncode, r.stderr[-2000:])
        job = preview_job(m)
        planted = Path(
            genworld.folder(m.cache, fake_gen.RELEASE, job.id) if step == genworld.GENERATE else job.data["args"]["staging"], "planted"
        )
        planted.parent.mkdir(parents=True, exist_ok=True)
        planted.write_text("left by the killed run")
        r = cli(m, "jobs", "resume", job.id, fake=True)
        assert r.returncode == 0, (point, r.stderr[-2000:])
        assert not planted.exists(), point
        assert staging.final(job) == want and tree(job.data["args"]["staging"]) == want, point
    print(f"{step}: {n} kill points, killed at {points(step, n)}")


@pytest.mark.parametrize("step", [plant.PLANT, plant.TREES])
def test_kill_9_inside_the_plant_boot_or_the_tree_merge_then_resume(tmp_path, monkeypatch, step):
    """A kill inside the plant boot leaves no merge list, so resume plants again; a kill inside the merge leaves staging half
    merged, so resume resets it to the copy and replays the list. Both end where an uninterrupted run ends."""
    fake_gen.use(monkeypatch)
    woods = {(x, z): {"ids": forest()} for x in range(4) for z in range(4)}
    ref = generating(tmp_path / "ref", chunk_specs=woods)
    n, (code, job) = counted(monkeypatch, step, lambda: ref.generate("--ring", RING, "--trees"))
    assert code == 0
    want = staging.final(job)
    for point in points(step, n):
        m = generating(tmp_path / f"{step}-{point}", chunk_specs=woods)
        argv = ("preview", m.root, "--area", m.area, "--search", m.search.id, "--ring", RING, "--trees")
        r = cli(m, *argv, kill=f"{step}:{point}", fake=True)
        assert r.returncode in KILLED, (point, r.returncode, r.stderr[-2000:])
        job = preview_job(m)
        r = cli(m, "jobs", "resume", job.id, fake=True)
        assert r.returncode == 0, (point, r.stderr[-2000:])
        assert staging.final(job) == want and tree(job.data["args"]["staging"]) == want, point
    print(f"{step}: {n} kill points, killed at {points(step, n)}")
