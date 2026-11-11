import dataclasses
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import psutil
import pytest
from conftest import chunk_root, make_world, terrain, write_region
from fake_sampler import NAMES, FakeSampler

from rwg2migrator import Refusal, appearance, area, jobs, scan, score, search, verdicts

SMALL = search.Plan(
    b_range=1024,
    coarse_max=150,
    coarse_full=25,
    coarse_land=8,
    frontier=(2, 4),
    frontier_check=3,
    top=3,
    sep=256,
    b_steps=(32, 16, 8),
    land_basins=2,
    l_range=4096,
    l_step=2048,
    landmass_full=4,
    land_seeds=2,
    l_steps=(512, 256),
)
CORE = {(3, 3), (4, 3)}
SEED = 1234567890123
RUNNER = Path(__file__).parent / "run_fake_search.py"
KILL_AFTER_BATCHES = 6
KILL_IN = "refine-biome"
WAIT = 60
FAR_APART = 640
LONG_PART = 700
COVER_STEP = 64


@pytest.fixture
def site(tmp_path, cache):
    w = make_world(str(tmp_path / "s"), chunk_specs={(x, z): {} for x in range(8) for z in range(8)})
    a = area.build({}, CORE, set(), 1)
    path = tmp_path / "area.json"
    path.write_text(json.dumps(a))
    return w, a, str(path), str(tmp_path / "data")


def inputs(w, a, cache, sampler):
    return search.Inputs(scan.scan(w, cache, workers=1), a, appearance.load_prefs(None), "2.9.0", SEED, lambda: sampler, SMALL)


def run(data, w, a, cache):
    job = jobs.create(data, "search", {}, w)
    result = jobs.run(job, lambda j: search.run_job(j, inputs(w, a, cache, FakeSampler())))
    return job, result


def tops(result):
    return [tuple(c["config"]) for c in result["candidates"]]


def batches(job):
    return [e for e in job.events() if e.get("event") == "batch"]


def sampled_once(job, cache):
    """Every part (edge, grid, land) of every config was sampled exactly once over all runs of the job."""
    have = {"edge": "seam", "grid": "biome", "land": "feasible"}
    return all(sum(e["sampled"] for e in batches(job) if p in e["parts"]) == sum(k in r for r in cache.values()) for p, k in have.items())


def final_cache(job):
    return search.load_cache(os.path.join(job.folder, "search", search.CACHE_FILE))


def test_search_finds_distinct_feasible_candidates_with_cards(site, cache):
    w, a, _, data = site
    job, result = run(data, w, a, cache)
    assert job.state() == jobs.DONE
    top = result["candidates"]
    assert 1 <= len(top) <= SMALL.picks and all(c["feasible"] for c in top)
    assert [c["total"] for c in top] == sorted(c["total"] for c in top)
    for x, y in zip(top, top[1:], strict=False):
        assert max(abs(x["config"][0] - y["config"][0]), abs(x["config"][1] - y["config"][1])) >= SMALL.sep
    recommended = [i for i, c in enumerate(top) if verdicts.RECOMMENDED in c["card"]["badges"]]
    assert recommended == [verdicts.recommended(top)] and top[0]["card"]["lines"][0].startswith("Border: ")
    assert result["reference"]["name"] == "Forest" and result["bans"] == ["other", "sand", "snow", "water", "wet"]
    assert sampled_once(job, final_cache(job)) and len(final_cache(job)) == result["search"]["configs"]
    assert any(e.get("event") == "stage" and e["stage"] == "coarse" for e in job.events())
    best = [e for e in job.events() if e.get("event") == "best"]
    assert len(best) >= 2 and best[-1]["top"][0]["lines"][0].startswith("Border: ")
    assert [b["config"] for b in best[-1]["top"]] == [c["config"] for c in top]
    _, again = run(data, w, a, cache)
    assert tops(again) == tops(result)


def test_kill_mid_search_then_resume_samples_nothing_twice(site, cache):
    w, a, area_path, data = site
    ref_job, reference = run(data, w, a, cache)
    job = jobs.create(data, "search", {}, w)
    p = subprocess.Popen([sys.executable, str(RUNNER), data, job.id, w, area_path, cache], cwd=Path(__file__).parent)
    deadline = time.monotonic() + WAIT
    while not any(e["stage"].startswith(KILL_IN) for e in batches(job)) and time.monotonic() < deadline and p.poll() is None:
        time.sleep(0.05)
    p.kill()
    p.wait()
    assert job.state() == jobs.RUNNING and any(e["stage"].startswith(KILL_IN) for e in batches(job))
    cached = len(search.load_cache(os.path.join(job.folder, "search", search.CACHE_FILE)))
    assert jobs.refresh(data) == [job.id] and job.state() == jobs.INTERRUPTED
    result = jobs.run(job, lambda j: search.run_job(j, inputs(w, a, cache, FakeSampler())))
    assert tops(result) == tops(reference)
    assert result["search"]["cached_before"] == cached > 0
    assert sampled_once(job, final_cache(job))
    assert final_cache(job) == final_cache(ref_job)


def _interrupt_and_resume(job, w, a, cache):
    job.update(state=jobs.INTERRUPTED)
    return jobs.run(job, lambda j: search.run_job(j, inputs(w, a, cache, FakeSampler())))


def resets(job):
    return [e["changed"] for e in job.events() if e.get("event") == "reset"]


def test_resume_after_the_ground_changed_drops_the_stale_cache(site, cache, tmp_path):
    w, a, _, data = site
    job, first = run(data, w, a, cache)
    make_world(str(tmp_path / "s"), chunk_specs={(x, z): {"ids": terrain(top=72)} for x in range(8) for z in range(8)})
    resumed = _interrupt_and_resume(job, w, a, cache)
    _, fresh = run(data, w, a, cache)
    assert resumed["search"]["cached_before"] == 0 and "site" in resets(job)[0]
    assert tops(resumed) == tops(fresh) != tops(first)
    assert final_cache(job) == final_cache(jobs.find(data, "latest"))


def test_resume_drops_a_cache_made_by_another_version_or_by_a_build_from_before_the_first_release(site, cache, monkeypatch):
    w, a, _, data = site
    job, _ = run(data, w, a, cache)
    inputs_file = Path(job.folder, "search", search.INPUTS_FILE)
    keys = json.loads(inputs_file.read_text())
    inputs_file.write_text(json.dumps({("tool" if k == "version" else k): v for k, v in keys.items()}))
    assert _interrupt_and_resume(job, w, a, cache)["search"]["cached_before"] == 0 and resets(job) == [["version"]]
    monkeypatch.setattr(search, "__version__", "another")
    assert _interrupt_and_resume(job, w, a, cache)["search"]["cached_before"] == 0 and resets(job) == [["version"]] * 2


def test_own_cache_drops_the_search_and_its_candidates_when_inputs_change(tmp_path):
    job = jobs.create(str(tmp_path / "data"), "search", {}, "w")
    search.own_cache(job, {"site": "a"})
    outputs = [Path(job.step_dir("search"), search.CACHE_FILE), Path(job.step_dir("candidates"), "candidates.json")]
    for f in outputs:
        f.write_text("{}\n")
    search.own_cache(job, {"site": "a"})
    assert all(f.exists() for f in outputs) and resets(job) == []
    search.own_cache(job, {"site": "b"})
    assert not any(f.exists() for f in outputs) and resets(job) == [["site"]]


def test_a_resave_that_leaves_the_site_alone_keeps_the_cache(site, cache):
    w, a, _, data = site
    job, first = run(data, w, a, cache)
    region_file = os.path.join(w, "region", "r.0.0.mca")
    later = os.path.getmtime(region_file) + 60
    os.utime(region_file, (later, later))
    resumed = _interrupt_and_resume(job, w, a, cache)
    assert resets(job) == [] and resumed["search"]["sampled_now"] == 0 and tops(resumed) == tops(first)


def test_a_changed_config_folder_starts_the_search_again(site, cache):
    w, a, _, data = site
    job = jobs.create(data, "search", {}, w)
    jobs.run(job, lambda j: search.run_job(j, dataclasses.replace(inputs(w, a, cache, FakeSampler()), config="before")))
    job.update(state=jobs.INTERRUPTED)
    again = jobs.run(job, lambda j: search.run_job(j, dataclasses.replace(inputs(w, a, cache, FakeSampler()), config="after")))
    assert resets(job) == [["lmap"]] and again["search"]["cached_before"] == 0 and search.sampled_config(job) == "after"
    assert sum(e.get("event") == "rendered" for e in job.events()) == 2


def test_tree_key_follows_settings_not_times_or_comments(tmp_path):
    (tmp_path / "sub").mkdir()
    cfg, ini = tmp_path / "sub" / "rwg2.cfg", tmp_path / "IC2.ini"
    cfg.write_text("# Configuration file\ngeneral {\n    I:size=1\n}\n")
    ini.write_text("; created Nov 2, 2026, 12:22:26 AM\n[world]\nx=1\n")
    before = jobs.tree_key(str(tmp_path))
    os.utime(cfg, (1, 1))
    ini.write_text("; created Nov 2, 2026, 4:57:51 AM\r\n[world]\r\nx=1\r\n")
    assert jobs.tree_key(str(tmp_path)) == before
    cfg.write_text("# Configuration file\ngeneral {\n    I:size=2\n}\n")
    assert jobs.tree_key(str(tmp_path)) != before and jobs.tree_key(None) is None and jobs.tree_key(str(tmp_path / "no")) is None


def test_a_search_on_a_sand_base_avoids_everything_but_sand(cache, tmp_path):
    ids = terrain()
    ids[63], ids[64] = 12, 0
    w = make_world(str(tmp_path / "d"), chunk_specs={(x, z): {"ids": ids} for x in range(8) for z in range(8)})
    job, result = run(str(tmp_path / "data"), w, area.build({}, CORE, set(), 1), cache)
    assert job.state() == jobs.DONE and result["reference"]["look"] == "sand"
    assert result["bans"] == ["other", "snow", "water", "wet"] and result["candidates"]


def test_cancel_keeps_the_best_so_far(site, cache):
    w, a, _, data = site

    class Cancelling(FakeSampler):
        def sample(self, configs, points):
            if self.calls == KILL_AFTER_BATCHES:
                job.request_cancel()
            return super().sample(configs, points)

    job = jobs.create(data, "search", {}, w)
    result = jobs.run(job, lambda j: search.run_job(j, inputs(w, a, cache, Cancelling())))
    assert job.state() == jobs.CANCELLED and result["search"]["cancelled"]
    assert os.path.isfile(os.path.join(job.folder, "candidates", "candidates.json"))
    assert len(batches(job)) == KILL_AFTER_BATCHES + 1


def test_lock_whose_pid_was_reused_makes_the_job_interrupted(tmp_path):
    data, world = str(tmp_path / "data"), str(tmp_path / "w")
    job = jobs.create(data, "search", {}, world)
    job.update(state=jobs.RUNNING)
    reused = {"pid": os.getpid(), "create_time": psutil.Process().create_time() - 1000, "job": job.id}
    path = jobs._lock_path(data, world)
    os.makedirs(os.path.dirname(path))
    Path(path).write_text(json.dumps(reused))
    assert jobs.refresh(data) == [job.id] and job.state() == jobs.INTERRUPTED
    with jobs.lock(data, world, "next"):
        assert json.loads(Path(path).read_text())["job"] == "next"
    assert not os.path.exists(path)


def test_lock_held_by_a_live_process_refuses(tmp_path):
    data, world = str(tmp_path / "data"), str(tmp_path / "w")
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        path = jobs._lock_path(data, world)
        os.makedirs(os.path.dirname(path))
        held = {"pid": child.pid, "create_time": psutil.Process(child.pid).create_time(), "job": "other"}
        Path(path).write_text(json.dumps(held))
        with pytest.raises(Refusal, match="already working") as refused, jobs.lock(data, world, "mine"):
            pass
        assert refused.value.action.endswith("rwg2-migrator jobs cancel other")
        assert json.loads(Path(path).read_text()) == held
    finally:
        child.kill()
        child.wait()


def test_the_lock_file_is_claimed_only_when_there_is_none(tmp_path):
    path = str(tmp_path / "locks" / "w.json")
    assert jobs._claim(path, {"job": "a"}) and not jobs._claim(path, {"job": "b"})
    assert json.loads(Path(path).read_text()) == {"job": "a"} and os.listdir(tmp_path / "locks") == ["w.json"]


def test_a_job_must_be_named(tmp_path):
    with pytest.raises(Refusal, match="no job was named"):
        jobs.find(str(tmp_path / "data"), None)


def test_step_manifest_detects_a_changed_output(tmp_path):
    job = jobs.create(str(tmp_path / "data"), "search", {}, "w")
    Path(job.step_dir("s"), "out.txt").write_text("a")
    job.finish_step("s", ["out.txt"], {"k": 1})
    assert job.step_valid("s", {"k": 1}) and not job.step_valid("s", {"k": 2})
    job.finish_step("s", ["out.txt"], {"steps": (1, 2)})
    assert job.step_valid("s", {"steps": (1, 2)})
    job.finish_step("s", ["out.txt"], {"k": 1})
    Path(job.step_dir("s"), "out.txt").write_text("b")
    assert not job.step_valid("s", {"k": 1})


def test_torn_cache_line_is_ignored(tmp_path):
    path = tmp_path / "configs.jsonl"
    jobs.append_line(path, [{"bx": 1, "bz": 2, "lx": 0, "lz": 0, "seam": 1.0}])
    with open(path, "a") as f:
        f.write('[{"bx": 3, "bz"')
    assert list(search.load_cache(path)) == [(1, 2, 0, 0)]
    jobs.append_line(path, [{"bx": 5, "bz": 6, "lx": 0, "lz": 0, "seam": 2.0}])
    assert list(search.load_cache(path)) == [(1, 2, 0, 0), (5, 6, 0, 0)]


def test_find_latest_and_prefix(tmp_path):
    data = str(tmp_path / "data")
    a = jobs.create(data, "search", {}, "w")
    time.sleep(1.1)
    b = jobs.create(data, "search", {}, "w")
    assert jobs.find(data, "latest").id == b.id and jobs.find(data, a.id[:-2]).id == a.id
    with pytest.raises(Refusal, match="no job"):
        jobs.find(data, "nope")


def test_tolerant_preferences_scale_the_penalty(site, cache):
    w, a, _, data = site
    job = jobs.create(data, "search", {}, w)
    inp = inputs(w, a, cache, FakeSampler())
    inp.prefs = inp.prefs | {"tolerance": 3.0}
    result = jobs.run(job, lambda j: search.run_job(j, inp))
    assert result["candidates"]
    for c in result["candidates"]:
        assert c["penalty"] == round(score.penalty(c["inc"], 3.0), 4) and c["total"] == pytest.approx(c["seam"] + c["penalty"], abs=2e-4)


def test_frontier_sees_only_configs_this_run_visited(tmp_path):
    site = score.Site(*(np.zeros((0, 2)) for _ in range(9)))
    later = {"bx": 8, "bz": 8, "lx": 0, "lz": 0, "level": "main", "seam": 0.5, "biome": 0.1, "feasible": True}
    later |= {f"inc_{b}": 0.0 for b in score.BAND_NAMES}
    jobs.append_line(tmp_path / search.CACHE_FILE, [later])
    ev = search.Evaluator(site, None, None, str(tmp_path))
    ev.plan = SMALL
    assert ev.frontier("f") == dict.fromkeys(SMALL.frontier)
    ev.run([(8, 8, 0, 0)], "replayed", search.LAND)
    assert ev.frontier("f")[2]["seam"] == 0.5


def brute_force_nearest(config, a):
    """Nearest Tainted Land and ocean within the reach of core, from every render column around each kept part on its own."""
    core, pts = set(map(tuple, a["core"])), []
    for p in area.parts(set(map(tuple, a["kept"]))):
        cs = np.array(sorted(p))
        lo, hi = cs.min(0) * area.CHUNK - verdicts.REACH, (cs.max(0) + 1) * area.CHUNK + verdicts.REACH
        gx, gz = np.meshgrid(np.arange(lo[0], hi[0], search.MID_BPP), np.arange(lo[1], hi[1], search.MID_BPP))
        pts.append(np.c_[gx.ravel(), gz.ravel()])
    pts = np.concatenate(pts)
    rec = FakeSampler().columns(config, pts)
    out = {}
    for name, mask in (("taint", rec["palette"] == NAMES.index("Tainted Land")), ("ocean", rec["ocean"] == 1)):
        d = area.distance(pts[mask], core)
        d = d[d <= verdicts.REACH]
        out[name] = round(float(d.min()), 1) if len(d) else None
    return out


def test_tainted_land_and_ocean_are_looked_for_around_every_kept_part(tmp_path, cache):
    w = make_world(str(tmp_path / "s"), chunk_specs={(x, z): {} for x in range(8) for z in range(8)})
    far = [(FAR_APART + x, FAR_APART + z) for x in range(8) for z in range(8)]
    write_region(os.path.join(w, "region", "r.20.20.mca"), {c: chunk_root(*c) for c in far})
    core = CORE | {(FAR_APART + x, FAR_APART + z) for x, z in CORE}
    a = area.build({}, core, set(), 1)
    _, result = run(str(tmp_path / "data"), w, a, cache)
    c = result["candidates"][0]
    truth = brute_force_nearest(tuple(c["config"]), a)
    assert truth["taint"] is not None and truth["ocean"] is not None
    assert (c["taint"]["blocks"], c["ocean"]) == (truth["taint"], truth["ocean"])
    assert c["card"]["lines"][3].startswith(f"Tainted Land: {round(truth['taint'])} blocks")
    kept_parts = area.parts(set(map(tuple, a["kept"])))
    assert [p["chunks"] for p in c["parts"]] == [len(p) for p in kept_parts] == [12, 12]
    assert sum(p["pairs"] for p in c["parts"]) == c["pairs"] and c["parts"][1]["centre"] == [FAR_APART * 16 + 64, FAR_APART * 16 + 56]
    assert c["card"]["lines"][4:] == [verdicts.part_border(i, p) for i, p in enumerate(c["parts"], 1)]
    assert c["card"]["lines"][5].startswith(f"Border of part 2 (12 chunks around {FAR_APART * 16 + 64}, {FAR_APART * 16 + 56}): ")


def test_each_column_takes_the_side_of_its_own_part():
    kept = area.dilate(CORE | {(FAR_APART + x, FAR_APART + z) for x, z in CORE}, 1)
    centres = search.part_centres(np.array([[64, 40], [10304, 10280]]), kept)
    assert centres.tolist() == [[64, 56], [10304, 10296]]
    look, same = np.zeros(2, int), np.zeros(2, bool)
    assert search.off_side(np.array([[64, 300], [10304, 10600]]), look, same, centres)["side"] == "south"


def test_find_windows_tile_every_part_grown_by_the_reach():
    kept = {(x, 0) for x in range(LONG_PART)} | {(FAR_APART * 3, FAR_APART * 3)}
    wins = search.find_windows(kept)
    unit = area.CHUNK * search.MID_BPP
    assert all(size <= search.TILE and x0 % area.CHUNK == z0 % area.CHUNK == size % unit == 0 for x0, z0, size in wins)
    boxes = area.windows(kept, verdicts.REACH)
    assert len(boxes) == 2 and len(wins) == 4
    for x0, z0, x1, z1 in boxes:
        gx, gz = np.meshgrid(np.r_[np.arange(x0, x1, COVER_STEP), x1 - 1], np.r_[np.arange(z0, z1, COVER_STEP), z1 - 1])
        assert search._inside(np.c_[gx.ravel(), gz.ravel()], wins).all()
