"""End to end on a COPY of the real fixture server tree (pytest -m fixture; the copy is left as it was found).

With RWG2_FIXTURE and RWG2_CACHE (a set-up generator: the user's PC): inspect, area (the claims, margin 1), search, a preview of
the match in green #1's biome basin, apply, verify and rollback, all through the CLI, with M5's numbers within the tolerances.

With RWG2_FIXTURE and RWG2_PC_JOB (Linux): a preview the PC stopped right after its generator steps, packed with
`e2e_fixture.py bundle`, resumes here through the harvest, smoothing and post-check, then apply and rollback, with the same
numbers and no generator.

  python tests/e2e_fixture.py bundle --data DATA --cache CACHE --preview JOB --out FOLDER"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import pytest
from conftest import golden, tree

from rwg2migrator import cli, generator, genworld, jobs, provision, score, staging, world

FIXTURE, CACHE, PC_JOB = (os.environ.get(k) for k in ("RWG2_FIXTURE", "RWG2_CACHE", "RWG2_PC_JOB"))
CORE, KEPT, KEPT_PAIRS = 908, 1090, 3008
BASIN = 256
BUNDLE = "bundle.json"
pytestmark = [pytest.mark.fixture, pytest.mark.skipif(not FIXTURE, reason="set RWG2_FIXTURE to a COPY of the pre-RWG2 server tree")]


def m5_problems(report, smoothed):
    """Each way a fixture preview's post-check misses M5's numbers; empty when it is within them."""
    nt, before, after = report["new_terrain"], report["seams"]["before"]["kept_border"], report["seams"]["after"]["kept_border"]
    reshaped = json.loads(golden("smooth-report.json").read_text())["columns_changed"]
    smooth = after["median"] == 0 and after["p90"] <= 1 and after["over10_pct"] == 0 and after["max"] <= 2
    checks = {
        "908 protected chunks identical": report["core_identical"] == CORE,
        "game vs preview: median 0": nt["median"] == 0,
        "game vs preview: at least 98% within 1 block": nt["within1_pct"] >= 98,
        "kept border: 3,008 pairs": before["pairs"] == KEPT_PAIRS,
        "before smoothing: median 1 +-1, p90 11 +-1": abs(before["median"] - 1) <= 1 and abs(before["p90"] - 11) <= 1,
        "before smoothing: 10.3% over 10 +-0.3": abs(before["over10_pct"] - 10.3) <= 0.3,
        "before smoothing: 0.6% over 30 +-0.3": abs(before["over30_pct"] - 0.6) <= 0.3,
        "after smoothing: median 0, p90 <= 1, 0% over 10, max <= 2": smooth,
        "columns reshaped: 16,410 +-5%": abs(report["columns_reshaped"] - reshaped) <= 0.05 * reshaped,
        "0 changes in protected chunks": report["core_columns_changed"] == 0 and smoothed["chunks_changed_in_core"] == 0,
        "0 new fluid walls": smoothed["new_fluid_walls"] == 0,
    }
    return [name for name, ok in checks.items() if not ok]


def numbers(job):
    """(post-check report, smoothing record) of a preview job."""
    return (
        jobs.read_json(os.path.join(job.folder, staging.POSTCHECK, staging.REPORT)),
        jobs.read_json(os.path.join(job.folder, "smooth", staging.RECORD)),
    )


def run(*argv):
    code = cli.main([str(a) for a in argv])
    assert code == 0, f"{argv[0]} exited with {code}"


def newest(data, kind):
    return [j for j in jobs.all_jobs(str(data)) if j.data.get("kind") == kind][-1]


def basin_rank(search_job):
    """The rank of the first candidate whose biome offset lies within BASIN blocks of green #1's (its landmass may differ)."""
    green = json.loads(golden("green1.json").read_text())
    found = jobs.read_json(os.path.join(search_job.folder, "candidates", "candidates.json"))["candidates"]
    off = lambda c: max(abs(a - b) for a, b in zip(c["config"][:2], green["config"][:2], strict=True))  # noqa: E731
    near = [i for i, c in enumerate(found, 1) if off(c) <= BASIN]
    assert near, f"no candidate in green #1's basin: {[c['config'] for c in found]}"
    return near[0], found, green


@pytest.fixture
def fixture_copy():
    """The fixture copy, checked to be left as it was found; the work folder the tool makes next to it is removed after."""
    place = world.locate(FIXTURE)
    work = Path(place.root, staging.WORK)
    existed = work.exists()
    before = tree(place.world), tree(place.config)
    yield place
    assert (tree(place.world), tree(place.config)) == before
    if not existed and work.exists():
        shutil.rmtree(work)


@pytest.mark.skipif(not CACHE, reason="set RWG2_CACHE to a cache with the generator set up (the user's PC)")
def test_inspect_area_search_preview_apply_verify_rollback(fixture_copy, tmp_path, capsys):
    place, data = fixture_copy, tmp_path / "data"
    opts = ("--data", data, "--cache", CACHE)
    run("inspect", place.root, "--cache", CACHE)
    run("area", place.root, "--claims", "all", "--margin", 1, "--out", tmp_path / "area.json", "--cache", CACHE)
    a = json.loads((tmp_path / "area.json").read_text())
    golden_kept = {tuple(c) for c in json.loads(golden("kept-green1.json").read_text())["kept"]}
    assert len(a["core"]) == CORE and {tuple(c) for c in a["kept"]} == golden_kept and len(golden_kept) == KEPT
    run("search", place.root, "--area", tmp_path / "area.json", "--accept-eula", *opts)
    searched = newest(data, "search")
    rank, found, green = basin_rank(searched)
    assert found[0]["total"] <= score.total(green["record"]) and found[rank - 1]["total"] <= score.total(green["record"])
    run("preview", place.root, "--area", tmp_path / "area.json", "--search", searched.id, "--pick", rank, "--accept-eula", *opts)
    preview = newest(data, "preview")
    report, smoothed = numbers(preview)
    assert m5_problems(report, smoothed) == [], json.dumps(report)
    run("apply", place.root, "--preview", preview.id, "--yes", "--auto-restart-off", *opts)
    run("verify", place.root, "--data", data, "--cache", CACHE)
    run("rollback", place.root, "--yes", "--auto-restart-off", *opts)
    capsys.readouterr()
    print(f"search: {[(c['config'], c['total']) for c in found]}; picked #{rank}; post-check {json.dumps(report)}")


def relocate(data, moves):
    """Point the job records under data at where things are on this machine: moves maps each folder as the other machine wrote it
    to the folder here. Only job.json and area.json name folders (step records hold relative paths and keys)."""
    for path in Path(data, "jobs").glob("*/*.json"):
        if path.name in (jobs.JOB_FILE, "area.json"):
            path.write_text(json.dumps(moved(json.loads(path.read_text(encoding="utf-8")), moves), indent=1), encoding="utf-8")


def moved(value, moves):
    if isinstance(value, dict):
        return {k: moved(v, moves) for k, v in value.items()}
    if isinstance(value, list):
        return [moved(v, moves) for v in value]
    if isinstance(value, str):
        for old, new in moves.items():
            if value == old or value.startswith((old + "\\", old + "/")):
                rest = value[len(old) :].replace("\\", "/").strip("/")
                return os.path.join(new, *rest.split("/")) if rest else new
    return value


@pytest.mark.skipif(not PC_JOB, reason="set RWG2_PC_JOB to a preview bundled on the PC after its generator steps")
def test_a_pc_preview_stopped_after_its_generator_steps_resumes_applies_and_rolls_back_here(fixture_copy, tmp_path, capsys, monkeypatch):
    """No generator boots here, so a JVM of this machine (a live GTNH server) is not part of the test: its heap headroom does not
    decide whether the smoothing has the memory it needs."""
    monkeypatch.setattr(generator, "jvm_headroom", lambda: 0)
    place, bundle = fixture_copy, Path(PC_JOB)
    pc = json.loads((bundle / BUNDLE).read_text())
    made = jobs.read_json(bundle / "data" / "jobs" / pc["preview"] / genworld.GENERATE / jobs.MANIFEST_FILE)["inputs"]["probe"]
    assert made == generator.PROBE_VERSION, f"bundle generated with probe {made}, not {generator.PROBE_VERSION}: make a new one on the PC"
    data, cache = tmp_path / "data", tmp_path / "cache"
    shutil.copytree(bundle / "data", data)
    shutil.copytree(bundle / "cache", cache)
    shutil.copytree(bundle / "work", Path(place.root, staging.WORK), dirs_exist_ok=True)
    relocate(data, {pc["root"]: place.root, pc["data"]: str(data), pc["cache"]: str(cache)})
    preview = jobs.Job(str(data), pc["preview"])
    jobs.refresh(str(data))
    seen = len(preview.events())
    assert preview.state() == jobs.INTERRUPTED and not os.path.isfile(os.path.join(preview.folder, staging.POSTCHECK, jobs.MANIFEST_FILE))
    run("jobs", "resume", preview.id, "--data", data, "--cache", cache)
    resumed = preview.events()[seen:]
    assert not [e for e in resumed if e.get("step") in ("boot", genworld.GENERATE, genworld.RENDER)], resumed
    report, smoothed = numbers(preview)
    assert preview.state() == jobs.DONE and m5_problems(report, smoothed) == [], json.dumps(report)
    run("apply", place.root, "--preview", preview.id, "--yes", "--auto-restart-off", "--data", data, "--cache", cache)
    run("rollback", place.root, "--yes", "--auto-restart-off", "--data", data, "--cache", cache)
    capsys.readouterr()
    print(f"resumed on {sys.platform}: post-check {json.dumps(report)}")


def bundle(opts):
    """Pack a preview stopped after its generator steps, with its search, gen-world, generator settings and working copy."""
    data, cache, out = Path(opts.data), Path(opts.cache), Path(opts.out)
    preview = jobs.find(str(data), opts.preview)
    args = preview.data["args"]
    gen = genworld.generated(preview)
    for job_id in (args["search"], preview.id):
        shutil.copytree(data / "jobs" / job_id, out / "data" / "jobs" / job_id)
    shutil.copytree(genworld.gen_dir(preview, gen), out / "cache" / Path(genworld.gen_dir(preview, gen)).relative_to(cache))
    for name in (generator.SETTINGS_FILE, provision.CACHE_MARKER):
        shutil.copy2(cache / name, out / "cache" / name)
    shutil.copytree(Path(args["staging"]).parent, out / "work" / preview.id)
    pc = {"root": args["root"], "data": str(data), "cache": args["cache"], "preview": preview.id, "search": args["search"]}
    (out / BUNDLE).write_text(json.dumps(pc, indent=1))
    print(f"bundled {preview.id} into {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Pack a preview job for the Linux resume test")
    ap.add_argument("command", choices=["bundle"])
    for flag in ("--data", "--cache", "--preview", "--out"):
        ap.add_argument(flag, required=True)
    bundle(ap.parse_args())
