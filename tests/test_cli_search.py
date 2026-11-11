import dataclasses
import json

import pytest
from conftest import make_world
from fake_sampler import FakeSampler
from test_search import SMALL

from rwg2migrator import Refusal, cli, cli_search, generator, jobs, maps, provision, region, search, world


@pytest.fixture
def fake(monkeypatch):
    orig = cli_search._inputs
    monkeypatch.setattr(cli_search, "_inputs", lambda job, sc, gens: dataclasses.replace(orig(job, sc, gens), plan=SMALL))
    monkeypatch.setattr(cli_search.Generators, "sampler", lambda self, seed, threads: FakeSampler)


@pytest.fixture
def setup(tmp_path):
    root = tmp_path / "server"
    make_world(str(root), claims={"alpha": [(3, 3), (4, 3)]}, chunk_specs={(x, z): {} for x in range(8) for z in range(8)})
    out = tmp_path / "area.json"
    assert cli.main(["area", str(root), "--claims", "all", "--out", str(out), "--cache", str(tmp_path / "c")]) == 0
    return root, out, ["--data", str(tmp_path / "data"), "--cache", str(tmp_path / "c")]


def test_search_candidates_jobs_and_map(fake, setup, capsys, tmp_path):
    root, area_file, common = setup
    capsys.readouterr()
    assert cli.main(["search", str(root), "--area", str(area_file), *common]) == 0
    out = capsys.readouterr()
    assert "#1  [Recommended]" in out.out and "Border: " in out.out and "search: Looking over the whole range:" in out.err
    assert cli.main(["candidates", "latest", "--all", "--data", common[1]]) == 0
    text = capsys.readouterr().out
    assert "Details: seam median" in text and "Offsets: biome" in text
    assert cli.main(["candidates", "latest", "--json", "--data", common[1]]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["candidates"][0]["card"]["rank"] == 1
    n, alts = len(result["candidates"]), result["alternatives"]
    assert alts and f"Alternative 1 (--pick {n + 1})" in text and f"Alternative {len(alts)} (--pick {n + len(alts)})" in text
    job = jobs.find(common[1], "latest")
    assert maps.pick(job, n + len(alts))["candidate"] == alts[-1] and maps.pick(job, n)["candidate"] == result["candidates"][-1]
    with pytest.raises(Refusal, match=f"1 to {n + len(alts)}"):
        maps.pick(job, n + len(alts) + 1)
    assert cli.main(["jobs", "list", "--data", common[1]]) == 0
    assert " search  done  " in capsys.readouterr().out
    png = tmp_path / "m.png"
    assert cli.main(["map", str(root), "--area", str(area_file), "--search", "latest", "--out", str(png), *common]) == 0
    assert png.stat().st_size > 0
    (root / "config").mkdir()
    (root / "config" / "rwg2.cfg").write_text("changed\n")
    capsys.readouterr()
    assert cli.main(["map", str(root), "--area", str(area_file), "--search", "latest", "--out", str(png), *common]) == 1
    assert "ran on GTNH's default RWG2 settings (no config folder), but the config folder" in capsys.readouterr().err


def test_search_runs_on_the_config_folder_named_and_its_resume_too(fake, setup, capsys, monkeypatch):
    root, area_file, common = setup
    capsys.readouterr()
    assert cli.main(["search", str(root), "--area", str(area_file), *common]) == 0
    assert f"Note: {world.NO_CONFIG}." in capsys.readouterr().err
    mine = root.parent / "mine"
    mine.mkdir()
    (mine / "rwg2.cfg").write_text("mine\n")
    seen, real = [], cli_search.Generators.__init__
    monkeypatch.setattr(cli_search.Generators, "__init__", lambda self, *a: seen.append(a[3]) or real(self, *a))
    assert cli.main(["search", str(root), "--area", str(area_file), "--config-dir", str(mine), *common]) == 0
    assert "Note: no config folder" not in capsys.readouterr().err
    job = jobs.find(common[1], "latest")
    job.update(state=jobs.INTERRUPTED)
    assert cli.main(["jobs", "resume", "latest", "--data", common[1]]) == 0
    assert seen == [str(mine), str(mine)] and search.sampled_config(job) == jobs.tree_key(str(mine))
    png, area_arg = root.parent / "m.png", ["--area", str(area_file), "--search", job.id, "--out", str(root.parent / "m.png")]
    assert cli.main(["map", str(root), *area_arg, *common]) == 0 and seen[-1] == str(mine) and png.stat().st_size > 0
    (mine / "rwg2.cfg").unlink()
    capsys.readouterr()
    assert cli.main(["map", str(root), *area_arg, *common]) == 1
    err = capsys.readouterr().err
    assert f"search job {job.id} ran on a config folder, but none is in use now" in err and err.rstrip().endswith(f"--area {area_file}.")


def test_a_config_change_is_named_for_a_folder_and_for_no_folder(tmp_path):
    (tmp_path / "rwg2.cfg").write_text("x\n")
    key = jobs.tree_key(str(tmp_path))
    assert jobs.config_change(None, None, "search job 1") is None and jobs.config_change(str(tmp_path), key, "search job 1") is None
    assert "ran on GTNH's default RWG2 settings (no config folder)" in jobs.config_change(str(tmp_path), None, "search job 1")
    assert jobs.config_change(None, False, "search job 1") == "search job 1 did not record which config folder it ran on"
    assert jobs.config_change(None, key, "search job 1") == "search job 1 ran on a config folder, but none is in use now"
    assert jobs.config_change(str(tmp_path), "other", "search job 1") == f"the config folder {tmp_path} changed since search job 1 ran"


def test_map_of_the_world_alone_takes_the_bpp_asked_for(setup, capsys, tmp_path):
    root, area_file, common = setup
    capsys.readouterr()
    assert cli.main(["map", str(root), "--area", str(area_file), "--bpp", "4", "--out", str(tmp_path / "now.png"), "--json", *common]) == 0
    assert [w["bpp"] for w in json.loads(capsys.readouterr().out)["windows"]] == [4]
    assert cli.main(["map", str(root), "--area", str(area_file), "--bpp", "3", "--out", str(tmp_path / "now.png"), *common]) == 1
    assert "--bpp 3 is not one of" in capsys.readouterr().err


def test_resume_of_an_interrupted_search_from_the_cli(fake, setup, capsys):
    root, area_file, common = setup
    assert cli.main(["search", str(root), "--area", str(area_file), *common]) == 0
    job = jobs.find(common[1], "latest")
    job.update(state=jobs.INTERRUPTED)
    capsys.readouterr()
    assert cli.main(["jobs", "resume", "latest", "--data", common[1]]) == 0
    assert "#1  [Recommended]" in capsys.readouterr().out and job.state() == jobs.DONE


def test_resume_refuses_a_finished_job_and_cancel_marks_it(fake, setup, capsys):
    root, area_file, common = setup
    assert cli.main(["search", str(root), "--area", str(area_file), *common]) == 0
    assert cli.main(["jobs", "resume", "latest", "--data", common[1]]) == 1
    assert "Only an interrupted" in capsys.readouterr().err
    assert cli.main(["jobs", "cancel", "latest", "--data", common[1]]) == 0
    assert cli.main(["jobs", "discard", "latest", "--data", common[1]]) == 0
    assert cli.main(["candidates", "latest", "--data", common[1]]) == 1


def test_area_of_another_world_is_refused(fake, setup, tmp_path, capsys):
    _, area_file, common = setup
    other = tmp_path / "other"
    make_world(str(other))
    assert cli.main(["search", str(other), "--area", str(area_file), *common]) == 1
    assert "was made for" in capsys.readouterr().err


def test_search_without_generator_needs_the_eula(tmp_path):
    gens = cli_search.Generators(str(tmp_path / "c"), type("A", (), {"accept_eula": False})(), print)
    with pytest.raises(Refusal, match="--accept-eula"):
        gens.context()


def test_search_with_eula_checks_on_the_pack_config_then_samples_on_the_users(tmp_path, monkeypatch):
    calls = []
    java = provision.Java("java", "21", 21, "amd64", "64")
    monkeypatch.setattr(provision, "install", lambda cache, rel, zip_path, emit: calls.append("install"))
    monkeypatch.setattr(provision, "choose_java", lambda *a: java)

    class Gen:
        def start_heartbeat(self):
            calls.append("heartbeat")

        def stop(self, emit=None):
            calls.append("stop")

    monkeypatch.setattr(generator, "start", lambda ctx, boot: calls.append(("start", ctx.accept_eula, boot.heap, boot.overlay)) or Gen())
    monkeypatch.setattr(generator, "check", lambda gen, entry, emit: calls.append("check"))
    user = tmp_path / "server" / "config"
    user.mkdir(parents=True)
    (user / "rwg2.cfg").write_text("user\n")
    cache, heap, eula = str(tmp_path / "c"), generator.DEFAULT_HEAP, type("A", (), {"accept_eula": True})()
    sampler = cli_search.Generators(cache, eula, print, str(user)).sampler(123, 4)()
    pristine, users = ("start", True, heap, None), ("start", True, heap, str(user))
    assert calls == ["install", pristine, "heartbeat", "check", "stop", users, "heartbeat"]
    assert generator.settings(cache) == {"release": "2.9.0", "java": "java", "heap": heap}
    assert (sampler.seed, sampler.threads) == ("123", 4)
    calls.clear()
    cli_search.Generators(cache, eula, print, str(user)).sampler(123, 4)()
    assert calls == [users, "heartbeat"]


def _break_chunk(root, pos):
    path = root / "World" / "region" / "r.0.0.mca"
    data = bytearray(path.read_bytes())
    loc = int.from_bytes(data[4 * region.slot_index(*pos) : 4 * region.slot_index(*pos) + 3], "big")
    data[loc * region.SECTOR + 5 : loc * region.SECTOR + 9] = b"bad!"
    path.write_bytes(bytes(data))


def test_an_area_with_unreadable_kept_chunks_is_flagged_and_the_search_refuses_it_before_it_starts(fake, setup, capsys, tmp_path):
    root, area_file, common = setup
    _break_chunk(root, (3, 3))
    capsys.readouterr()
    assert cli.main(["area", str(root), "--claims", "all", "--out", str(area_file), "--cache", common[3]]) == 0
    assert "Warning: 1 chunk of the area to keep does not read in the world (3,3)" in capsys.readouterr().out
    assert cli.main(["search", str(root), "--area", str(area_file), *common]) == 1
    err = capsys.readouterr().err
    assert "Refused: 1 chunk of the area to keep does not read" in err and "restore the region files" in err, err
    assert jobs.all_jobs(common[1]) == []
