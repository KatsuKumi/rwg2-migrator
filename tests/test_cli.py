import json
import os
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest
from conftest import LAZY_POS, Unreadable, chunk_root, lazy_chunk, make_world, string, tag, write_region, zero_body
from test_scan import grave_te

import rwg2migrator
from rwg2migrator import ISSUES, area, cli, nbt, procs, world


def test_pyproject_uv_lock_and_the_module_name_one_version():
    root = Path(__file__).parent.parent
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))["project"]["version"]
    locked = tomllib.loads((root / "uv.lock").read_text(encoding="utf-8"))["package"]
    assert [p["version"] for p in locked if p["name"] == "rwg2-migrator"] == [project] == [rwg2migrator.__version__]


def run(capsys, *argv):
    code = cli.main([str(a) for a in argv])
    out, err = capsys.readouterr()
    return code, out, err


def refused(capsys, *argv):
    code, out, err = run(capsys, *argv)
    lines = [line for line in err.strip().splitlines() if not line.startswith("scanning: ")]
    assert code == cli.EXIT_REFUSED and out == "" and len(lines) == 1, (code, out, err)
    assert lines[0].startswith("Refused: ") and ". Next: " in lines[0]
    return lines[0]


def test_version_prints_the_version_and_exits_0(capsys):
    with pytest.raises(SystemExit) as e:
        cli.main(["--version"])
    assert e.value.code == 0 and capsys.readouterr().out == f"rwg2-migrator {rwg2migrator.__version__}\n"


def test_inspect_prints_the_checks(capsys, server, cache):
    code, out, _ = run(capsys, "inspect", server, "--cache", cache, "--workers", 1)
    assert code == 0
    for line in (
        "GTNH: 2.9.0 (dreamcraft 2.9.99)",
        "generatorName RWG",
        "biomeListVersion 1",
        "Noise: UNKNOWN (Perlin; kept)",
        "Seed: 1234567890123",
        "team alpha 2",
        "Overworld chunks: 16 (EndlessIDs 16); 0 unpopulated",
        "In use: stopped",
    ):
        assert line in out, line
    assert f"Config: {world.NO_CONFIG}." in out
    os.makedirs(os.path.join(server, "config"))
    open(os.path.join(server, "config", "rwg2.cfg"), "w").close()
    code, out, _ = run(capsys, "inspect", server, "--cache", cache, "--workers", 1)
    assert f"Config: {os.path.join(server, 'config')} (the server's)" in out
    code, out, _ = run(capsys, "inspect", server, "--cache", cache, "--json")
    report = json.loads(out)
    assert code == 0 and report["scan"]["regions_cached"] == 1 and report["claims"] == {"alpha": 2}


def test_inspect_names_the_config_folder_in_use_or_says_there_is_none(capsys, server, cache, tmp_path):
    w = os.path.join(server, "World")
    os.remove(os.path.join(server, "server.properties"))
    code, out, _ = run(capsys, "inspect", w, "--cache", cache)
    assert code == 0 and "Config: no config folder of its own was found next to the world" in out and "--config-dir" in out
    mine = tmp_path / "mine"
    mine.mkdir()
    (mine / "rwg2.cfg").write_text("")
    code, out, _ = run(capsys, "inspect", w, "--cache", cache, "--config-dir", mine, "--json")
    assert code == 0 and (json.loads(out)["config"], json.loads(out)["config_from"]) == (str(mine), "chosen by you")


@pytest.mark.parametrize(
    "case",
    [
        ({"generator": "RWG2_CONTINENTS"}, "already uses RWG2"),
        ({"dreamcraft": "2.7.268"}, "last opened in GTNH 2.8.4 (core mod version 2.7.268)"),
    ],
)
def test_inspect_refuses_wrong_worlds(capsys, tmp_path, cache, case):
    level, words = case
    root = str(tmp_path / "s")
    make_world(root, **level)
    assert words in refused(capsys, "inspect", root, "--cache", cache)


def test_inspect_refuses_missing_files(capsys, server, cache):
    w = os.path.join(server, "World")
    os.remove(os.path.join(w, "data", "GregTech_OregenPattern.dat"))
    assert "GregTech_OregenPattern.dat is missing" in refused(capsys, "inspect", server, "--cache", cache)
    os.remove(os.path.join(w, "level.dat"))
    assert "there is no level.dat" in refused(capsys, "inspect", w, "--cache", cache)


def test_inspect_require_stopped_with_running_child(capsys, server, cache, java_child):
    line = refused(capsys, "inspect", server, "--cache", cache, "--require-stopped")
    assert f"java (pid {java_child.pid}) is running in {server}" in line and "Stop the server" in line
    code, out, _ = run(capsys, "inspect", server, "--cache", cache, "--workers", 1)
    assert code == 0 and "In use: running" in out and f". {cli.FIRST}" in out


def test_inspect_unreadable_java_needs_confirmation(capsys, server, cache, monkeypatch):
    monkeypatch.setattr(procs.psutil, "process_iter", lambda attrs: iter([Unreadable()]))
    assert "the world may be in use: " in refused(capsys, "inspect", server, "--cache", cache, "--require-stopped")
    code, out, _ = run(capsys, "inspect", server, "--cache", cache, "--require-stopped", "--confirm-stopped", "--workers", 1)
    assert code == 0 and "In use: unknown" in out


def test_area_writes_json_and_counts(capsys, tmp_path, cache):
    root = str(tmp_path / "s")
    make_world(root, claims={"alpha": [(1, 1)], "beta": [(3, 3)]}, players={"u1": ("Alice", 8, 8), "u2": ("Bob", 60, 60)})
    out_json = tmp_path / "area.json"
    code, out, _ = run(capsys, "area", root, "--claims", "alpha", "--margin", 1, "--out", out_json, "--cache", cache, "--workers", 1)
    a = json.loads(out_json.read_text())
    assert code == 0 and len(a["core"]) == 1 and len(a["kept"]) == 9 and a["teams"] == ["alpha"] and a["margin"] == 1
    assert a["status"]["dropped"] == 7 and a["removed"]["other_teams_claims"] == {"beta": 1} and a["removed"]["players"] == ["Bob"]
    assert "Protected (kept byte-identical): 1 chunk (" in out and "Kept (protected + 1-chunk margin): 9 chunks" in out
    assert "7 overworld chunks become new terrain" in out and "other teams' claims: beta 1" in out
    assert "margin of 0" in refused(capsys, "area", root, "--claims", "all", "--margin", 0, "--out", out_json, "--cache", cache)
    assert "empty" in refused(capsys, "area", root, "--out", out_json, "--cache", cache)


def test_area_add_remove_and_suggest(capsys, tmp_path, cache):
    root = str(tmp_path / "s")
    make_world(root, chunk_specs={(x, 0): {"inhabited": 100_000 if x == 3 else 0} for x in range(4)})
    out_json = tmp_path / "area.json"
    code, _, _ = run(
        capsys, "area", root, "--claims", "all", "--add", "0,0:0,0", "--remove", "2,1:2,1", "--suggest", "--out", out_json, "--cache", cache
    )
    a = json.loads(out_json.read_text())
    assert code == 0 and a["core"] == [[0, 0], [1, 1], [3, 0]] and a["erased_claims"] == [[2, 1]]


def test_area_suggest_keeps_the_base_part_and_lists_far_parts_as_extras(capsys, tmp_path, cache):
    root = str(tmp_path / "s")
    played = {0, 1, 20, 21}
    make_world(root, claims={"alpha": [(1, 0)]}, chunk_specs={(x, 0): {"inhabited": 100_000 if x in played else 0} for x in range(24)})
    out_json = tmp_path / "area.json"
    base = ["area", root, "--claims", "all", "--suggest", "--out", out_json, "--cache", cache]
    code, out, _ = run(capsys, *base)
    assert code == 0 and json.loads(out_json.read_text())["core"] == [[0, 0], [1, 0]] and "in 1 part;" in out
    assert "  1. 2 chunks, about 288 blocks east (not kept)" in out and area.PARTS_WARNING not in out
    code, out, _ = run(capsys, *base, "--suggest-extra", 1)
    assert code == 0 and json.loads(out_json.read_text())["core"] == [[0, 0], [1, 0], [20, 0], [21, 0]]
    assert "in 2 parts;" in out and f"Warning: {area.parts_warning(2)}" in out and "(kept)" in out
    assert "there is no suggested extra part 2" in refused(capsys, *base, "--suggest-extra", 2)
    one = cli.suggestion_lines({"extra": [{"chunks": 1, "blocks": 1104, "direction": "west"}]}, set())
    assert one[1] == "  1. 1 chunk, about 1,104 blocks west (not kept)"


def test_area_names_graves_and_splits_probably_yours_from_world_generation(capsys, tmp_path, cache):
    root = str(tmp_path / "s")
    specs = {(x, 0): {} for x in range(6)}
    specs[(3, 0)] = {"tes": (grave_te("Bob", 4),)}
    specs[(4, 0)] = {"tes": ("Chest", ("BaseMetaTileEntity", {"mOwnerName": string("Alice")}))}
    specs[(5, 0)] = {"tes": ("Chest", "Chest", "Skull")}
    make_world(root, claims={"alpha": [(0, 0)]}, chunk_specs=specs)
    code, out, _ = run(capsys, "area", root, "--claims", "all", "--out", tmp_path / "a.json", "--cache", cache)
    assert code == 0 and f"  probably yours ({area.YOURS_WHY}): 1 block (GregTech machines 1)" in out
    assert "  1 more in their chunks, of kinds villages and dungeons have too: chests 1" in out
    assert "  Bob's grave at 48, 64, 0 (4 stacks of items)" in out
    assert "  3 chests, skulls… from world generation (villages, dungeons, ruins): not counted" in out


def test_unknown_chunk_layout_is_refused_by_inspect_and_area(capsys, server, cache, tmp_path):
    roots = {(x, z): chunk_root(x, z) for x in range(4) for z in range(4)}
    roots[(2, 3)]["Level"][1]["Sections"][1][1][0]["Blocks16"] = tag(nbt.BYTE_ARRAY, bytes(8192))
    write_region(os.path.join(server, "World", "region", "r.0.0.mca"), roots)
    for cmd in (["inspect", server], ["area", server, "--claims", "all", "--out", tmp_path / "a.json"]):
        line = refused(capsys, *cmd, "--cache", cache, "--workers", 1)
        assert "1 overworld chunks, the first at chunk 2,3, use a chunk layout that is neither EndlessIDs nor NEID" in line
        assert "report the chunk position" in line


def test_chunk_with_lazy_sections_is_read_by_inspect_and_area(capsys, server, cache, tmp_path):
    write_region(os.path.join(server, "World", "region", "r.2.1.mca"), {LAZY_POS: lazy_chunk()[1]})
    code, out, _ = run(capsys, "inspect", server, "--cache", cache, "--workers", 1)
    assert code == 0 and "Overworld chunks: 17 (EndlessIDs 17); 0 unpopulated; 0 unreadable" in out
    rect = "{0},{1}:{0},{1}".format(*LAZY_POS)
    code, out, _ = run(capsys, "area", server, "--add", rect, "--out", tmp_path / "a.json", "--cache", cache)
    assert code == 0 and "unpopulated kept chunks: 0 (protected: 0); unreadable kept chunks: 0" in out


def test_list_saves_without_world(capsys, monkeypatch):
    monkeypatch.setattr(cli.world, "saves", lambda: [{"name": "Base", "path": "/x/saves/Base"}])
    code, out, _ = run(capsys, "inspect")
    assert code == 0 and out == "Base\t/x/saves/Base\n"


def test_module_entry_point(server, cache):
    r = subprocess.run(
        [sys.executable, "-m", "rwg2migrator", "inspect", server, "--cache", cache, "--workers", "1"], capture_output=True, text=True
    )
    assert r.returncode == 0 and "GTNH: 2.9.0" in r.stdout


def test_negative_rectangles_after_a_space(capsys, tmp_path, cache):
    root = str(tmp_path / "s")
    make_world(root, claims={})
    out_json = tmp_path / "area.json"
    code, _, err = run(capsys, "area", root, "--add", "-5,-3:-4,-3", "--remove", "-5,-3:-5,-3", "--out", out_json, "--cache", cache)
    assert code == 0, err
    assert json.loads(out_json.read_text())["core"] == [[-4, -3]]
    assert cli.joined(["--add", "-1,2:3,4", "--margin", "-1", "--add=-2,0:0,0"]) == ["--add=-1,2:3,4", "--margin", "-1", "--add=-2,0:0,0"]


def test_unreadable_side_files_warn_and_continue(capsys, server, cache, tmp_path):
    w = os.path.join(server, "World")
    open(os.path.join(w, "playerdata", "dead.dat"), "wb").close()
    open(os.path.join(w, "data", "rwg_data.dat"), "wb").close()
    code, out, _ = run(capsys, "inspect", server, "--cache", cache, "--workers", 1)
    assert code == 0 and "Players: Alice" in out and "Noise: unknown" in out
    assert out.count("Warning: ") == 2 and "dead.dat cannot be read" in out and "rwg_data.dat cannot be read" in out
    code, out, _ = run(capsys, "area", server, "--claims", "all", "--out", tmp_path / "a.json", "--cache", cache)
    assert code == 0 and "Warning: the player file" in out and json.loads((tmp_path / "a.json").read_text())["warnings"]


def test_broken_level_dat_is_refused_not_a_crash(capsys, server, cache):
    zero_body(os.path.join(server, "World", "level.dat"))
    assert "Restore level.dat from level.dat_old" in refused(capsys, "inspect", server, "--cache", cache)


def test_area_out_must_be_outside_the_world_in_an_existing_folder(capsys, server, cache, tmp_path):
    level = Path(server, "World", "level.dat")
    before = level.read_bytes()
    assert "inside the world folder" in refused(capsys, "area", server, "--claims", "all", "--out", level, "--cache", cache)
    assert level.read_bytes() == before
    assert "does not exist" in refused(capsys, "area", server, "--claims", "all", "--out", tmp_path / "nope" / "a.json", "--cache", cache)


def test_list_saves_shows_a_broken_save(capsys, monkeypatch):
    monkeypatch.setattr(cli.world, "saves", lambda: [{"name": "B", "path": "/x/B", "error": "level.dat cannot be read (x)"}])
    assert run(capsys, "inspect")[1] == "B\t/x/B\t(level.dat cannot be read (x))\n"


def test_world_without_overworld_chunks_is_refused(capsys, server, cache):
    region_file = Path(server, "World", "region", "r.0.0.mca")
    region_file.write_bytes(b"")
    assert "holds no overworld chunks" in refused(capsys, "inspect", server, "--cache", cache)
    region_file.unlink()
    region_file.parent.rmdir()
    out_json = os.path.join(server, "a.json")
    assert "holds no overworld chunks" in refused(capsys, "area", server, "--claims", "all", "--out", out_json, "--cache", cache)


def test_nothing_is_written_inside_the_world(capsys, server, tmp_path, monkeypatch):
    w = Path(server, "World")
    assert "cache folder" in refused(capsys, "inspect", server, "--cache", w / "cache")
    monkeypatch.setenv("RWG2_CACHE", str(w))
    assert "cache folder" in refused(capsys, "inspect", server)
    assert not (w / "cache").exists() and not (w / "scan").exists()
    monkeypatch.delenv("RWG2_CACHE")
    link = w / "area.json"
    link.symlink_to(tmp_path / "elsewhere.json")
    args = ("--claims", "all", "--cache", tmp_path / "c")
    assert "inside the world folder" in refused(capsys, "area", server, "--out", link, *args)
    assert "is a folder" in refused(capsys, "area", server, "--out", tmp_path, *args)


def test_within_treats_other_drives_as_outside(monkeypatch):
    def other_drive(paths):
        raise ValueError("Paths don't have the same drive")

    monkeypatch.setattr(cli.os.path, "commonpath", other_drive)
    assert not cli._within("/d/area.json", "/c/World")


def test_an_unexpected_error_ends_with_where_to_report_it(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli.world, "locate", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    assert cli.main(["inspect", str(tmp_path)]) == cli.EXIT_REFUSED
    err = capsys.readouterr().err
    assert "Traceback" in err and err.rstrip().endswith(f"Next: report it at {ISSUES} with these details.")
