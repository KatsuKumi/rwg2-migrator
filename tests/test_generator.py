import hashlib
import json
import os
import stat
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import psutil
import pytest

from rwg2migrator import Refusal, cli, forgecfg, generator, procs, provision

FAKE = Path(__file__).parent / "fake_probe.py"
JVM_HEADROOM = generator.jvm_headroom
SU = "auto_shutdown {\n    B:enabled=true\n}\n\nbackups {\n    S:backup_folder_path=/srv/backups/server/\n    B:enable_backups=true\n}\n"
ENTRY = {"rwg2": "2.0.6", "files": "files-2.9.0.json", "java": {"min": 17, "max": 26}}
OFFSETS = (-12000, 8000, 2048, 6144)
FAST_STOP = 5
MAX_NICE = 19


@pytest.fixture
def pack(tmp_path):
    cache = tmp_path / "cache"
    provision.claim_cache(str(cache))
    folder = cache / "packs" / "2.9.0"
    (folder / "config.pristine" / "sub").mkdir(parents=True)
    (folder / "config.pristine" / "a.cfg").write_text("pristine\n")
    (folder / "mods").mkdir()
    (folder / "mods" / "rwg2probe-0.1.0.jar").write_bytes(b"old")
    (folder / "serverutilities").mkdir()
    (folder / "serverutilities" / "serverutilities.cfg").write_text(SU)
    (folder / "eula.txt").write_text("eula=false\n")
    return cache, folder


@pytest.fixture
def fake_java(tmp_path):
    if os.name == "nt":
        pytest.skip("the stand-in for Java is a shell script; dev/pc_acceptance.py checks the generator's lifecycle on Windows")
    exe = tmp_path / "bin" / "java"
    exe.parent.mkdir()
    exe.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{FAKE}" "$@"\n')
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    return provision.Java(str(exe), "25.0.5", 25, "amd64", "64")


@pytest.fixture
def ctx(pack, fake_java, monkeypatch):
    monkeypatch.setattr(generator, "check_pack", lambda *a: None)
    events = []
    c = generator.Context(str(pack[0]), "2.9.0", ENTRY, fake_java, True, events.append)
    c.events = events
    return c


def test_prepare_rebuilds_config_and_forces_safe_settings(pack, tmp_path):
    _, folder = pack
    overlay = tmp_path / "user-config"
    overlay.mkdir()
    (overlay / "a.cfg").write_text("user\n")
    (folder / "config").mkdir()
    (folder / "config" / "planted-by-last-boot.cfg").write_text("x")
    token = generator.prepare(str(folder), generator.Boot(kind=generator.GEN, world="gen-1", overlay=str(overlay), offsets=OFFSETS), True)
    assert sorted(os.listdir(folder / "config")) == ["a.cfg", "rwg2.cfg", "sub"]
    assert (folder / "config" / "a.cfg").read_text() == "user\n" and (overlay / "a.cfg").read_text() == "user\n"
    su = str(folder / generator.SU_CFG)
    assert forgecfg.get(su, ("backups", "enable_backups")) == "false"
    assert forgecfg.get(su, ("backups", "backup_folder_path")) == "./backups/"
    assert forgecfg.get(su, ("auto_shutdown", "enabled")) == "false"
    rwg2 = str(folder / generator.RWG2_CFG)
    got = [int(forgecfg.get(rwg2, (generator.RWG2_CATEGORY, k))) for k in generator.OFFSET_KEYS]
    assert got == list(OFFSETS)
    assert os.listdir(folder / "mods") == [generator.PROBE_JAR]
    assert (folder / "eula.txt").read_text().endswith("eula=true\n")
    assert (folder / generator.TOKEN_FILE).read_text() == token and len(token) == 2 * generator.TOKEN_BYTES
    assert os.name == "nt" or stat.S_IMODE((folder / generator.TOKEN_FILE).stat().st_mode) == 0o600
    props = (folder / "server.properties").read_text()
    assert "server-ip=127.0.0.1\n" in props and "level-name=gen-1\n" in props and "level-type=FLAT" not in props
    assert "server-port=0\n" in props, "the server binds a port the OS picks then; a port picked before the boot can be taken meanwhile"


def _lock(*paths):
    for p in paths:
        os.chmod(p, stat.S_IREAD | (stat.S_IEXEC if p.is_dir() else 0))


def test_a_read_only_overlay_boots_every_time_and_cleans(pack, tmp_path):
    cache, folder = pack
    overlay = tmp_path / "user-config"
    (overlay / "sub").mkdir(parents=True)
    forgecfg.set_keys(str(overlay / "rwg2.cfg"), generator.offset_edits((0, 0, 0, 0)))
    (overlay / "sub" / "locked.cfg").write_text("user\n")
    _lock(overlay / "rwg2.cfg", overlay / "sub" / "locked.cfg", overlay / "sub")
    boot = generator.Boot(kind=generator.GEN, world="gen-1", overlay=str(overlay), offsets=OFFSETS)
    for _ in range(2):
        generator.prepare(str(folder), boot, True)
        (folder / "config" / "planted.cfg").write_text("x")
    rwg2 = str(folder / generator.RWG2_CFG)
    assert [int(forgecfg.get(rwg2, (generator.RWG2_CATEGORY, k))) for k in generator.OFFSET_KEYS] == list(OFFSETS)
    assert os.access(folder / "config" / "sub" / "locked.cfg", os.W_OK) and os.access(folder / "config" / "sub", os.W_OK)
    assert not os.access(overlay / "sub" / "locked.cfg", os.W_OK)
    _lock(folder / "config" / "sub" / "locked.cfg", folder / "config" / "sub")
    assert cli.main(["clean", "--generator", "--cache", str(cache)]) == 0
    assert os.listdir(cache) == [provision.CACHE_MARKER]


def test_without_the_eula_nothing_is_written(pack):
    _, folder = pack
    with pytest.raises(Refusal, match="EULA has not been accepted") as e:
        generator.prepare(str(folder), generator.Boot(), False)
    assert "--accept-eula" in e.value.action
    assert (folder / "eula.txt").read_text() == "eula=false\n"
    assert not (folder / "config").exists() and not (folder / generator.TOKEN_FILE).exists()


def test_an_accepted_eula_stays_accepted(pack):
    _, folder = pack
    generator.prepare(str(folder), generator.Boot(), True)
    generator.prepare(str(folder), generator.Boot(), False)
    assert "level-type=FLAT\n" in (folder / "server.properties").read_text()


def test_the_launch_line_holds_the_marker_but_not_the_token(pack):
    _, folder = pack
    token = generator.prepare(str(folder), generator.Boot(), True)
    args = generator.jvm_args("java", generator.Boot(heap="4G"), "abc123")
    assert token not in " ".join(args)
    assert "-Drwg2probe.instance=abc123" in args and f"-Drwg2probe.parentPid={os.getpid()}" in args and "-Xmx4G" in args
    assert args[-3:] == ["-jar", "lwjgl3ify-forgePatches.jar", "nogui"] and "-Dfml.queryResult=confirm" in args
    quiet = {"-XX:-UsePerfData", "-Djdk.disableLastUsageTracking=true"}
    assert quiet | {"-Djava.io.tmpdir=tmp", "-Djdk.net.unixdomain.tmpdir=tmp"} <= set(args)
    assert not any(str(folder) in a or os.sep + "tmp" in a for a in args), "a path the JVM gets must be relative to the pack folder"


def test_heap_sizes():
    assert generator.heap_bytes("3G") == 3 * generator.GB and generator.heap_bytes("4096m") == 4 * generator.GB
    assert generator._xmx(["java", "-Xms1G", "-Xmx16G", "-jar", "x"]) == 16 * generator.GB
    with pytest.raises(Refusal, match="not a Java heap size"):
        generator.heap_bytes("lots")


def test_low_memory_refuses_the_boot_with_a_plain_message(monkeypatch):
    monkeypatch.setattr(generator.psutil, "virtual_memory", lambda: SimpleNamespace(available=int(2.1 * generator.GB)))
    monkeypatch.setattr(generator, "jvm_headroom", lambda: 0)
    with pytest.raises(Refusal) as e:
        generator.require_ram("3G")
    assert e.value.reason == "The generator needs 4.0 GB free; 2.1 GB is free now"
    assert e.value.action.startswith("Close GTNH (or other Java programs)")


def test_running_jvms_count_what_they_may_still_claim(monkeypatch):
    server = SimpleNamespace(cmdline=lambda: ["java", "-Xmx16G"], memory_info=lambda: SimpleNamespace(rss=12 * generator.GB))
    full = SimpleNamespace(cmdline=lambda: ["javaw", "-Xmx4G"], memory_info=lambda: SimpleNamespace(rss=5 * generator.GB))
    monkeypatch.setattr(generator.procs, "java_processes", lambda: iter([server, full]))
    assert JVM_HEADROOM() == 4 * generator.GB


def test_boot_answers_by_token_and_stops(ctx):
    gen = generator.start(ctx, generator.Boot(heap="1G"))
    record = generator.read_record(ctx.cache)
    assert record == gen.record and record["port"] and record["kind"] == generator.SAMPLER
    cmdline = " ".join(psutil.Process(record["pid"]).cmdline())
    token = (Path(record["folder"]) / generator.TOKEN_FILE).read_text()
    assert token not in cmdline and generator.MARKER + record["instance"] in cmdline
    assert gen.client.send({"op": "status", "token": "wrong"})["ok"] is False
    assert gen.client.send({"op": "status"})["ok"] is False
    assert gen.call("echo", value=7)["echo"] == 7
    assert [e["event"] for e in ctx.events if e["step"] == "boot"][-1] == "ready"
    assert gen.stop() == 0 and generator.read_record(ctx.cache) is None


def test_the_ready_time_counts_the_whole_boot_as_the_preview_summary_does(ctx, monkeypatch):
    pause = 0.5
    monkeypatch.setattr(generator, "check_pack", lambda *a: time.sleep(pause))
    t0 = time.monotonic()
    gen = generator.start(ctx, generator.Boot(heap="1G"))
    took = time.monotonic() - t0
    ready = [e for e in ctx.events if e["step"] == "boot" and e["event"] == "ready"][-1]["seconds"]
    assert gen.stop() == 0 and pause <= ready <= round(took, 1)


def test_a_boot_that_dies_names_the_log(ctx, monkeypatch):
    monkeypatch.setenv("FAKE_PROBE", "die-at-boot")
    with pytest.raises(Refusal, match="exit code 7.*fake boot failure"):
        generator.start(ctx, generator.Boot(heap="1G"))


def test_a_generator_that_ignores_stop_is_killed(ctx, monkeypatch):
    monkeypatch.setenv("FAKE_PROBE", "ignore-stop")
    gen = generator.start(ctx, generator.Boot(heap="1G"))
    started = time.monotonic()
    code = gen.stop(timeout=FAST_STOP)
    assert code not in (0, None) and not psutil.pid_exists(gen.record["pid"]) or psutil.Process(gen.record["pid"]).status() == "zombie"
    assert time.monotonic() - started < 3 * FAST_STOP


def test_a_record_whose_pid_is_another_process_is_never_signalled(tmp_path, java_child):
    cache = tmp_path / "c"
    cache.mkdir()
    real = psutil.Process(java_child.pid).create_time()
    for create_time, instance in ((real + 7, "abc"), (real, "not-in-its-command-line")):
        record = {"pid": java_child.pid, "create_time": create_time, "instance": instance, "port": 1, "folder": str(tmp_path)}
        (cache / generator.RECORD_FILE).write_text(json.dumps(record))
        assert generator.running(str(cache)) is None
        assert generator.stop_running(str(cache)) == (False, None)
        assert java_child.poll() is None and not (cache / generator.RECORD_FILE).exists()


def test_the_generator_lock(tmp_path, java_child):
    cache = tmp_path / "c"
    cache.mkdir()
    other = {"pid": java_child.pid, "create_time": psutil.Process(java_child.pid).create_time(), "url": "http://127.0.0.1:8123/?t=x"}
    (cache / generator.LOCK_FILE).write_text(json.dumps(other))
    with pytest.raises(Refusal, match="the app is already running at http://127.0.0.1:8123") as refused, generator.lock(str(cache)):
        pass
    assert "t=x" not in str(refused.value) + refused.value.action
    with generator.lock(str(cache / "mine"), "http://127.0.0.1:8124/?t=secret"):
        assert "secret" not in (cache / "mine" / generator.LOCK_FILE).read_text()
    (cache / generator.LOCK_FILE).write_text(json.dumps(other | {"create_time": 1.0}))
    with generator.lock(str(cache)):
        assert json.loads((cache / generator.LOCK_FILE).read_text())["pid"] == os.getpid()
    assert not (cache / generator.LOCK_FILE).exists()


class FakeSampler:
    """Answers sample like the probe: config-major records with a palette of the biome names it used."""

    def call(self, op, **r):
        n, m = len(r["points"]), len(r["configs"])
        recs = np.zeros(n * m, generator.RECORD)
        flat = np.array([c[0] + p[0] for c in r["configs"] for p in r["points"]])
        recs["h"], recs["meta"], recs["ocean"] = flat * 0.5, flat % 7, flat % 2
        names = sorted({f"b{v % 3}" for v in flat}, reverse=True)
        recs["palette"] = [names.index(f"b{v % 3}") for v in flat]
        import base64

        return {"data": base64.b64decode(base64.b64encode(recs.tobytes())), "palette": names, "n": n, "m": m}


def test_the_sample_digest_does_not_depend_on_the_batch_size(monkeypatch):
    configs = np.array([[1, i, 0, 0, 0] for i in range(10)])
    points = np.array([[x, 0] for x in range(5)])
    whole = generator.sample_digest(FakeSampler(), 1, configs, points)
    monkeypatch.setattr(generator, "BATCH_RECORDS", 15)
    assert generator.sample_digest(FakeSampler(), 1, configs, points) == whole


def test_the_selftest_inputs_are_a_frozen_ring_and_grid():
    spec, points, configs = generator.selftest()
    lo, hi = spec["ring"]["x0"] - 1, spec["ring"]["x0"] + spec["ring"]["size"]
    assert len(points) == 4 * (hi - lo) and all(lo in p or hi in p for p in points.tolist())
    assert ((points >= lo) & (points <= hi)).all() and len(np.unique(points, axis=0)) == len(points)
    assert configs.shape == (1000, 5) and spec["render"]["config"] in configs[:, 1:].tolist()
    assert hashlib.sha256(points.tobytes() + configs.tobytes()).hexdigest()[:16] == "7dfcebdf7b4afd9e"


def test_generate_goes_row_by_row_in_batches(monkeypatch):
    seen = []
    gen = SimpleNamespace(call=lambda op, **r: seen.append(r) or {"existed": 1, "generated": len(r["chunks"]) - 1, "populatedByProbe": 2})
    monkeypatch.setattr(generator, "GEN_BATCH", 4)
    cells = {(x, z) for x in range(3) for z in range(3)}
    events = []
    totals = generator.generate(gen, OFFSETS, cells, events.append)
    order = [tuple(c) for r in seen for c in r["chunks"]]
    assert order == sorted(cells, key=lambda c: (c[1], c[0])) and [len(r["chunks"]) for r in seen] == [4, 4, 1]
    assert all(r["offsets"] == list(OFFSETS) for r in seen)
    assert totals == {"existed": 3, "generated": 6, "populatedByProbe": 6} and events[-1]["done"] == 9


def test_cli_setup_without_eula_stops_before_eula_txt(capsys, pack, fake_java, monkeypatch):
    cache, folder = pack
    monkeypatch.setattr(provision, "install", lambda *a: pytest.fail("refused before any download"))
    monkeypatch.setattr(provision, "java_info", lambda p: fake_java)
    code = cli.main(["generator", "setup", "--cache", str(cache), "--java", fake_java.path])
    err = capsys.readouterr().err.strip().splitlines()[-1]
    assert code == cli.EXIT_REFUSED and err.startswith("Refused: the generator is a Minecraft server, and the Minecraft EULA")
    assert (folder / "eula.txt").read_text() == "eula=false\n" and not (cache / generator.RECORD_FILE).exists()
    assert not (cache / generator.SETTINGS_FILE).exists()


def test_cli_setup_refused_for_memory_keeps_no_settings(capsys, pack, fake_java, monkeypatch):
    cache, folder = pack
    monkeypatch.setattr(generator, "check_pack", lambda *a: None)
    monkeypatch.setattr(provision, "install", lambda *a: str(folder))
    monkeypatch.setattr(provision, "java_info", lambda p: fake_java)
    code = cli.main(["generator", "setup", "--cache", str(cache), "--java", fake_java.path, "--heap", "4000G", "--accept-eula"])
    err = capsys.readouterr().err.strip().splitlines()[-1]
    assert code == cli.EXIT_REFUSED and err.startswith("Refused: The generator needs 4001.0 GB free;")
    assert not (cache / generator.SETTINGS_FILE).exists()


def test_cli_status_and_clean(capsys, pack):
    cache, folder = pack
    assert cli.main(["generator", "status", "--cache", str(cache)]) == 0
    assert "installed: 2.9.0" in capsys.readouterr().out
    (cache / "downloads").mkdir()
    (cache / "scan").mkdir()
    assert cli.main(["clean", "--generator", "--cache", str(cache)]) == 0
    assert sorted(os.listdir(cache)) == [provision.CACHE_MARKER, "scan"]
    assert cli.main(["generator", "stop", "--cache", str(cache)]) == 0
    assert "No generator was running" in capsys.readouterr().out


def test_process_identity_rejects_a_bare_pid(java_child):
    assert not procs.is_alive(procs.ProcessId(java_child.pid, psutil.Process(java_child.pid).create_time(), generator.MARKER + "x"))


def test_another_process_can_stop_it_while_the_owner_stays_connected(ctx):
    owner = generator.start(ctx, generator.Boot(heap="1G"))
    owner.start_heartbeat(every=0.05)
    started = time.monotonic()
    was, code = generator.stop_running(ctx.cache)
    assert was and code in (0, None) and not owner.alive() and owner.proc.wait(FAST_STOP) == 0
    assert time.monotonic() - started < FAST_STOP


def test_clean_and_setup_refuse_a_folder_the_tool_did_not_make(capsys, tmp_path):
    home = tmp_path / "fakehome"
    files = [home / "downloads" / "tax-return.pdf", home / "java" / "myproject" / "Main.java", home / "Documents" / "a.txt"]
    for f in files:
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text("mine")
    assert cli.main(["clean", "--generator", "--cache", str(home)]) == cli.EXIT_REFUSED
    assert "is not a cache folder of rwg2-migrator" in capsys.readouterr().err
    assert cli.main(["generator", "setup", "--cache", str(home), "--accept-eula"]) == cli.EXIT_REFUSED
    assert "already holds other files" in capsys.readouterr().err
    assert all(f.read_text() == "mine" for f in files) and sorted(os.listdir(home)) == ["Documents", "downloads", "java"]
    assert cli.main(["clean", "--generator", "--cache", str(tmp_path / "never-made")]) == 0
    assert "Removed: nothing" in capsys.readouterr().out and not (tmp_path / "never-made").exists()


@pytest.mark.skipif(os.name == "nt", reason="niceness is POSIX")
def test_a_tool_already_running_below_normal_still_spawns(tmp_path):
    code = (
        "import os, sys; os.nice(15); from rwg2migrator import generator; "
        "p = generator.spawn([sys.executable, '-c', 'import os; print(os.nice(0))'], sys.argv[1]); print(p.wait())"
    )
    out = subprocess.run([sys.executable, "-c", code, str(tmp_path)], capture_output=True, text=True, check=True)
    niced = min(os.nice(0) + 15, MAX_NICE)
    assert out.stdout.split() == ["0"] and (tmp_path / generator.CONSOLE_LOG).read_text().split() == [str(niced)]


def test_a_config_folder_whose_rwg2_cfg_lacks_the_offset_lines_is_refused_by_its_own_name(pack, tmp_path):
    _, folder = pack
    overlay = tmp_path / "user-config"
    overlay.mkdir()
    (overlay / "rwg2.cfg").write_text("# Configuration file\n\ngeneral {\n    B:x=true\n}\n")
    boot = generator.Boot(kind=generator.GEN, world="gen-1", overlay=str(overlay), offsets=OFFSETS)
    with pytest.raises(Refusal, match="has a setting twice, or is missing one the tool writes") as e:
        generator.prepare(str(folder), boot, True)
    assert e.value.reason.startswith(str(overlay / "rwg2.cfg")) and "Start the server or game once" in e.value.action
