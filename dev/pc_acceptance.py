"""Generator lifecycle checks against the real server pack (M2 acceptance); run on a machine with the pack set up.

  python -I dev/pc_acceptance.py SCENARIO --cache CACHE [--overlay CONFIG_DIR] [--watch DIR ...] [--skip DIR ...]

Scenarios: token, stop, exit, kill, console, silent, foreign, sidefx. Each prints one JSON line of evidence.
"""

import argparse
import ctypes
import json
import os
import re
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import psutil

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rwg2migrator import forgecfg, generator, procs, provision  # noqa: E402

EXIT_LIMIT = 60
WM_CLOSE = 0x0010
POLL = 0.25
HUGE_SLEEP = 24 * 3600


def emit(event):
    print(json.dumps(event), file=sys.stderr, flush=True)


def context(cache):
    s = generator.settings(cache)
    rel = provision.release(s["release"])
    java = provision.choose_java(cache, rel.entry["java"], provision.JavaWanted(s["java"]))
    return generator.Context(cache, rel.name, rel.entry, java, False, emit)


def gone_after(record, since, limit):
    """Seconds from since until the generator process (by pid + create time) is gone, or None."""
    ident = generator.ident(record)
    while time.monotonic() - since < limit:
        if not procs.is_alive(ident):
            return round(time.monotonic() - since, 1)
        time.sleep(POLL)
    return None


def child(a):
    gen = generator.start(context(a.cache), generator.Boot(idle_minutes=a.idle, jvm=tuple(a.jvm)))
    window = ctypes.windll.kernel32.GetConsoleWindow() if os.name == "nt" else 0
    print(json.dumps({"ready": True, "record": gen.record, "console_window": window}), flush=True)
    if a.mode == "exit":
        sys.exit(0)
    if a.mode == "sleep":
        gen.start_heartbeat()
    time.sleep(HUGE_SLEEP)


def spawn_child(a, mode, flags=0):
    args = [sys.executable, "-I", __file__, "child", "--cache", a.cache, "--mode", mode, "--idle", str(a.idle)]
    args += [f"--jvm={j}" for j in a.jvm]
    p = subprocess.Popen(args, stdout=subprocess.PIPE, text=True, creationflags=flags)
    for line in p.stdout:
        if line.startswith("{"):
            ready = json.loads(line)
            return p, ready["record"] | {"console_window": ready["console_window"]}
    raise SystemExit(f"the child exited with {p.wait()} before its generator was ready")


def result(name, **facts):
    print(json.dumps({"scenario": name} | facts), flush=True)


def scenario_token(a):
    gen = generator.start(context(a.cache), generator.Boot())
    cmdline = psutil.Process(gen.record["pid"]).cmdline()
    token = Path(gen.record["folder"], generator.TOKEN_FILE).read_text().strip()
    missing = gen.client.send({"op": "status"})
    wrong = gen.client.send({"op": "status", "token": "0" * len(token)})
    after = gen.call("status")
    world_op = gen.client.send({"op": "generate", "token": token, "offsets": [0, 0, 0, 0], "chunks": [[0, 0]]})
    t0 = time.monotonic()
    code = gen.stop()
    result(
        "token",
        missing=missing,
        wrong=wrong,
        same_connection_status_ok=after["ok"],
        instance=after["instance"],
        gen_op_in_sampler=world_op,
        token_on_cmdline=any(token in arg for arg in cmdline),
        stop_seconds=round(time.monotonic() - t0, 1),
        exit_code=code,
    )


def scenario_stop(a):
    p, record = spawn_child(a, "sleep")
    t0 = time.monotonic()
    out = subprocess.run(
        [sys.executable, "-I", "-m", "rwg2migrator", "generator", "stop", "--cache", a.cache], capture_output=True, text=True
    )
    result(
        "stop",
        cli_out=out.stdout.strip(),
        cli_err=out.stderr.strip()[-300:],
        exited_after=gone_after(record, t0, EXIT_LIMIT * 2),
        child_still_alive=p.poll() is None,
    )
    p.kill()


def scenario_exit(a):
    p, record = spawn_child(a, "exit")
    p.wait()
    result("exit", child_exit_code=p.returncode, exited_after=gone_after(record, time.monotonic(), EXIT_LIMIT * 2))


def scenario_kill(a):
    p, record = spawn_child(a, "sleep")
    t0 = time.monotonic()
    p.kill()
    result("kill", child_exit_code=p.wait(), exited_after=gone_after(record, t0, EXIT_LIMIT * 2))


def scenario_console(a):
    """Close the tool's console window the way its X button does (WM_CLOSE); Windows then ends the tool without atexit."""
    p, record = spawn_child(a, "sleep", subprocess.CREATE_NEW_CONSOLE)
    t0 = time.monotonic()
    posted = ctypes.windll.user32.PostMessageW(record["console_window"], WM_CLOSE, 0, 0)
    try:
        code = p.wait(EXIT_LIMIT)
    except subprocess.TimeoutExpired:
        code = None
    exited = gone_after(record, t0, EXIT_LIMIT * 2)
    result("console", window=record["console_window"], posted=bool(posted), tool_exit_code=code, exited_after=exited)
    if code is None:
        p.kill()


def scenario_silent(a):
    p, record = spawn_child(a, "silent")
    t0 = time.monotonic()
    exited = gone_after(record, t0, a.idle * EXIT_LIMIT + EXIT_LIMIT * 2)
    result("silent", idle_minutes=a.idle, exited_after=exited, child_still_alive=p.poll() is None)
    p.kill()


def dummy_javaw(a):
    src = Path(a.cache, "Dummy.java")
    src.write_text("public class Dummy { public static void main(String[] a) throws Exception { Thread.sleep(86400000L); } }\n")
    p = subprocess.Popen([a.javaw, str(src)], creationflags=getattr(subprocess, "DETACHED_PROCESS", 0))
    time.sleep(3)
    return p


def scenario_foreign(a):
    dummy = dummy_javaw(a)
    real = psutil.Process(dummy.pid).create_time()
    rows = []
    for create_time, instance in ((real + 7.5, "0" * 32), (real, "f" * 32)):
        record = {"pid": dummy.pid, "create_time": create_time, "instance": instance, "port": 1, "folder": a.cache}
        Path(a.cache, generator.RECORD_FILE).write_text(json.dumps(record))
        out = subprocess.run(
            [sys.executable, "-I", "-m", "rwg2migrator", "generator", "stop", "--cache", a.cache], capture_output=True, text=True
        )
        rows.append({"create_time_matches": create_time == real, "cli": out.stdout.strip(), "dummy_alive": dummy.poll() is None})
    name = psutil.Process(dummy.pid).name()
    result("foreign", dummy_pid=dummy.pid, dummy_name=name, runs=rows)
    print(json.dumps({"dummy": {"pid": dummy.pid, "create_time": real}}), flush=True)


WATCHER = Path(__file__).with_name("pc-watch.ps1")
WATCH_READY_SECONDS = 120
EVENT = re.compile(r"^(\S+) (\w+) (.*)$")


def start_watcher(a, folder):
    log, stop = Path(folder, "fsw.log"), Path(folder, "fsw.stop")
    for f in (log, stop):
        f.unlink(missing_ok=True)
    args = ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(WATCHER), "-Log", str(log), "-Stop", str(stop)]
    p = subprocess.Popen([*args, "-Exclude", a.cache, "-Roots", ";".join(a.watch)])
    t0 = time.monotonic()
    while "WATCHING" not in (log.read_text(errors="replace") if log.exists() else ""):
        if p.poll() is not None or time.monotonic() - t0 > WATCH_READY_SECONDS:
            raise SystemExit(f"the file watcher did not start: {log}")
        time.sleep(POLL)
    return p, log, stop


def stop_watcher(p, log, stop):
    stop.write_text("stop")
    p.wait(WATCH_READY_SECONDS)
    return log.read_text(errors="replace").splitlines()


def outside(lines, skip):
    """Watcher events whose path is outside the skipped folders, and the watcher's own overflow or unwatched lines."""
    events, problems = Counter(), []
    for line in lines:
        m = EVENT.match(line)
        if not m or m.group(2) in ("WATCHING", "END"):
            continue
        if m.group(2) in ("OVERFLOW", "UNWATCHED"):
            problems.append(line)
        elif not any((m.group(3).lower() + "\\").startswith(s.lower().rstrip("\\") + "\\") for s in skip):
            events[f"{m.group(2)} {m.group(3)}"] += 1
    return dict(events), problems


HOSTILE_SU = {
    ("backups", "enable_backups"): ("B", "true"),
    ("backups", "backup_timer"): ("S", "1m"),
    ("backups", "need_online_players"): ("B", "false"),
    ("auto_shutdown", "enabled"): ("B", "true"),
}
SHUTDOWN_TIMES = re.compile(r"(S:times <\n)(.*?)(\s*>)", re.DOTALL)
SHUTDOWN_IN = (3, 4)


def plant_su(folder, outside):
    """A ServerUtilities config that backs up into outside every minute and shuts down 3 and 4 minutes from now (during the hold)."""
    path = Path(folder, generator.SU_CFG)
    forgecfg.set_keys(str(path), HOSTILE_SU | {("backups", "backup_folder_path"): ("S", outside)})
    times = "".join(time.strftime("        %H:%M\n", time.localtime(time.time() + 60 * m)) for m in SHUTDOWN_IN)
    path.write_text(SHUTDOWN_TIMES.sub(lambda m: m.group(1) + times.rstrip("\n") + m.group(3), path.read_text(), count=1))
    return times.split()


def scenario_sidefx(a):
    """Java discovery plus one boot with the user's config overlay, under a file watcher on everything but the cache."""
    folder = provision.pack_dir(a.cache, generator.settings(a.cache)["release"])
    shutdown_at = plant_su(folder, a.plant_su) if a.plant_su else []
    planted = Path(folder, "config", "planted-by-an-earlier-boot.cfg")
    planted.parent.mkdir(exist_ok=True)
    planted.write_text("x")
    backups_before = sorted(os.listdir(Path(folder, "backups"))) if Path(folder, "backups").exists() else []
    watch_dir = Path(a.cache, "sidefx")
    watch_dir.mkdir(exist_ok=True)
    watcher = start_watcher(a, watch_dir)
    t0 = time.monotonic()
    ctx = context(a.cache)
    found = provision.choose_java(a.cache, provision.release(ctx.release).entry["java"], provision.SEARCH)
    gen = generator.start(ctx, generator.Boot(overlay=a.overlay))
    planted_gone = not planted.exists()
    time.sleep(a.hold)
    code = gen.stop()
    seconds = round(time.monotonic() - t0, 1)
    events, problems = outside(stop_watcher(*watcher), [a.cache, *a.skip])
    log = Path(folder, generator.CONSOLE_LOG).read_text(encoding="utf-8", errors="replace").splitlines()
    result(
        "sidefx",
        watched_seconds=seconds,
        java_found=found.path,
        exit_code=code,
        planted_gone=planted_gone,
        overlay_files_in_config=sum(len(f) for _, _, f in os.walk(a.overlay)),
        written_outside_cache=events,
        watcher_problems=problems,
        backups_before=backups_before,
        backups_after=sorted(os.listdir(Path(folder, "backups"))) if Path(folder, "backups").exists() else [],
        backup_log_lines=[ln for ln in log if "backup" in ln.lower()][:10],
        shutdown_log_lines=[ln for ln in log if "auto-shutdown" in ln.lower() or "autoshutdown" in ln.lower()][:10],
        su_cfg={k[1]: forgecfg.get(str(Path(folder, generator.SU_CFG)), k) for k in generator.SU_FORCED},
        planted_su={
            "backup_folder": a.plant_su,
            "shutdown_at": shutdown_at,
            "backup_folder_exists": bool(a.plant_su) and os.path.exists(a.plant_su),
        },
    )


SCENARIOS = {name[len("scenario_") :]: fn for name, fn in globals().items() if name.startswith("scenario_")}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("scenario", choices=[*SCENARIOS, "child"])
    p.add_argument("--cache", required=True)
    p.add_argument("--mode", default="sleep")
    p.add_argument("--idle", type=float, default=generator.IDLE_MINUTES)
    p.add_argument("--overlay")
    p.add_argument("--javaw", default=r"C:\Program Files\Microsoft\jdk-11.0.16.101-hotspot\bin\javaw.exe")
    p.add_argument("--watch", action="append", default=[])
    p.add_argument("--skip", action="append", default=[])
    p.add_argument("--hold", type=float, default=EXIT_LIMIT)
    p.add_argument("--jvm", action="append", default=[], help="an extra JVM argument for the generator (a hang to reproduce)")
    p.add_argument("--plant-su", metavar="OUTSIDE_DIR", help="sidefx: plant a ServerUtilities config that backs up into OUTSIDE_DIR")
    a = p.parse_args(argv)
    a.cache = os.path.abspath(a.cache)
    (child if a.scenario == "child" else SCENARIOS[a.scenario])(a)


if __name__ == "__main__":
    main()
