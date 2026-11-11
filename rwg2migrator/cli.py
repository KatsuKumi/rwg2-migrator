import argparse
import json
import os
import sys
import traceback
from pathlib import Path

import numpy as np

from . import ISSUES, Refusal, __version__, area, cli_apply, cli_search, generator, jobs, procs, provision, scan, world
from .region import remove_file, rmtree, write_atomic

EXIT_REFUSED = 1
PROGRESS_STEPS = 10
TOP_TYPES = 12
TOP_WORDS = 3
RECT_FLAGS = ("--add", "--remove")
FIRST = "Close the game or stop the server first, and keep it closed until Apply is done"


def _n(v):
    return f"{v:,}"


def _progress(done, total):
    if done == total or done % max(total // PROGRESS_STEPS, 1) == 0:
        print(f"scanning: {done}/{total} region files", file=sys.stderr, flush=True)


def _within(path, folder):
    a, b = os.path.normcase(path), os.path.normcase(os.path.realpath(folder))
    try:
        return os.path.commonpath([a, b]) == b
    except ValueError:
        return False


def _scan(place, args, progress=_progress):
    cache = args.cache or scan.cache_root()
    if _within(os.path.realpath(cache), place.world):
        raise Refusal(f"the cache folder {cache} is inside the world folder", "Pass --cache a folder outside the world, or leave it out")
    sc = scan.scan(place.world, cache, args.workers, progress)
    t = sc.chunk_table()
    if not len(t["cx"]):
        raise Refusal(
            f"{os.path.join(place.world, 'region')} holds no overworld chunks",
            "Check that this is the world the game or server plays; a world that was opened once has region files",
        )
    unknown = np.flatnonzero((t["status"] == scan.OK) & (t["format"] == scan.OTHER_FORMAT))
    if len(unknown):
        raise Refusal(
            f"{_n(len(unknown))} overworld chunks, the first at chunk {t['cx'][unknown[0]]},{t['cz'][unknown[0]]}, "
            "use a chunk layout that is neither EndlessIDs nor NEID",
            f"This version cannot read that layout safely; report the chunk position at {ISSUES}",
        )
    return sc


def _side_files(place, level):
    """rwg_data, players, and a warning for each side file the game would skip."""
    rwg_data = world.rwg_data(place.world)
    found, problems = world.players(place.world, place.root, level)
    return rwg_data, found, problems + ([rwg_data["error"]] if "error" in rwg_data else [])


def _usage(place, args):
    usage = procs.remote() if args.remote else procs.in_use(place.root, place.world)
    if args.require_stopped:
        procs.require_stopped(usage, args.confirm_stopped)
    return usage


def world_report(place, args):
    """(level, report) of everything inspect shows but the scan."""
    level, report = world.check(place.world)
    usage = _usage(place, args)
    claims = world.claims(place.world)
    rwg_data, found, warnings = _side_files(place, level)
    return level, report | {
        "world": place.world,
        "kind": place.kind,
        "root": place.root,
        "config": place.config,
        "config_from": world.config_from(place, args.config_dir),
        "seed": level["Data"]["RandomSeed"],
        "spawn": list(world.spawn(level)),
        "rwg_data": rwg_data,
        "claims": {t: len(c) for t, c in claims.items()},
        "players": found,
        "warnings": warnings,
        "in_use": {"state": usage.state, "detail": usage.detail},
        "onedrive": world.on_onedrive(place.world),
    }


def inspect(args):
    if not args.world:
        return list_saves(args)
    place = world.with_config(world.locate(args.world), args.config_dir)
    _, report = world_report(place, args)
    report["scan"] = _scan(place, args).summary()
    print(json.dumps(report, indent=1) if args.json else "\n".join(inspect_lines(report)))
    return 0


def _noise(rd):
    noise = rd["noiseImplementation"]
    if "error" in rd:
        return "unknown (see the warning below)"
    if noise is None:
        return "no rwg_data.dat (Perlin)"
    if noise in world.PERLIN:
        return f"{noise} (Perlin; kept)"
    return f"{noise} (not Perlin: the migration deletes rwg_data.dat)"


def _mods_lines(report):
    diff = report["mod_diff"]
    beyond = ", ".join(f"{m} {v}" for m, v in diff["beyond_pack"].items()) or "none"
    lines = [f"Mods: {report['mods']}; beyond the GTNH {report['gtnh']} pack: {beyond}"]
    if diff["beyond_pack"]:
        lines.append("  If one of these adds biomes or world generation, the new terrain the tool makes will not contain its features.")
    if diff["other_version"]:
        other = ", ".join(f"{m} {a} (pack {b})" for m, (a, b) in diff["other_version"].items())
        lines.append(f"  Pack mods at another version: {other}")
    if diff["not_in_world"]:
        lines.append(f"  Pack mods not in this world: {', '.join(diff['not_in_world'])}")
    return lines


def _chunks_line(s):
    formats = ", ".join(f"{f} {_n(n)}" for f, n in sorted(s["formats"].items(), key=lambda kv: -kv[1]))
    return (
        f"Overworld chunks: {_n(s['chunks'])} ({formats}); {_n(s['unpopulated'])} unpopulated; "
        f"{_n(s['unreadable'])} unreadable; {_n(s['portal_blocks'])} nether-portal blocks"
    )


def inspect_lines(r):
    s, rd = r["scan"], r["rwg_data"]
    claims = ", ".join(f"team {t} {_n(n)}" for t, n in r["claims"].items()) or "none"
    players = ", ".join(f"{p['name']} (dim {p['dim']} at {p['pos'][0]:.0f}, {p['pos'][2]:.0f})" for p in r["players"]) or "none"
    lines = [
        f"World: {r['world']} ({r['kind']}; folder checked for a running java: {r['root']})",
        f"GTNH: {r['gtnh']} (dreamcraft {r['dreamcraft']})",
        f"Generator: generatorName {r['generator']} (legacy RWG, can be migrated)",
        f"Biome lists: biomeListVersion {rd['biomeListVersion']} (reported only; RWG2 does not read it)",
        f"Noise: {_noise(rd)}",
        f"Seed: {r['seed']}; world spawn {r['spawn'][0]}, {r['spawn'][1]}",
        _config_line(r),
        *_mods_lines(r),
        f"Claims (ServerUtilities, overworld): {jobs.count(sum(r['claims'].values()), 'chunk')}: {claims}",
        f"Players: {players}",
        _chunks_line(s),
        f"Scan: {s['seconds']:.1f} s ({s['regions_read']} region files read, {s['regions_cached']} from the cache)",
        f"In use: {r['in_use']['state']} ({r['in_use']['detail']})" + ("" if r["in_use"]["state"] == procs.STOPPED else f". {FIRST}"),
        *_warning_lines(r["warnings"]),
    ]
    if r["onedrive"]:
        lines.append(
            "Warning: this world is in a OneDrive folder; OneDrive holds files open and can upload half-written ones. Pause syncing."
        )
    return lines


def _config_line(r):
    if r["config"]:
        return f"Config: {r['config']} ({r['config_from']})"
    return f"Config: {world.NO_CONFIG}. Pass --config-dir to use the config folder of the server or game instance that will run it"


def _warning_lines(warnings):
    return [f"Warning: {w}" for w in warnings]


def list_saves(args):
    found = world.saves()
    if args.json:
        print(json.dumps(found, indent=1))
    elif not found:
        print("No single-player saves found in the usual launcher folders. Pass the path of a world, a save or a server folder.")
    for s in [] if args.json else found:
        print(f"{s['name']}\t{s['path']}" + (f"\t({s['error']})" if s.get("error") else ""))
    return 0


def _suggested(args, sug):
    extra = sug["extra"] if sug else []
    bad = [n for n in args.suggest_extra if not 1 <= n <= len(extra)]
    if bad:
        pick = f"Pick one of 1 to {len(extra)}, as the area output lists them" if extra else "This area has no suggested extra parts"
        raise Refusal(f"there is no suggested extra part {bad[0]}", pick if extra else f"{pick}; leave out --suggest-extra")
    picked = set().union(*(map(tuple, extra[n - 1]["cells"]) for n in args.suggest_extra))
    return picked | (set(map(tuple, sug["main"])) if args.suggest else set())


def _painted(args, sug):
    painted = set().union(*(area.rect(r) for r in args.add))
    if args.chunks:
        painted |= area.chunk_file(args.chunks)
    return painted | _suggested(args, sug)


def suggestion_lines(sug, picked):
    """The suggested extra parts, off unless picked with --suggest-extra."""
    if not sug or not sug["extra"]:
        return []
    out = ["Suggested extra parts, far from your base (add one with --suggest-extra N):"]
    for n, p in enumerate(sug["extra"], 1):
        state = "kept" if n in picked else "not kept"
        out.append(f"  {n}. {jobs.count(p['chunks'], 'chunk')}, about {jobs.count(p['blocks'], 'block')} {p['direction']} ({state})")
    return out


def _check_out(path, place):
    folder = os.path.realpath(os.path.dirname(os.path.abspath(path)))
    out = os.path.join(folder, os.path.basename(os.path.abspath(path)))
    if not os.path.isdir(folder):
        raise Refusal(f"the folder for --out {path} does not exist", "Create it, or pass a path in an existing folder")
    if os.path.isdir(out):
        raise Refusal(f"--out {path} is a folder", "Pass a file name, for example --out area.json")
    if _within(out, place.world):
        raise Refusal(f"--out {path} is inside the world folder", "Write the area file outside the world, for example --out area.json")


def make_area(args):
    place = world.locate(args.world)
    _check_out(args.out, place)
    level, _ = world.check(place.world)
    claims = world.claims(place.world)
    _, found, warnings = _side_files(place, level)
    sc = _scan(place, args)
    table = sc.chunk_table()
    erased = set().union(*(area.rect(r) for r in args.remove))
    claimed = set().union(*claims.values()) if claims else set()
    sug = area.suggestion(table, claimed) if args.suggest or args.suggest_extra else None
    a = area.build(area.chosen_claims(claims, args.claims), _painted(args, sug), erased, args.margin)
    scene = area.scene(sc, claims, found, world.spawn(level))
    a, st, rm = area.record(a, table, scene)
    a |= {"world": place.world, "warnings": warnings}
    write_atomic(args.out, (json.dumps(a, indent=1) + "\n").encode())
    bad = area.unreadable_refusal(st)
    lines = [
        *area_lines(a, st, rm, scene),
        *suggestion_lines(sug, args.suggest_extra),
        *_warning_lines(warnings),
        *([f"Warning: {bad.reason}. Next: {bad.action}."] if bad else []),
    ]
    extras = [{k: p[k] for k in ("chunks", "blocks", "direction")} for p in sug["extra"]] if sug else []
    out = {"status": st, "removed": rm, "warnings": warnings, "parts_warning": area.parts_warning(st["parts"]), "suggested_extra": extras}
    print(json.dumps(out, indent=1) if args.json else "\n".join(lines))
    print(f"Written: {os.path.abspath(args.out)}", file=sys.stderr)
    return 0


def _names(items):
    return ", ".join(items) or "none"


def _kinds(group):
    named = list(group["named"].items())
    more = f" and {len(named) - TOP_TYPES} more kinds" if len(named) > TOP_TYPES else ""
    return (", ".join(f"{t} {_n(n)}" for t, n in named[:TOP_TYPES]) or "none") + more


def _words(group):
    return ", ".join(list(group["named"])[:TOP_WORDS]) or "blocks"


def _beside_lines(beside):
    if not beside["count"]:
        return []
    return [f"  {_n(beside['count'])} more in their chunks, of kinds villages and dungeons have too: {_kinds(beside)}"]


def _removed_lines(st, rm, scene):
    yours, beside = rm["yours"], rm["beside"]
    spawn = (
        "inside the kept area"
        if rm["spawn_in_kept"]
        else "outside the kept area: it falls in new terrain (/setworldspawn in game moves it)"
    )
    claims = _names(f"{t} {_n(n)}" for t, n in rm["other_teams_claims"].items())
    return [
        "Will be removed (outside the kept area):",
        f"  {_n(st['dropped'])} overworld chunks become new terrain",
        f"  probably yours ({area.YOURS_WHY}): {jobs.count(yours['count'], 'block')} ({_kinds(yours)})",
        *_beside_lines(beside),
        *(f"  {area.grave_text(g)}" for g in rm["graves"]),
        f"  {_n(rm['worldgen']['count'])} {_words(rm['worldgen'])}… from world generation (villages, dungeons, ruins): not counted",
        f"  {_n(rm['portal_blocks'])} nether-portal blocks",
        f"  other teams' claims: {claims}",
        f"  claimed chunks you erased: {_n(rm['erased_claims'])}",
        f"  players standing outside: {_names(rm['players'])} (Apply moves them into the kept area)",
        f"  beds outside: {_names(rm['beds'])}",
        f"  world spawn ({scene['spawn'][0]}, {scene['spawn'][1]}): {spawn}",
    ]


def area_lines(a, st, rm, scene):
    wins = "; ".join(f"x {w[0]}..{w[2] - 1}, z {w[1]}..{w[3] - 1}" for w in st["windows"])
    return [
        f"Area for {a['world']}",
        f"Protected (kept byte-identical): {jobs.count(st['core'], 'chunk')} (claims of: {_names(a['teams'])})",
        f"Kept (protected + {a['margin']}-chunk margin): {jobs.count(st['kept'], 'chunk')} in {jobs.count(st['parts'], 'part')}; "
        f"{jobs.count(st['margin_chunks'], 'margin chunk')}",
        *([f"Warning: {area.parts_warning(st['parts'])}"] if st["parts"] > 1 else []),
        f"Never explored kept chunks: {_n(len(st['never_generated']))}; "
        f"unpopulated kept chunks: {_n(len(st['unpopulated_kept']))} (protected: {_n(len(st['unpopulated_core']))}); "
        f"unreadable kept chunks: {_n(len(st['unreadable_kept']))}",
        f"Search windows: {wins}",
        *_removed_lines(st, rm, scene),
    ]


def _bytes(n):
    return f"{n / provision.MB:,.0f} MB"


def describe(e):
    """One plain line per progress event."""
    step, kind = e.get("step"), e.get("event")
    if step == "download" and kind == "start":
        how = f"resuming at {e['resume_from']:,} bytes" if e["resume_from"] else "from the start"
        return f"download: {os.path.basename(e['url'])}, {e['size']:,} bytes, {how} (HTTP {e['http']})"
    if kind == "progress":
        return f"{step}: {e['done']:,} / {e['total']:,}"
    if step == "boot" and kind == "waiting":
        return f"generator: starting ({e['seconds']} s)"
    if step == "boot" and kind == "ready":
        return f"generator: ready after {e['seconds']} s (Java {e['java']})"
    if step == "check":
        return f"check: sample {e['got']['sample']}, render {e['got']['render']} (expected {e['want']['sample']}, {e['want']['render']})"
    return f"{step}: " + ", ".join(f"{k} {v}" for k, v in e.items() if k != "step")


def _cache(args):
    return os.path.abspath(args.cache or scan.cache_root())


def gen_setup(args):
    cache, emit = _cache(args), cli_search._printer(args.json, describe)
    rel = provision.release(args.release)
    if not args.accept_eula:
        generator.require_eula(provision.pack_dir(cache, rel.name))
    provision.claim_cache(cache)
    with generator.lock(cache):
        provision.install(cache, rel, args.zip and world.normalise(args.zip), emit)
        want = provision.JavaWanted(args.java and world.normalise(args.java), not args.no_java_search)
        java = provision.choose_java(cache, rel.entry["java"], want, emit)
        result = generator.boot_and_check(generator.Context(cache, rel.name, rel.entry, java, args.accept_eula, emit), args.heap)
        generator.save_settings(cache, {"release": rel.name, "java": java.path, "heap": args.heap})
    print(json.dumps(result, indent=1) if args.json else f"Generator ready: GTNH {rel.name} in {cache}, Java {java.version}; check passed")
    return 0


def _settings(cache):
    s = generator.settings(cache)
    if not s:
        raise Refusal(f"the generator is not set up in {cache}", "Run: rwg2-migrator generator setup --accept-eula")
    return s


def gen_check(args):
    cache, emit = _cache(args), cli_search._printer(args.json, describe)
    s = _settings(cache)
    rel = provision.release(s["release"])
    with generator.lock(cache):
        java = provision.choose_java(cache, rel.entry["java"], provision.JavaWanted(s["java"]), emit)
        result = generator.boot_and_check(generator.Context(cache, rel.name, rel.entry, java, args.accept_eula, emit), s["heap"])
    print(
        json.dumps(result, indent=1)
        if args.json
        else f"Check passed: sample {result['check']['sample']}, render {result['check']['render']}"
    )
    return 0


def gen_status(args):
    cache = _cache(args)
    packs = os.path.join(cache, "packs")
    installed = sorted(d for d in os.listdir(packs) if not d.endswith(provision.PARTIAL)) if os.path.isdir(packs) else []
    out = {"cache": cache, "settings": generator.settings(cache), "installed": installed, "running": None}
    gen = generator.running(cache)
    if gen:
        out["running"] = gen.record
        try:
            out["status"] = gen.connect(generator.STATUS_TIMEOUT).call("status")
        except (OSError, generator.GeneratorError) as e:
            out["status"] = {"error": f"no answer ({e}); it may be busy with another request"}
    if args.json:
        print(json.dumps(out, indent=1))
        return 0
    print(f"Cache: {cache}; installed: {', '.join(installed) or 'nothing'}")
    run = out["running"]
    print(f"Running: {run['kind']} generator, pid {run['pid']}, port {run['port']}" if run else "Running: no generator")
    return 0


def gen_stop(args):
    was, code = generator.stop_running(_cache(args), cli_search._printer(args.json, describe))
    if args.json:
        print(json.dumps({"was_running": was, "exit_code": code}))
    else:
        print(("Stopped" + (f" (exit code {code})" if code is not None else "")) if was else "No generator was running")
    return 0


def _remove_generator(cache):
    removed = {}
    with generator.lock(cache):
        generator.stop_running(cache)
        for part in ("packs", "downloads", "java"):
            path = os.path.join(cache, part)
            if os.path.isdir(path):
                removed[part] = sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file())
                rmtree(path)
        for name in (generator.SETTINGS_FILE, generator.RECORD_FILE):
            remove_file(os.path.join(cache, name))
    return removed


def clean(args):
    if not args.generator:
        raise Refusal("nothing to clean was named", "Pass --generator to remove the downloaded generator")
    cache = _cache(args)
    if os.path.isdir(cache) and not provision.own_cache(cache):
        raise Refusal(
            f"{cache} is not a cache folder of rwg2-migrator (it has no {provision.CACHE_MARKER}), so nothing was removed",
            "Pass --cache the folder the tool used, or leave --cache out",
        )
    removed = _remove_generator(cache) if os.path.isdir(cache) else {}
    print(json.dumps(removed) if args.json else "Removed: " + (", ".join(f"{k} ({_bytes(v)})" for k, v in removed.items()) or "nothing"))
    return 0


def _generator_parser(sub, common):
    g = sub.add_parser("generator", help="the RWG2 generator: the official GTNH server pack run headless on this machine")
    gs = g.add_subparsers(dest="action", required=True)
    s = gs.add_parser("setup", parents=[common], help="download, unpack and check the generator")
    s.add_argument("--release", help="GTNH release (default: the newest this tool knows)")
    s.add_argument("--zip", help="use this GTNH server zip instead of downloading it")
    s.add_argument("--java", help="the java executable to use (64-bit, in the release's Java range)")
    s.add_argument("--no-java-search", action="store_true", help="ignore installed Java; use (or download) the tool's own")
    s.add_argument("--heap", default=generator.DEFAULT_HEAP, help="generator heap (default 3G)")
    s.add_argument("--accept-eula", action="store_true", help="I accept the Minecraft EULA (https://aka.ms/MinecraftEULA)")
    s.set_defaults(fn=gen_setup)
    c = gs.add_parser("check", parents=[common], help="boot the generator and compare its output with the pinned digests")
    c.add_argument("--accept-eula", action="store_true", help="I accept the Minecraft EULA (https://aka.ms/MinecraftEULA)")
    c.set_defaults(fn=gen_check)
    gs.add_parser("status", parents=[common], help="what is installed and whether a generator runs").set_defaults(fn=gen_status)
    gs.add_parser("stop", parents=[common], help="stop the tool's generator").set_defaults(fn=gen_stop)
    k = sub.add_parser("clean", parents=[common], help="remove downloaded data")
    k.add_argument("--generator", action="store_true", help="remove the server pack, the downloads and the tool's own Java")
    k.set_defaults(fn=clean)


def _common():
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--json", action="store_true", help="print machine-readable JSON")
    common.add_argument("--cache", help="cache folder (default: the per-user cache folder)")
    common.add_argument("--workers", type=int, help="processes for the world scan (default: cores - 2, at most 6)")
    return common


def _inspect_parser(sub, common):
    i = sub.add_parser("inspect", parents=[common], help="check a world and scan its overworld (read-only)")
    i.add_argument("world", nargs="?", help="world, save or server folder, or level.dat; leave out to list single-player saves")
    i.add_argument("--config-dir", help=cli_search.CONFIG_HELP)
    i.add_argument("--remote", action="store_true", help="this world was copied from another machine")
    i.add_argument("--require-stopped", action="store_true", help="exit non-zero unless the world is stopped (the check Apply runs)")
    i.add_argument("--confirm-stopped", action="store_true", help="with --require-stopped: I checked that the server is stopped")
    i.set_defaults(fn=inspect)


def _area_parser(sub, common):
    a = sub.add_parser("area", parents=[common], help="choose the area to keep and see what the rest removes (read-only)")
    a.add_argument("world")
    a.add_argument("--claims", metavar="all|TEAM,...", help="keep the ServerUtilities claims of these teams")
    a.add_argument(
        "--add", action="append", default=[], metavar="X1,Z1:X2,Z2", help="keep a rectangle of chunks (chunk coordinates, may be negative)"
    )
    a.add_argument("--remove", action="append", default=[], metavar="X1,Z1:X2,Z2", help="do not keep a rectangle of chunks")
    a.add_argument("--chunks", metavar="FILE", help="keep the chunks listed in FILE (X,Z per line, or JSON)")
    a.add_argument(
        "--suggest", action="store_true", help="also keep the part of the world, around your claims, where players spent at least an hour"
    )
    a.add_argument(
        "--suggest-extra", action="append", type=int, default=[], metavar="N", help="also keep suggested extra part N (listed by area)"
    )
    a.add_argument("--margin", type=int, default=area.DEFAULT_MARGIN, help="safety band around the core, in chunks (at least 1)")
    a.add_argument("--out", required=True, metavar="AREA.json")
    a.set_defaults(fn=make_area)


def _ui(args):
    from .web import server

    return server.serve(args)


def _ui_parser(sub, common):
    u = sub.add_parser("ui", parents=[common], help="open the app in your browser (the default with no command)")
    u.add_argument("--port", type=int, help="port on 127.0.0.1 (default: a free one)")
    u.add_argument("--no-browser", action="store_true", help="only print the address (for a headless machine: ssh -L PORT:127.0.0.1:PORT)")
    u.add_argument("--data", help="jobs folder (default: the per-user data folder)")
    u.add_argument("--replay", metavar="JOB", help=argparse.SUPPRESS)
    u.add_argument("--speed", type=float, help=argparse.SUPPRESS)
    u.set_defaults(fn=_ui)


def parser():
    ap = argparse.ArgumentParser(
        prog="rwg2-migrator",
        description="Move a GTNH legacy-RWG world to RWG2 Smooth, keeping the area you choose. One rule: close the game or stop "
        "the server before you start, and keep it closed until Apply is done.",
    )
    ap.add_argument("--version", action="version", version=f"rwg2-migrator {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)
    common = _common()
    _inspect_parser(sub, common)
    _area_parser(sub, common)
    _generator_parser(sub, common)
    for name, fn in cli_search.add_parsers(sub, common).items():
        sub.choices[name].set_defaults(fn=fn if name == "candidates" else _with_scan(fn))
    sub.choices["jobs"].set_defaults(fn=lambda args: cli_search.jobs_cmd(args, _scan, cli_apply.RESUME))
    for name, fn in cli_apply.add_parsers(sub, common).items():
        sub.choices[name].set_defaults(fn=_with_scan(fn))
    _ui_parser(sub, common)
    return ap


def _with_scan(fn):
    return lambda args: fn(args, _scan)


def joined(argv):
    """argparse reads a value such as -5,-3:-1,-1 after a space as an option, so it is joined to its flag with '='."""
    out = []
    for a in argv:
        if out and out[-1] in RECT_FLAGS and a[:1] == "-" and a[1:2].isdigit():
            out[-1] += "=" + a
        else:
            out.append(a)
    return out


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    args = parser().parse_args(joined(argv or ["ui"]))
    try:
        return args.fn(args)
    except Refusal as e:
        print(f"Refused: {e.reason}. Next: {e.action.rstrip('.')}.", file=sys.stderr)
        return EXIT_REFUSED
    except generator.GeneratorError as e:
        print(f"Failed: the generator answered {e}. Next: see the log under {os.path.join(_cache(args), 'packs')}.", file=sys.stderr)
        return EXIT_REFUSED
    except Exception as e:
        traceback.print_exc()
        print(f"Failed: an unexpected error ({type(e).__name__}: {e}). Next: report it at {ISSUES} with these details.", file=sys.stderr)
        return EXIT_REFUSED


if __name__ == "__main__":
    sys.exit(main())
