"""The CLI commands of the search: search, candidates, map and jobs."""

import dataclasses
import json
import os
import shutil
import sys
from contextlib import nullcontext

from . import Refusal, appearance, apply, area, generator, jobs, maps, provision, scan, search, world
from .region import write_atomic

SPARE_CORES = 2
CONFIG_HELP = "the config folder of the server or game instance (default: next to the world; none: GTNH's default RWG2 settings)"
AREA_FILE = "area.json"
EVENTS_SHOWN = 8
PCT = 100
CHANGED = {
    "site": "the ground around the area to keep",
    "reference": "the look of your base",
    "lmap": "the generator's inputs (seed, GTNH release, your config folder)",
    "prefs": "your preferences",
    "plan": "the search settings",
    "bans": "the looks to avoid",
    "version": "the tool version",
}


def _data(args):
    return os.path.abspath(args.data or jobs.data_root())


def _cache(args):
    return os.path.abspath(args.cache or scan.cache_root())


def _threads(args):
    return args.threads or max(1, (os.cpu_count() or 1) - SPARE_CORES)


def read_area(path, place):
    try:
        with open(path, encoding="utf-8") as f:
            a = json.load(f)
        a["core"], a["kept"] = [tuple(c) for c in a["core"]], [tuple(c) for c in a["kept"]]
    except (OSError, ValueError, KeyError, TypeError) as e:
        raise Refusal(
            f"the area file {path} cannot be read ({e})", "Make it with: rwg2-migrator area WORLD --claims all --out area.json"
        ) from e
    if os.path.normcase(os.path.realpath(a.get("world", place.world))) != os.path.normcase(os.path.realpath(place.world)):
        raise Refusal(
            f"the area file {path} was made for {a['world']}, not {place.world}", "Make an area for this world with the area command"
        )
    return a


class Generators:
    """Boots the generator on the user's config folder on first use (provisioning and checking it with --accept-eula), one boot
    at a time, and stops it at the end."""

    def __init__(self, cache, args, emit, overlay=None):
        self.cache, self.args, self.emit, self.gen = cache, args, emit, None
        self.overlay = overlay if overlay and os.path.isdir(overlay) else None
        self.config = jobs.tree_key(self.overlay)
        self.threads = _threads(args) if hasattr(args, "threads") else None

    def context(self):
        s = generator.settings(self.cache)
        accept = getattr(self.args, "accept_eula", False)
        if not s:
            if not accept:
                raise Refusal(
                    f"the generator is not set up in {self.cache}",
                    "Run this again with --accept-eula, or run: rwg2-migrator generator setup",
                )
            rel = provision.release()
            provision.claim_cache(self.cache)
            provision.install(self.cache, rel, None, self.emit)
            java = provision.choose_java(self.cache, rel.entry["java"], provision.SEARCH, self.emit)
            s = {"release": rel.name, "java": java.path, "heap": generator.DEFAULT_HEAP}
        rel = provision.release(s["release"])
        java = provision.choose_java(self.cache, rel.entry["java"], provision.JavaWanted(s["java"]), self.emit)
        return generator.Context(self.cache, rel.name, rel.entry, java, accept, self.emit), s

    def start(self, boot):
        """Boot one generator of this kind on the user's config, after the release check on first use."""
        ctx, s = self.context()
        if not generator.settings(self.cache):
            generator.boot_and_check(ctx, s["heap"])
            generator.save_settings(self.cache, s)
        self.stop()
        self.gen = generator.start(ctx, dataclasses.replace(boot, heap=s["heap"], overlay=self.overlay))
        self.gen.start_heartbeat()
        return self.gen

    def sampler(self, seed, threads):
        return lambda: search.ProbeSampler(self.start(generator.Boot()), seed, threads)

    def stop(self):
        """Stop the generator this run started; returns its exit code (None when none was running)."""
        if self.gen is None:
            return None
        code = self.gen.stop(emit=self.emit)
        self.gen = None
        return code


def describe(e):
    """One line per search event worth showing on a console."""
    kind = e.get("event")
    if kind == "stage":
        return f"search: {search.stage_label(e['stage'])[0]}: {e['configs']:,} settings in {e['seconds']} s"
    if kind == "excellent":
        stop = f": rwg2-migrator jobs cancel {e['job']}" if e.get("job") else ""
        return f"search: excellent match found after {e['seconds']} s: you can stop the search now{stop}"
    if kind == "reset":
        what = "; ".join(CHANGED.get(k, k) for k in e["changed"])
        return f"search: changed since this job last sampled: {what}. Its cached samples were dropped; the search starts again"
    if kind == "rendered":
        return f"search: drew the land map around your base in {e['seconds']} s"
    if kind in ("start", "done", "cancelled", "interrupted", "failed", "prefilter"):
        return f"{e.get('step')}: {kind}" + (f" ({e['error']})" if "error" in e else "")
    return None


def _printer(as_json, lines=describe):
    def show(e):
        line = json.dumps(e) if as_json else lines(e)
        if line:
            print(line, file=sys.stdout if as_json else sys.stderr, flush=True)

    return show


def _inputs(job, sc, gens):
    d = job.data
    a = search.load(os.path.join(job.folder, AREA_FILE))
    a["core"], a["kept"] = [tuple(c) for c in a["core"]], [tuple(c) for c in a["kept"]]
    level = world.read_level(d["world"])
    rel, _ = world.release_for(world.mods(level), world.releases())
    seed = level["Data"]["RandomSeed"]
    return search.Inputs(sc, a, d["args"]["prefs"], rel, seed, gens.sampler(seed, d["args"]["threads"]), config=gens.config)


def defaults_note(config):
    """The console line of a step that runs on GTNH's default RWG2 settings because no config folder is in use."""
    if not (config and os.path.isdir(config)):
        print(f"Note: {world.NO_CONFIG}. Pass --config-dir to use the config folder of the server or game instance", file=sys.stderr)


def _run(job, args, sc, place):
    cache = job.data["args"]["cache"]
    show = _printer(args.json)
    job.listeners.append(show)
    config = job.data["args"].get("config_dir", place.config)
    defaults_note(config)
    gens = Generators(cache, args, show, config)
    try:
        with generator.lock(cache):
            result = jobs.run(job, lambda j: search.run_job(j, _inputs(j, sc, gens)))
    finally:
        gens.stop()
    print(json.dumps(result, indent=1, default=float) if args.json else "\n".join(card_lines(result, all_=False)))
    return 0


def searched_place(args, job):
    """The world's place on the config folder named, else on the one the search job ran on while it still holds rwg2.cfg."""
    ran_on = job and job.data["args"].get("config_dir")
    kept = ran_on if ran_on and world.holds_rwg2(ran_on) else None
    return world.with_config(world.locate(args.world), args.config_dir or kept)


def search_cmd(args, scan_world):
    place = world.with_config(world.locate(args.world), args.config_dir)
    a = read_area(args.area, place)
    prefs = appearance.load_prefs(args.prefs)
    sc = scan_world(place, args)
    bad = area.unreadable_refusal(area.status(a, sc.chunk_table()))
    if bad:
        raise bad
    cache = _cache(args)
    job_args = {"prefs": prefs, "threads": _threads(args), "cache": cache, "config_dir": place.config}
    job = jobs.create(_data(args), "search", job_args, place.world)
    write_atomic(os.path.join(job.folder, AREA_FILE), (json.dumps(a) + "\n").encode())
    print(f"Job {job.id} ({job.folder})", file=sys.stderr, flush=True)
    return _run(job, args, sc, place)


def resume(args, scan_world, others=None):
    """Resume a job; jobs of the kinds in others {kind: fn(args, job, scan_world)} are resumed by their own command."""
    job = jobs.find(_data(args), args.id)
    jobs.refresh(job.root)
    if job.state() not in jobs.RESUMABLE:
        raise Refusal(f"job {job.id} is {job.state()}", "Only an interrupted, failed or cancelled job can be resumed")
    if job.data.get("kind") in (others or {}):
        return others[job.data["kind"]](args, job, scan_world)
    place = world.locate(job.data["world"])
    args.cache = job.data["args"]["cache"]
    return _run(job, args, scan_world(place, args), place)


def card_lines(result, all_):
    ref = result["reference"]
    lines = [f"Your base looks like: {ref['name']} (avoiding: {', '.join(result['bans']) or 'nothing'})"]
    s = result["search"]
    lines.append(f"Search: {s['configs']} configs in {s['seconds']} s" + (" (stopped early: best so far)" if s["cancelled"] else ""))
    lines += [line for i, c in enumerate(result["candidates"], 1) for line in candidate_lines(c, f"#{i}", all_)]
    first = len(result["candidates"]) + 1
    for i, c in enumerate(result["alternatives"] if all_ else [], first):
        lines += candidate_lines(c, f"Alternative {i - first + 1} (--pick {i})", all_)
    if not result["candidates"]:
        lines.append(
            "No match is on dry land (no ocean nearby, little of your land below sea level). Try a larger area or more tolerant settings."
        )
    return lines


def candidate_lines(c, title, details):
    card = c["card"]
    out = ["", f"{title}  " + "  ".join(f"[{b}]" for b in card["badges"])]
    out += [f"  {line}" for line in card["lines"]]
    if details:
        bx, bz, lx, lz = c["config"]
        out.append(
            f"  Details: seam median {c['median']:g}, p90 {c['p90']:g}, {PCT * c['over10']:.2f}% over 10, "
            f"{PCT * c['over30']:.2f}% over 30; total {c['total']:.3f} (seam {c['seam']:.3f} + look {c['penalty']:.3f}); "
            f"edge {_mix(c['edge_biomes'])}"
        )
        out.append(f"  Offsets: biome {bx},{bz}  landmass {lx},{lz}")
    return out


def _mix(m):
    return ", ".join(f"{k} {v:g}%" for k, v in m.items())


def candidates_cmd(args):
    job = jobs.find(_data(args), args.job)
    path = os.path.join(job.folder, "candidates", "candidates.json")
    if not os.path.isfile(path):
        raise search.no_candidates(job)
    result = search.load(path)
    print(json.dumps(result, indent=1) if args.json else "\n".join(card_lines(result, args.all)))
    return 0


def jobs_cmd(args, scan_world, others=None):
    data = _data(args)
    jobs.refresh(data)
    action = args.action or "list"
    if action == "list":
        listing = [j.data for j in jobs.all_jobs(data)]
        print(
            json.dumps(listing, indent=1)
            if args.json
            else "\n".join(f"{d['id']}  {d['kind']}  {d['state']}  {d['world']}" for d in listing)
        )
        return 0
    if action == "resume":
        return resume(args, scan_world, others)
    job = jobs.find(data, args.id)
    if action == "show":
        lines = [line for line in map(describe, job.events()) if line]
        print(json.dumps(job.data, indent=1) + ("" if args.json else "\n" + "\n".join(lines[-EVENTS_SHOWN:])))
    elif action == "cancel":
        job.request_cancel()
        print(f"Asked job {job.id} to stop after its current batch (a search keeps its best so far)")
    else:
        if job.state() == jobs.RUNNING:
            raise Refusal(f"job {job.id} is running", f"Cancel it first: rwg2-migrator jobs cancel {job.id}")
        busy = apply.needed_by(data, job)
        if busy:
            raise Refusal(
                f"apply job {busy.id} may have changed your config or moved your world aside, and only that Apply can undo it",
                f"Resume it: rwg2-migrator jobs resume {busy.id}. It either finishes Apply or puts your config back; "
                "then the job can be discarded",
            )
        apply.drop_working_copy(job)
        shutil.rmtree(job.folder)
        print(f"Discarded job {job.id}")
    return 0


def map_cmd(args, scan_world):
    job = jobs.find(_data(args), args.search) if args.search else None
    place = searched_place(args, job)
    a = read_area(args.area, place)
    sc = scan_world(place, args)
    level = world.read_level(place.world)
    rel, _ = world.release_for(world.mods(level), world.releases())
    pick = maps.pick(job, args.pick) if job else None
    wins = [maps.window_arg(args.window)] if args.window else [tuple(w) for w in area.windows(set(a["kept"]))]
    cache = job.data["args"]["cache"] if pick else _cache(args)
    defaults_note(place.config)
    gens = Generators(cache, args, _printer(args.json), place.config)
    if pick:
        pick["generator"] = [level["Data"]["RandomSeed"], gens.config]
    changed = pick and jobs.config_change(place.config, search.sampled_config(job), f"search job {job.id}")
    if changed:
        raise Refusal(
            f"{changed}, so its candidates may no longer be what RWG2 makes",
            f"Run the search again: {apply.search_again(place, args.area)}",
        )
    try:
        with generator.lock(cache) if pick and not maps.cached(pick, wins, args.bpp) else nullcontext():
            renderer = maps.Renderer(pick, gens.sampler(level["Data"]["RandomSeed"], _threads(args))) if pick else None
            out = maps.draw(maps.Scene(sc, a, rel, pick, wins, args.bpp), renderer, args.out)
    finally:
        gens.stop()
    print(json.dumps(out) if args.json else f"Written: {os.path.abspath(args.out)} ({jobs.count(len(wins), 'window')})")
    return 0


def add_parsers(sub, common):
    s = sub.add_parser("search", parents=[common], help="find the RWG2 offsets whose new terrain fits the kept area best")
    s.add_argument("world")
    s.add_argument("--area", required=True, metavar="AREA.json")
    s.add_argument("--prefs", metavar="PREFS.json", help='{"base": biome name, "banned": [looks], "tolerance": 1.0}')
    s.add_argument("--config-dir", help=CONFIG_HELP)
    s.add_argument("--threads", type=int, help="generator threads (default: cores - 2)")
    s.add_argument(
        "--accept-eula", action="store_true", help="I accept the Minecraft EULA (https://aka.ms/MinecraftEULA); sets up the generator"
    )
    s.add_argument("--data", help="jobs folder (default: the per-user data folder)")
    c = sub.add_parser("candidates", parents=[common], help="the best matches of a search job, in plain words")
    c.add_argument("job", help="job id, a unique prefix of it, or latest")
    c.add_argument("--all", action="store_true", help="also the alternatives, with the numbers behind each line")
    c.add_argument("--data")
    m = sub.add_parser("map", parents=[common], help="draw the world (and a candidate) as a PNG")
    m.add_argument("world")
    m.add_argument("--area", required=True, metavar="AREA.json")
    m.add_argument("--search", metavar="JOB", help="a search job id, or latest")
    m.add_argument("--pick", type=int, default=1, help="candidate rank (default 1)")
    m.add_argument("--window", metavar="X0,Z0,SIZE", help="one window in blocks instead of one per kept part")
    m.add_argument("--bpp", type=int, help="blocks per pixel (1, 2, 4, 8 or 16; default: fits 1024 pixels)")
    m.add_argument("--out", required=True, metavar="PNG")
    m.add_argument("--config-dir", help=CONFIG_HELP)
    m.add_argument("--threads", type=int)
    m.add_argument("--accept-eula", action="store_true")
    m.add_argument("--data")
    j = sub.add_parser("jobs", parents=[common], help="list, show, resume, cancel or discard jobs")
    j.add_argument("action", nargs="?", choices=["list", "show", "resume", "cancel", "discard"])
    j.add_argument("id", nargs="?")
    j.add_argument("--accept-eula", action="store_true")
    j.add_argument("--data")
    return {"search": search_cmd, "candidates": candidates_cmd, "map": map_cmd, "jobs": jobs_cmd}
