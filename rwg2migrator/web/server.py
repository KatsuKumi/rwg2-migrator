"""The local web UI: a stdlib HTTP server on 127.0.0.1 with a session token and a Host check. Long steps run as CLI subprocesses
(the same commands, the same jobs); the browser polls their events.jsonl, pulls map tiles cached on disk and hears of each tile
block drawn over one SSE stream."""

import bisect
import hmac
import itertools
import json
import logging
import math
import mimetypes
import os
import platform
import secrets
import shutil
import string
import subprocess
import sys
import threading
import time
import webbrowser
from collections import deque
from contextlib import ExitStack
from dataclasses import dataclass, field, replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from types import SimpleNamespace
from urllib.parse import parse_qs, unquote, urlsplit

import numpy as np
import psutil

from .. import (
    ISSUES,
    Refusal,
    __version__,
    appearance,
    apply,
    area,
    cli,
    cli_search,
    generator,
    genworld,
    jobs,
    maps,
    plant,
    procs,
    scan,
    score,
    search,
    smooth,
    staging,
    world,
)
from ..region import write_atomic
from . import tiles

LOCALHOST = "127.0.0.1"
TOKEN_BYTES = 24
UI_FILE = "ui.json"
PRIVATE_FILE = 0o600
OWNER_ONLY = 0o077
SETTINGS_FILE = "ui-settings.json"
RECENT = 8
HEAT_WAIT = 3.0
PICK_AGAIN = "Click Pick this one again on Best matches"
IN_USE_SECONDS = 3.0
DRAWN_KEPT = 4096
ASK_EXPIRE = 6.0
BOOT_RETRY = 10.0
HEARTBEAT = 15.0
LOG_TAIL = 40
TILE_PATH_PARTS = 6
LIVE_LINES = 60
MB = 1 << 20
TILE_CACHE = "public, max-age=31536000, immutable"
NO_STORE = (("Cache-Control", "no-store"),)
CLI_ENV = "RWG2_CLI"
KEEP_OPEN = "Keep this window open. Closing it pauses the job; you can continue later."
NEXT = ". Next: "
ERROR_HEADS = ("Refused: ", "Failed: ")
REPORT = f"Click 'Copy diagnostic info' and report it at {ISSUES}"
ROLLING_BACK = "rolling back"
CACHE_ONLY = ("generator", "clean")
FILLS_IN = "This map fills in once it is done"
MINECRAFT = ".minecraft"
REPLAY_SPEED = 60.0
REPLAY_GAP = 5.0
REPLAY_ONLY = "this replay shows only the maps its recorded run drew, and no generator can start here"
STAGES, stage_label = search.STAGES, search.stage_label
PREVIEW_STEPS = (
    (staging.COPY, "Copy your world"),
    (staging.SNAPSHOT, "Note what must stay identical"),
    (staging.TRANSPLANT, "Keep only the area you chose"),
    (genworld.GENERATE, "Generate the new terrain around it"),
    (genworld.RENDER, "Draw the chosen match"),
    (genworld.HARVEST, "Put the new terrain into the working copy"),
    (smooth.SMOOTH, "Smooth the border"),
    (plant.PLANT, "Grow trees on the reshaped border"),
    (plant.TREES, "Add those trees"),
    (staging.FINAL_STEP, "List every file of the result"),
    (staging.POSTCHECK, "Check the result"),
)
AFTER_LAYERS = {"planned": "PREVIEW", "before": "BEFORE APPLY", "applied": "NOW"}
GENERATED_STEPS = {genworld.GENERATE, genworld.RENDER, genworld.HARVEST, smooth.SMOOTH, plant.PLANT, plant.TREES, staging.POSTCHECK}
log = logging.getLogger("rwg2migrator.ui")


def labelled(e):
    """An event as the browser gets it: a search stage carries a plain label, which is all the page shows of it."""
    if not isinstance(e.get("stage"), str):
        return e
    label, share = stage_label(e["stage"])
    return e | {"label": label, "progress": share}


def preview_steps(job, cutoff=None):
    """Each step of the preview pipeline with a plain label, whether it applies to this preview and whether it is done (by the
    time cutoff, when given)."""
    a = job.data.get("args", {})
    skip = set()
    if a.get("no_generate"):
        skip |= GENERATED_STEPS
    if not a.get("smooth", True):
        skip |= {smooth.SMOOTH, plant.PLANT, plant.TREES}
    if not a.get("trees"):
        skip |= {plant.PLANT, plant.TREES}
    manifests = {name: jobs.read_json(os.path.join(job.folder, name, jobs.MANIFEST_FILE)) or {} for name, _ in PREVIEW_STEPS}
    finished = {name: m.get("finished") for name, m in manifests.items()}
    done = {name for name, t in finished.items() if t is not None and (cutoff is None or t <= cutoff)}
    return [{"label": label, "done": name in done} for name, label in PREVIEW_STEPS if name not in skip]


def static(name):
    return resources.files("rwg2migrator").joinpath("web", "static", *name.split("/"))


def cli_prefix():
    """The command that runs this tool's CLI: the frozen app itself, or this Python with the package."""
    if os.environ.get(CLI_ENV):
        return json.loads(os.environ[CLI_ENV])
    return [sys.executable] if getattr(sys, "frozen", False) else [sys.executable, "-m", "rwg2migrator"]


def refusal(e):
    return {"reason": e.reason, "action": e.action}


def refusal_of(problem):
    return problem and {"reason": problem["reason"], "action": problem["action"]}


def _split(message):
    head = next(h for h in ERROR_HEADS if message.startswith(h))
    reason, _, action = message[len(head) :].rpartition(NEXT)
    return {"reason": reason, "action": action.rstrip(".")}


def parse_error(text, log_path):
    """The refusal (or failure) a CLI subprocess ended with, as reason and next action. A reason may span lines (a generator log
    tail): then the message runs from its first line to the end of the text, whose last line holds the next action."""
    lines = [line for line in text.splitlines() if line.strip()]
    first = next((i for i, line in enumerate(lines) if line.startswith(ERROR_HEADS)), None)
    if first is not None and NEXT in lines[-1]:
        return _split(lines[-1] if lines[-1].startswith(ERROR_HEADS) else "\n".join(lines[first:]))
    one = next((line for line in reversed(lines) if line.startswith(ERROR_HEADS) and NEXT in line), None)
    if one:
        return _split(one)
    last = lines[-1] if lines else "the step stopped without a message"
    return {
        "reason": f"the step stopped unexpectedly ({last})",
        "action": f"{REPORT}; the full log is {log_path}",
    }


def tail(path, n=LOG_TAIL):
    """The last n lines of a text file (all of them with n None); none when it cannot be read."""
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            lines = f.read().splitlines()
    except OSError:
        return []
    return lines if n is None else lines[-n:]


class Runner:
    """At most one CLI subprocess at a time, its output in log files. It shares the console, so closing the console stops it
    too and its job becomes interrupted, to be continued from the home screen."""

    def __init__(self, ui):
        self.ui, self.proc, self.info, self.starting = ui, None, None, False
        self._lock, self.count = threading.Lock(), itertools.count(1)

    def busy(self):
        """A step runs, or is being started: the map's generator must not be booted again between its release and the step."""
        return self.starting or (self.proc is not None and self.proc.poll() is None)

    def start(self, kind, argv, job=None):
        """job: the id of the job the step continues, when the step does not print it (jobs resume)."""
        with self._lock:
            if self.busy():
                raise Refusal("another step is still running", "Wait for it, or stop it, then try again")
            self.starting = True
            try:
                return self._start(kind, argv, job)
            finally:
                self.starting = False

    def _start(self, kind, argv, job):
        self.ui.maps.source.release()
        logs = os.path.join(self.ui.data, "logs")
        os.makedirs(logs, exist_ok=True)
        base = os.path.join(logs, f"ui-{time.strftime('%Y%m%d-%H%M%S')}-{next(self.count)}-{kind}")
        data = [] if argv[0] in CACHE_ONLY else ["--data", self.ui.data]
        cmd = [*cli_prefix(), *argv, *data, "--cache", self.ui.cache]
        env = os.environ | {"PYTHONIOENCODING": "utf-8"}
        with open(base + ".out", "wb") as out, open(base + ".err", "wb") as err:
            self.proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=out, stderr=err, env=env)
        self.info = {"kind": kind, "argv": argv, "job": job, "started": time.time(), "out": base + ".out", "err": base + ".err"}
        log.info("started %s: %s", kind, cmd)
        return self.state()

    def stop(self):
        """Stop the running step on purpose: its job stays resumable, and the page shows no error for it. A step still being
        started is waited for, so the step stopped is the new one."""
        with self._lock:
            if self.proc is not None and self.proc.poll() is None:
                self.info["stopped"] = True
                self.proc.terminate()

    def cutoff(self, job_id):
        """The time of the last event of job_id the page may see; None for all of them (only a replay holds events back)."""
        return None

    def state(self):
        if not self.info:
            return None
        code = self.proc.poll()
        err = tail(self.info["err"], LIVE_LINES)
        st = {k: self.info[k] for k in ("kind", "started")} | {"running": code is None, "code": code}
        st["job"] = self.info["job"] or _job_id(err)
        st["live"] = _live(self.info["out"], err)
        if code is None:
            return st
        st["result"] = self._result()
        if self.info.get("stopped"):
            st["stopped"] = True
        elif code != 0 and st["result"] is None:
            st["error"] = parse_error("\n".join(tail(self.info["err"], None)), self.info["err"])
        return st

    def _result(self):
        """The JSON document a finished step printed (verify prints one even when it found problems), else None."""
        try:
            with open(self.info["out"], encoding="utf-8") as f:
                return json.load(f)
        except (OSError, ValueError):
            return None


class Replay(Runner):
    """ui --replay: a search or preview started on the page is not run. The recorded job plays back instead: its events are
    released speed times faster, with pauses between its runs cut to REPLAY_GAP, and it stays hidden until then. Every other step
    (Apply among them) runs as usual."""

    def __init__(self, ui, search_id, speed=REPLAY_SPEED):
        super().__init__(ui)
        searched = jobs.find(ui.data, search_id)
        if searched.data.get("kind") != "search" or searched.state() != jobs.DONE:
            raise Refusal(f"job {searched.id} is not a finished search", "Replay a finished search job: rwg2-migrator jobs list")
        previews = [j for j in jobs.all_jobs(ui.data) if j.data.get("kind") == "preview" and j.data["args"].get("search") == searched.id]
        ready = [j for j in previews if j.state() == jobs.DONE]
        self.recorded = {"search": searched.id} | ({"preview": ready[-1].id} if ready else {})
        self.speed, self.played, self.playing = speed, {}, None

    def start(self, kind, argv, job=None):
        if kind not in self.recorded:
            self.playing = None
            return super().start(kind, argv, job)
        recorded = jobs.Job(self.ui.data, self.recorded[kind])
        pick = argv[argv.index("--pick") + 1] if "--pick" in argv else None
        if pick is not None and str(recorded.data["args"]["pick"]) != pick:
            raise Refusal(f"this replay holds the preview of match #{recorded.data['args']['pick']}", "Pick that match")
        times = [e["t"] for e in recorded.events() if "t" in e]
        play = [0.0]
        for a, b in zip(times, times[1:], strict=False):
            play.append(play[-1] + min(b - a, REPLAY_GAP))
        self.playing = {"kind": kind, "job": recorded.id, "started": time.time(), "times": times, "play": play}
        self.played[recorded.id] = self.playing
        return self.state()

    def cutoff(self, job_id):
        p = self.played.get(job_id)
        if p is None:
            return -math.inf if job_id in self.recorded.values() else None
        i = bisect.bisect_right(p["play"], (time.time() - p["started"]) * self.speed)
        return None if i >= len(p["times"]) else p["times"][i - 1]

    def state(self):
        p = self.playing
        if p is None:
            return super().state()
        cut = self.cutoff(p["job"])
        shown = [e for e in jobs.Job(self.ui.data, p["job"]).events() if cut is None or e.get("t", 0) <= cut]
        return {
            "kind": p["kind"],
            "started": p["started"],
            "running": cut is not None,
            "code": None if cut is not None else 0,
            "job": p["job"],
            "live": next((e for e in reversed(shown) if e.get("step")), None),
            "result": None,
        }


def _job_id(err_lines):
    for line in err_lines:
        if line.startswith("Job ") and " (" in line:
            return line.split()[1]
    return None


def _live(out_path, err_lines):
    """The last progress line worth showing while a step runs: a JSON event of the generator, or a console line."""
    for line in reversed(tail(out_path, LIVE_LINES)):
        try:
            e = json.loads(line)
        except ValueError:
            continue
        if isinstance(e, dict) and e.get("step"):
            return e
    return {"text": err_lines[-1]} if err_lines and not err_lines[-1].startswith((" ", "Traceback")) else None


class Asks:
    """One worker for generator work (candidate tiles, seam heat): the most recently asked first, and work nobody asked for in
    ASK_EXPIRE seconds (a tile scrolled out of view) is dropped."""

    def __init__(self):
        self.items, self.cv = {}, threading.Condition()
        threading.Thread(target=self._run, daemon=True, name="ui-generator-work").start()

    def ask(self, key, fn, wait=0.0):
        """(done, result or error) after waiting at most wait seconds; an error is handed out once, and the next ask tries again."""
        with self.cv:
            item = self.items.get(key)
            if item is None:
                item = self.items[key] = {"fn": fn, "done": threading.Event()}
            item["asked"] = time.monotonic()
            self.cv.notify()
        if not item["done"].wait(wait):
            return False, None
        with self.cv:
            self.items.pop(key, None)
        return True, item.get("error") or item.get("result")

    def _next(self):
        now = time.monotonic()
        idle = [k for k, v in self.items.items() if now - v["asked"] > ASK_EXPIRE and (v["done"].is_set() or not v.get("taken"))]
        for k in idle:
            del self.items[k]
        todo = [(v["asked"], k) for k, v in self.items.items() if not v["done"].is_set() and not v.get("taken")]
        return max(todo)[1] if todo else None

    def _run(self):
        while True:
            with self.cv:
                while (key := self._next()) is None:
                    self.cv.wait(1.0)
                item = self.items[key]
                item["taken"] = True
            try:
                item["result"] = item["fn"]()
            except Refusal as e:
                item["error"] = {"error": refusal(e)}
            except (generator.GeneratorError, OSError) as e:
                log.warning("generator work %s failed: %s", key, e)
                item["error"] = {"error": {"reason": f"the generator did not answer ({e})", "action": "The map tries again by itself"}}
            except Exception as e:
                log.exception("generator work %s failed", key)
                item["error"] = {"error": {"reason": f"this map could not be drawn ({type(e).__name__}: {e})", "action": REPORT}}
            item["done"].set()


class Source:
    """Where candidate renders come from: the sampler of a running search of the open world on the same config (after its land
    map), else one the UI boots itself on the open world's seed and config folder, which must still be the one the match's search
    sampled on. It stops it (freeing the generator) before it starts any step or when another world is opened."""

    def __init__(self, ui):
        self.ui, self.gens, self.stack, self.own, self.clients = ui, None, None, None, {}
        self.status, self.booted, self.problem = "", None, None
        self._lock = threading.RLock()

    def _searching(self, gen, want):
        st = self.ui.runner.state() or {}
        if not (st.get("running") and st["kind"] in ("search", "resume") and st.get("job")):
            return False
        job = jobs.Job(self.ui.data, st["job"])
        mine = job.data.get("kind") == "search" and world.same(job.data.get("world", ""), self.ui.need_world().place.world)
        sampled = os.path.isfile(os.path.join(job.folder, "lmap", jobs.MANIFEST_FILE)) and search.sampled_config(job) == want[1]
        return mine and sampled and gen.record.get("kind") == "sampler"

    def sampler(self, want):
        """A sampler for want, the [seed, config key] a match's search sampled on; the seed must still be the open world's."""
        with self._lock:
            if seed_of(self.ui.need_world()) != want[0]:
                raise Refusal("another world was opened since this map was asked for", "Pick the match again on the Best matches screen")
            gen = generator.running(self.ui.cache)
            if self.own is None and gen and self._searching(gen, want):
                return search.ProbeSampler(self._client(gen), want[0], cli_search._threads(SimpleNamespace(threads=None)))
            if self.ui.runner.busy():
                raise Refusal("the step that is running uses the generator", FILLS_IN)
            if self.problem and self.problem["want"] == want and time.monotonic() - self.problem["at"] < BOOT_RETRY:
                raise Refusal(self.problem["reason"], self.problem["action"])
            if self.own is not None and self.booted != want:
                self.release()
            if self.own is None:
                self._boot_or_quiet(want)
            return self.own

    def _boot_or_quiet(self, want):
        """In a replay a generator that cannot start is no problem, and one that is not set up is never set up (that downloads the
        pack): the maps show what the recorded run drew, and the rest stays hatched, with no error. A generator set up on another
        machine (a bundle's cache) fails its Java check here and is asked again only after BOOT_RETRY seconds."""
        if self.ui.replay and generator.settings(self.ui.cache) is None:
            raise Refusal(REPLAY_ONLY, FILLS_IN)
        try:
            self._boot(want)
        except Exception as e:
            if not self.ui.replay:
                raise
            self.problem = {"reason": REPLAY_ONLY, "action": FILLS_IN, "want": want, "at": time.monotonic()}
            raise Refusal(REPLAY_ONLY, FILLS_IN) from e

    def _client(self, gen):
        key = gen.record["instance"]
        if key not in self.clients:
            self.clients = {key: gen.connect()}
        return self.clients[key]

    def _boot(self, want):
        """Boot the UI's own generator; a refusal or a crash is kept for BOOT_RETRY seconds for every map that asks (shown on the
        page as the maps' note), so the tiles waiting for it do not boot it again one by one."""
        self.release()
        opened = self.ui.need_world()
        self.status = "starting the map generator (about 1 to 2.5 minutes)"
        stack = ExitStack()
        try:
            stack.enter_context(generator.lock(self.ui.cache, self.ui.url))
            args = SimpleNamespace(accept_eula=self.ui.settings.get("eula", False), threads=None)
            gens = cli_search.Generators(self.ui.cache, args, lambda e: None, opened.place.config)
            changed = jobs.config_change(opened.place.config, want[1], "this search")
            if changed:
                raise Refusal(f"{changed}, so its matches may no longer be what RWG2 makes", "Search again on the Best matches screen")
            self.own = gens.sampler(want[0], cli_search._threads(args))()
        except Refusal as e:
            self._failed(stack, want, refusal(e))
            raise
        except Exception as e:
            self._failed(stack, want, {"reason": f"the map generator could not start ({type(e).__name__}: {e})", "action": REPORT})
            raise
        except BaseException:
            self._failed(stack, want, None)
            raise
        self.gens, self.stack, self.status, self.booted = gens, stack, "", want

    def _failed(self, stack, want, problem):
        stack.close()
        self.status = ""
        self.problem = problem and problem | {"want": want, "at": time.monotonic()}

    def release(self):
        """Stop the UI's own generator and free the generator for a step; a remembered boot problem goes with it."""
        with self._lock:
            if self.gens:
                self.gens.stop()
            if self.stack:
                self.stack.close()
            self.gens = self.stack = self.own = self.booted = self.problem = None

    def render(self, want):
        """A sampler with one retry: a generator that stopped (idle timeout, its search ended) is replaced."""
        try:
            return self.sampler(want)
        except generator.GeneratorError:
            self.release()
            return self.sampler(want)


@dataclass
class Opened:
    """The world the UI works on: its check, its scan (read in the background) and what the area screen needs."""

    place: world.Place
    remote: bool
    level: dict
    report: dict
    progress: list = field(default_factory=lambda: [0, 0])
    sc: object = None
    error: dict = None
    table: dict = None
    scene: dict = None
    claims: dict = None
    release: str = None
    looked: float = field(default_factory=time.monotonic)

    def usage(self):
        """Whether a game or server has the world open, looked at again every IN_USE_SECONDS: the screens show it live."""
        if time.monotonic() - self.looked > IN_USE_SECONDS:
            u = usage_of(self.place, self.remote)
            self.report["in_use"], self.looked = {"state": u.state, "detail": u.detail}, time.monotonic()
        return self.report["in_use"]

    def card(self):
        free = shutil.disk_usage(self.place.root).free
        self.usage()
        out = self.report | {"remote": self.remote, "free_disk_mb": free // MB, "free_ram_mb": generator.free_ram() // MB}
        out["scan"] = self.sc.summary() if self.sc else {"done": self.progress[0], "total": self.progress[1], "error": self.error}
        out["release"] = self.release
        if self.sc:
            out["bounds"] = bounds(self.sc)
        return out


def scan_point(sc, release, xz):
    """Ground and biome name of a scanned world at xz; both None where nothing was explored."""
    g = float(sc.at("ground", [xz])[0])
    if np.isnan(g):
        return {"ground": None, "biome": None}
    return {"ground": int(g), "biome": appearance.biome_names(release).get(int(sc.at("biome", [xz])[0]))}


def seed_of(opened):
    return opened.level["Data"]["RandomSeed"]


def usage_of(place, remote):
    """procs.in_use, also for a world folder that is missing between the two renames of an Apply or a Roll back."""
    if remote:
        return procs.remote()
    return procs.in_use(place.root, place.world if os.path.isdir(place.world) else None)


def migrated_report(place, rec, chosen=None):
    """The World screen's card of a world this tool migrated, or of one whose Roll back stopped between its two renames; its
    config folder is the one Verify reads."""
    u = usage_of(place, rec and rec.get("remote"))
    return {
        "world": place.world,
        "kind": place.kind,
        "root": place.root,
        "config": apply.config_in_use(place, rec, chosen),
        "config_from": world.CHOSEN if chosen else None,
        "config_chosen": chosen,
        "claims": {},
        "warnings": [],
        "onedrive": world.on_onedrive(place.world),
        "in_use": {"state": u.state, "detail": u.detail},
        "migrated": {"state": rec["state"] if rec else ROLLING_BACK, "date": (rec or {}).get("date"), "job": (rec or {}).get("apply_job")},
    }


def bounds(sc):
    xs = [k[0] for k in sc.regions]
    zs = [k[1] for k in sc.regions]
    side = scan.BLOCKS
    return [min(xs) * side, min(zs) * side, (max(xs) + 1) * side, (max(zs) + 1) * side] if xs else [0, 0, side, side]


class Drawn:
    """The tile blocks drawn so far, numbered, for the browser's tile stream: it retries a waiting tile as soon as its block is on
    disk instead of on a timer."""

    def __init__(self):
        self.events, self.seq, self.cv = deque(maxlen=DRAWN_KEPT), 0, threading.Condition()
        tiles.Layer.listeners.append(self.add)

    def add(self, folder, zoom, block):
        tx, ty, n = block
        with self.cv:
            self.seq += 1
            self.events.append({"id": self.seq, "v": os.path.basename(folder), "z": zoom, "x": tx, "y": ty, "n": n})
            self.cv.notify_all()

    def since(self, seq, timeout):
        with self.cv:
            self.cv.wait_for(lambda: self.seq > seq, timeout)
            return [e for e in self.events if e["id"] > seq]


class Maps:
    """The tile layers: the world (by the reference look it is judged against), candidates, and a preview's working copy."""

    def __init__(self, ui):
        self.ui, self.layers, self.luts, self.scans, self.after = ui, {}, {}, {}, None
        self.source, self.asks, self.drawn = Source(ui), Asks(), Drawn()
        self._lock = threading.RLock()

    def migrated(self):
        """The Verify map's parts of the open migrated world, read once in the background: {"parts": None, "progress": text} until
        then. The backup's scan comes mostly from the cache of the world before Apply: Apply only renamed its files."""
        with self._lock:
            o = self.ui.need_world(migrated=True)
            rec = apply.record_of(o.place.world)
            if not rec or rec.get("state") != apply.APPLIED or rec.get("apply_job") != o.report["migrated"]["job"]:
                raise Refusal(
                    f"{o.place.world} was rolled back or applied again since it was opened here",
                    "Open it again on the World screen",
                )
            a = self.after
            if a is None or a["world"] != o.place.world:
                a = self.after = {"world": o.place.world, "progress": "", "parts": None, "error": None}
                threading.Thread(target=self._read_migrated, args=(o.place, a), daemon=True, name="ui-after-map").start()
        if a["error"]:
            raise Refusal(a["error"]["reason"], a["error"]["action"])
        return a

    def _read_migrated(self, place, a):
        try:
            a["parts"] = self._migrated_parts(place, a)
        except Refusal as e:
            a["error"] = refusal(e)
        except Exception as e:
            log.exception("the map after Apply failed")
            a["error"] = {"reason": f"the map of your world could not be read ({type(e).__name__}: {e})", "action": REPORT}

    def _migrated_parts(self, place, a):
        rec = apply.record_of(place.world)
        preview = jobs.Job(self.ui.data, rec.get("preview_job") or "-")
        if not preview.data or not os.path.isdir(os.path.join(preview.folder, genworld.RENDER)):
            raise Refusal(
                "the preview of this migration is no longer in the data folder, or it did not generate the new terrain",
                "The checks on this screen work without the map",
            )
        chosen = maps.pick(jobs.find(self.ui.data, preview.data["args"]["search"]), preview.data["args"]["pick"])
        release = world.release_for(world.mods(world.read_level(place.world)), world.releases())[0]
        backup = apply.resolve(rec["backup"], place.root)

        def progress(what):
            return lambda done, total: a.update(progress=f"{what}: {done:,} of {total:,} region files")

        cache, before = self.ui.cache, None
        if os.path.isdir(backup):
            scan.adopt(backup, place.world, cache)
            before = scan.scan(backup, cache, None, progress("Reading your world from before Apply"))
        now = scan.scan(place.world, cache, None, progress("Reading your world now"))
        area_json = jobs.read_json(os.path.join(preview.folder, "area.json"))
        ref = maps.reference_of(maps.Scene(before or now, area_json, release, chosen if chosen["reference"] else None, []))
        lut_now = self.lut(now, ref, release)
        lut_before = self.lut(before, ref, release) if before else lut_now
        judge = score.Judge(ref, frozenset(chosen["bans"]), appearance.table(release))
        lut = tiles.merge(lut_before, lut_now)
        return SimpleNamespace(preview=preview, area=area_json, ref=ref, bans=chosen["bans"], release=release, before=before, now=now,
                               lut_before=lut_before, lut=lut, render=genworld.Render(preview), judge=judge)  # fmt: skip

    def after_parts(self):
        p = self.migrated()["parts"]
        if p is None:
            raise Refusal("the map of your world is still being read", "Wait a moment; the map appears when it is done")
        return p

    def folder(self, kind, parts):
        return os.path.join(self.ui.cache, "tiles", f"{kind}-{jobs.key_of([parts, tiles.VERSION, tiles.look_rules()])}")

    def lut(self, sc, ref, release):
        key = (search.scan_key(sc), jobs.key_of(ref.__dict__))
        if key not in self.luts:
            self.luts[key] = tiles.biome_lut(sc, ref, release)
        return self.luts[key]

    def layer(self, name):
        kind, _, rest = name.partition(".")
        if kind not in AFTER_LAYERS:
            self.ui.need_world()
        elif rest != self.after_parts().preview.id:
            raise Refusal(f"the map {name} is of another migrated world than the one open now", "Reload this page")
        with self._lock:
            if name not in self.layers:
                self.layers[name] = self._make(name)
            return self.layers[name]

    def _make(self, name):
        kind, _, rest = name.partition(".")
        if kind in AFTER_LAYERS:
            return self._after_layer(kind)
        opened = self.ui.need_world()
        if kind == "area":
            ref = self.ui.area_reference()
            draw = tiles.world_draw(opened.sc, self.lut(opened.sc, ref, opened.release))
            return tiles.Layer(self.folder("now", [search.scan_key(opened.sc), ref.__dict__]), draw)
        if kind == "now":
            sc, ref = opened.sc, self.ui.search_parts(rest)[1]
            draw = tiles.world_draw(sc, self.lut(sc, ref, opened.release))
            return tiles.Layer(self.folder("now", [search.scan_key(sc), ref.__dict__]), draw)
        if kind == "smooth":
            preview = jobs.find(self.ui.data, rest)
            sc, ref = self.staging_scan(preview), self.ui.search_parts(preview.data["args"]["search"])[1]
            lut = tiles.merge(self.lut(opened.sc, ref, opened.release), self.lut(sc, ref, opened.release))
            parts = [search.scan_key(sc), search.scan_key(opened.sc), ref.__dict__]
            return tiles.Layer(self.folder("smooth", parts), tiles.world_draw(sc, lut))
        if kind == "cand":
            job_id, _, cfg = rest.partition(".")
            scene = self.ui.candidate_scene(*_cand_name(rest))
            lut = self.lut(opened.sc, maps.reference_of(scene), opened.release)
            want = scene.pick["generator"]
            draw = tiles.candidate_draw(scene, lut, lambda: self.source.render(want))
            parts = [job_id, cfg, want, search.scan_key(opened.sc), jobs.key_of(scene.area["kept"]), scene.pick["reference"]]
            parts.append(scene.pick["bans"])
            return tiles.Layer(self.folder("cand", parts), draw, tiles.render_block)
        raise KeyError(name)

    def _after_layer(self, kind):
        """The Verify map's layers: what the preview planned, the world from before Apply, and the world now."""
        p = self.after_parts()
        if kind == "before":
            if p.before is None:
                raise Refusal("your world from before Apply is no longer in its backup folder", "The other maps still show")
            return tiles.Layer(self.folder("now", [search.scan_key(p.before), p.ref.__dict__]), tiles.world_draw(p.before, p.lut_before))
        if kind == "applied":
            parts = [search.scan_key(p.now), p.before and search.scan_key(p.before), p.ref.__dict__]
            return tiles.Layer(self.folder("smooth", parts), tiles.world_draw(p.now, p.lut))
        kept = set(map(tuple, p.area["kept"]))
        parts = [p.preview.id, search.scan_key(p.now), p.ref.__dict__, p.bans]
        return tiles.Layer(self.folder("planned", parts), tiles.planned_draw((p.now, p.lut), p.render, p.judge, kept))

    def after_probe(self, kind, xz):
        """Ground and biome at xz on a Verify map: the preview's render outside the kept area, else the world (now or before)."""
        p = self.after_parts()
        if kind == "planned" and (xz[0] // maps.CHUNK, xz[1] // maps.CHUNK) not in set(map(tuple, p.area["kept"])):
            g, b = p.render.at([xz])[0], int(p.render.biome_at([xz])[0])
            return {"ground": None if np.isnan(g) else int(g), "biome": p.render.names[b] if b >= 0 else None}
        return scan_point(p.before if kind == "before" and p.before else p.now, p.release, xz)

    def url(self, name):
        layer = self.layer(name)
        return f"/tiles/{name}/{{z}}/{{x}}/{{y}}.png?v={os.path.basename(layer.folder)}&t={self.ui.token}"

    def staging_scan(self, preview):
        stage = preview.data["args"]["staging"]
        if stage not in self.scans:
            if not os.path.isdir(stage):
                raise Refusal("the working copy of this preview is gone (applied or replaced)", "Look at your world on the Verify screen")
            self.scans[stage] = scan.scan(stage, self.ui.cache)
        return self.scans[stage]

    def tile(self, name, v, at):
        """(path, None) or (None, the reason the tile is not there yet). at: (zoom, tx, ty). v: the layer folder the page's tile
        URL names; a tile URL is cached as immutable, so one of a world or step no longer open is refused, never answered from
        the open one."""
        zoom, tx, ty = at
        layer = self.layer(name)
        if v != os.path.basename(layer.folder):
            raise Refusal(f"the map {name} on this page was drawn for a world or step that is no longer open", "Reload this page")
        if os.path.isfile(layer.path(zoom, tx, ty)):
            return layer.path(zoom, tx, ty), None
        if not name.startswith("cand."):
            return layer.tile(zoom, tx, ty), None
        n = layer.block(tiles.bpp_of(zoom)) // tiles.TILE
        done, out = self.asks.ask((name, zoom, tx // n, ty // n), lambda: layer.tile(zoom, tx, ty))
        if done and isinstance(out, str):
            return out, None
        return None, (out or {}).get("error") or {"reason": self.source.status or "drawing", "action": "wait"}


class UI:
    def __init__(self, data, cache, port=0, replay=None):
        """replay: (search job, speed) for ui --replay."""
        self.data, self.cache = os.path.abspath(data), os.path.abspath(cache)
        self.token = secrets.token_urlsafe(TOKEN_BYTES)
        self.settings = jobs.read_json(os.path.join(self.data, SETTINGS_FILE)) or {}
        self.replay = bool(replay)
        self.opened, self.area, self.maps = None, None, Maps(self)
        self.runner = Replay(self, *replay) if replay else Runner(self)
        self.shown = threading.local()
        self.server = Server((LOCALHOST, port), Handler)
        self.server.ui = self
        self.port = self.server.server_address[1]
        self.url = f"http://{LOCALHOST}:{self.port}/?t={self.token}"
        self.hosts = {f"{LOCALHOST}:{self.port}", f"localhost:{self.port}"}

    def save_settings(self, **changes):
        self.settings |= changes
        os.makedirs(self.data, exist_ok=True)
        write_atomic(os.path.join(self.data, SETTINGS_FILE), json.dumps(self.settings, indent=1).encode())

    def need_world(self, migrated=False):
        """The open world, refused to a page that still shows another one: the open world is one for all tabs. migrated: a world
        this tool migrated (the Verify screen's map), else one to migrate."""
        if self.opened is None:
            raise Refusal("no world is open", "Pick your world on the World screen")
        shown = getattr(self.shown, "world", None)
        if shown and not world.same(shown, self.opened.place.world):
            raise Refusal(
                f"this page still shows {shown}, but {self.opened.place.world} was opened since, in another tab or window",
                f"Reload this page to go on with the world open now, or open {shown} again on the World screen",
            )
        if migrated:
            if self.opened.report.get("migrated", {}).get("state") != apply.APPLIED:
                raise Refusal(f"{self.opened.place.world} is not a world this tool migrated", "The map here shows a migrated world")
            return self.opened
        if "migrated" in self.opened.report:
            raise Refusal(
                f"{self.opened.place.world} already uses RWG2: this tool migrated it",
                "Check it on the Verify screen; to migrate another world, open it on the World screen",
            )
        if self.opened.sc is None:
            raise Refusal("the map of your world is still being read", "Wait a moment; the map appears when it is done")
        return self.opened

    def ui_folder(self):
        return os.path.join(self.data, "ui", jobs.key_of(os.path.normcase(os.path.realpath(self.opened.place.world))))

    def area_reference(self):
        """The look the area screen colours the world by: the base inside the claims, else where players spent their time,
        else around the world spawn."""
        o = self.need_world()
        claimed = set().union(*o.claims.values()) if o.claims else set()
        spawn = area.dilate({area._chunk(*world.spawn(o.level))}, 4)
        for core in (claimed, set(area.suggestion(o.table, claimed)["main"]), spawn):
            if core:
                try:
                    return appearance.reference(o.sc, core, o.release)
                except Refusal:
                    continue
        raise Refusal("no explored ground was found near the world spawn or the claims", "Pick another world")

    def saved_area(self, world_dir):
        """The area last chosen on the Area screen for world_dir: (its file, its content), or (None, None)."""
        saved = self.settings.get("areas", {}).get(world_dir)
        a = saved and jobs.read_json(saved["file"])
        return (saved["file"], a) if a else (None, None)

    def area_changed(self, job):
        """The area chosen now for the search's world, as {core, kept} counts, when it differs from the one the search was made
        for (in core or kept); else None."""
        _, now = self.saved_area(job.data.get("world"))
        then = jobs.read_json(os.path.join(job.folder, cli_search.AREA_FILE))
        if not (now and then) or all(sorted(map(tuple, now[k])) == sorted(map(tuple, then[k])) for k in ("core", "kept")):
            return None
        return {"core": len(now["core"]), "kept": len(now["kept"])}

    def search_parts(self, job_id):
        """(job, reference look, bans) of a search job, before and after it has candidates."""
        job = jobs.find(self.data, job_id)
        r = jobs.read_json(os.path.join(job.folder, "site", "reference.json"))
        if not r:
            raise Refusal("the search has not measured your base yet", "Wait a moment")
        ref = appearance.Reference.from_dict(r)
        return job, ref, sorted(appearance.bans_for(ref, job.data["args"]["prefs"]))

    def candidate_scene(self, job_id, config):
        """A match of a search of the open world, drawn by a generator on the seed and config folder the search sampled on (as map
        does); the generator refuses to boot once that config folder changed."""
        job, ref, bans = self.search_parts(job_id)
        o = self.need_world()
        if not world.same(job.data.get("world", ""), o.place.world):
            raise Refusal(f"this search was made for {job.data.get('world')}, not for the open world", "Open that world on the Home screen")
        a = jobs.read_json(os.path.join(job.folder, cli_search.AREA_FILE))
        want = [seed_of(o), search.sampled_config(job)]
        pick = {"job": job, "candidate": {"config": config}, "reference": ref.__dict__, "bans": bans, "rank": 0, "generator": want}
        return maps.Scene(o.sc, a, o.release, pick, [])

    def open_world(self, path, remote=None):
        """Open a world to migrate, or one this tool migrated (for Verify, the rules and Roll back). remote None keeps the choice
        last made for this world: 'I copied this from another machine' holds from the download to Apply and Roll back."""
        place = self._place(path)
        rec = apply.record_of(place.world)
        if remote is None:
            remote = self.remote_of(place.world)
        else:
            self.save_settings(remote=self.settings.get("remote", {}) | {place.world: remote})
        migrated = rec or not os.path.isdir(place.world)
        chosen = self.config_choice(place.world)
        o = Opened(place, remote, {}, migrated_report(place, rec, chosen)) if migrated else self._legacy(place, remote)
        if self.opened is None or not world.same(self.opened.place.world, place.world):
            self.maps.source.release()
        self.opened, self.area = o, None
        self.maps.layers.clear()
        self.maps.scans.clear()
        self.maps.after = None
        recent = [place.world] + [p for p in self.settings.get("recent", []) if p != place.world]
        self.save_settings(recent=recent[:RECENT])
        return o.card()

    def _place(self, path):
        """The world at path. A world folder missing because an Apply or a Roll back stopped between its two renames is placed by
        its parent: Continue on the Home screen finishes that Apply, Roll back on the Verify screen that Roll back."""
        place = world.locate(path, exists=False)
        if os.path.isdir(place.world) or os.path.isfile(apply.rollback_journal(place)):
            return place
        if any(apply.is_open(j) and world.same(j.data.get("world", ""), place.world) for j in jobs.all_jobs(self.data)):
            raise Refusal(
                f"{place.world} is not there now: an Apply stopped between moving your world aside and moving the new one in",
                "Click Continue for that Apply on the Home screen: it finishes the move",
            )
        return world.locate(path)

    def config_choice(self, world_dir):
        """The config folder last picked on the World screen for world_dir, while it is still a folder; else None."""
        chosen = self.settings.get("configs", {}).get(world_dir)
        return chosen if chosen and os.path.isdir(chosen) else None

    def config_flag(self, world_dir):
        """--config-dir for a step of world_dir, when a config folder was picked for it on the World screen."""
        chosen = self.config_choice(world_dir)
        return ["--config-dir", chosen] if chosen else []

    def remote_of(self, world_dir):
        """'I copied this from another machine' as last chosen for this world, else as its migration record says."""
        rec = apply.record_of(world_dir)
        return self.settings.get("remote", {}).get(world_dir, bool(rec and rec.get("remote")))

    def _legacy(self, place, remote):
        chosen = self.config_choice(place.world)
        place = replace(place, config=chosen) if chosen else place
        level, report = cli.world_report(
            place, SimpleNamespace(remote=remote, require_stopped=False, confirm_stopped=False, config_dir=chosen)
        )
        o = Opened(place, remote, level, report | {"config_chosen": chosen})
        o.release = world.release_for(world.mods(level), world.releases())[0]
        o.claims = world.claims(place.world)
        threading.Thread(target=self._scan, args=(o,), daemon=True, name="ui-scan").start()
        return o

    def _scan(self, o):
        def progress(done, total):
            o.progress[:] = [done, total]

        try:
            sc = cli._scan(o.place, SimpleNamespace(cache=self.cache, workers=None), progress)
            _, found, warnings = cli._side_files(o.place, o.level)
            o.table = sc.chunk_table()
            o.scene = area.scene(sc, o.claims, found, world.spawn(o.level))
            o.report["warnings"] = warnings
            o.sc = sc
        except Refusal as e:
            o.error = refusal(e)
        except Exception as e:
            log.exception("scan failed")
            o.error = {"reason": f"the world could not be read ({type(e).__name__}: {e})", "action": REPORT}

    def area_of(self, body):
        """The area as the area command builds it, from the screen's teams, painted and erased chunks and margin."""
        o = self.need_world()
        teams = body.get("teams") or []
        chosen = {t: o.claims[t] for t in teams if t in o.claims}
        painted, erased = ({tuple(c) for c in body.get(k) or []} for k in ("painted", "erased"))
        a = area.build(chosen, painted, erased, int(body.get("margin", area.DEFAULT_MARGIN)))
        a, st, rm = area.record(a, o.table, o.scene)
        return a | {"world": o.place.world, "warnings": o.report.get("warnings", [])}, st, rm

    def diagnostic(self):
        vm = psutil.virtual_memory()
        o = self.opened
        st = self.runner.state() or {}
        lines = [
            f"rwg2-migrator {__version__}",
            f"OS: {platform.platform()}; Python {platform.python_version()}",
            f"RAM: {vm.total // MB:,} MB, {vm.available // MB:,} MB free",
            f"World: {o.place.world if o else 'none'}; GTNH {o.release if o else '?'}",
            f"Step: {st.get('kind')} (exit code {st.get('code')})",
            "Last log lines:",
            *(tail(self.runner.info["err"]) if self.runner.info else []),
        ]
        return "\n".join(lines)


def _job_view(job):
    """What the Home screen lists of a job. An apply job also says which preview it applies and whether it has written configs
    or moved folders: only then does Continue resume it, else the Apply screen asks again."""
    d = job.data
    view = {k: d.get(k) for k in ("id", "kind", "state", "world", "created", "updated", "error", "rolled_back", "applied")}
    if d.get("kind") == "apply":
        view |= {"preview": d.get("args", {}).get("preview"), "open": apply.is_open(job)}
    return view


def _read_new(path, offset, until=None):
    """(complete JSON lines written since offset, the offset after them); a line still being written waits, and so does every
    line from the first event later than until."""
    try:
        with open(path, "rb") as f:
            f.seek(offset)
            data = f.read()
    except FileNotFoundError:
        return [], offset
    out, at = [], offset
    for line in data[: data.rfind(b"\n") + 1].splitlines(keepends=True):
        try:
            e = json.loads(line)
        except ValueError:
            at += len(line)
            continue
        if until is not None and isinstance(e, dict) and e.get("t", 0) > until:
            break
        out.append(e)
        at += len(line)
    return out, at


def _tile_stream(handler, drawn):
    """Every tile block drawn from now on, until the page goes away."""
    handler.send_response(200)
    handler.send_header("Content-Type", "text/event-stream")
    handler.send_header("Cache-Control", "no-store")
    handler.end_headers()
    seq = drawn.seq
    while True:
        events = drawn.since(seq, HEARTBEAT)
        for e in events:
            handler.wfile.write(f"data: {json.dumps(e)}\n\n".encode())
            seq = e["id"]
        handler.wfile.write(b"" if events else b": ok\n\n")
        handler.wfile.flush()


def drives():
    if os.name == "nt":
        return [f"{d}:\\" for d in string.ascii_uppercase if os.path.isdir(f"{d}:\\")]
    return ["/", os.path.expanduser("~")]


def kind_of(path):
    if os.path.isfile(os.path.join(path, world.SERVER_PROPERTIES)):
        return "server"
    if os.path.isfile(os.path.join(path, world.LEVEL)):
        return "world"
    return "folder"


def browse(path, ext=""):
    """Drives, or the folders of path (with its files ending in ext, when given), each with what it looks like."""
    if not path:
        return {"path": "", "parent": None, "entries": [{"name": d, "path": d, "kind": "drive"} for d in drives()]}
    p = world.normalise(path)
    try:
        names = sorted(os.listdir(p), key=str.casefold)
    except OSError as e:
        raise Refusal(f"{p} cannot be opened ({e.strerror})", "Pick another folder") from e
    folders = [os.path.join(p, n) for n in names if (not n.startswith(".") or n == MINECRAFT) and os.path.isdir(os.path.join(p, n))]
    found = [os.path.join(p, n) for n in names if ext and n.lower().endswith(ext) and os.path.isfile(os.path.join(p, n))]
    parent = os.path.dirname(p)
    return {
        "path": p,
        "kind": kind_of(p),
        "parent": parent if parent != p else "",
        "entries": [{"name": os.path.basename(f), "path": f, "kind": kind_of(f)} for f in folders]
        + [{"name": os.path.basename(f), "path": f, "kind": "file"} for f in found],
    }


def open_folder(path):
    if not os.path.isdir(path):
        raise Refusal(f"{path} does not exist", "It may have been moved or deleted")
    try:
        if os.name == "nt":
            os.startfile(path)
        else:
            opener = "open" if sys.platform == "darwin" else "xdg-open"
            subprocess.Popen([opener, path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except OSError as e:
        raise Refusal(f"this computer could not open {path}", "Open that folder yourself") from e
    return {"opened": path}


def _flags(body, names):
    return [f for key, f in names.items() if body.get(key)]


APPLY_FLAGS = {
    "remote": "--remote",
    "zip": "--zip",
    "leave_players": "--leave-players",
    "allow_shared_instance": "--allow-shared-instance",
    "reset_journeymap": "--reset-journeymap",
    "confirm_stopped": "--confirm-stopped",
    "auto_restart_off": "--auto-restart-off",
}


class Api:
    """The routes; each takes (ui, query, body) and returns JSON, or raises Refusal for the error panel."""

    @staticmethod
    def get_state(ui, q, b):
        runner = ui.runner.state()
        jobs.refresh(ui.data)
        listing = [_job_view(j) for j in jobs.all_jobs(ui.data) if ui.runner.cutoff(j.id) != -math.inf]
        return {
            "version": __version__,
            "data": ui.data,
            "cache": ui.cache,
            "eula": ui.settings.get("eula", False),
            "generator": generator.settings(ui.cache) is not None,
            "recent": ui.settings.get("recent", []),
            "world": ui.opened.card() if ui.opened else None,
            "area": ui.settings.get("areas", {}).get(ui.opened.place.world) if ui.opened else None,
            "runner": runner,
            "jobs": listing,
            "drawn": tiles.Layer.drawn,
            "redrawn": tiles.Layer.redrawn,
            "map_status": "" if ui.replay else ui.maps.source.status,
            "map_problem": None if ui.replay else refusal_of(ui.maps.source.problem),
        }

    @staticmethod
    def get_saves(ui, q, b):
        return {"saves": world.saves()}

    @staticmethod
    def get_browse(ui, q, b):
        return browse(q.get("path", [""])[0], q.get("ext", [""])[0])

    @staticmethod
    def get_legend(ui, q, b):
        return tiles.legend()

    @staticmethod
    def get_diag(ui, q, b):
        return {"text": ui.diagnostic()}

    @staticmethod
    def post_eula(ui, q, b):
        ui.save_settings(eula=bool(b.get("accept")))
        return {"eula": ui.settings["eula"]}

    @staticmethod
    def post_world(ui, q, b):
        if ui.runner.busy():
            raise Refusal("a step is still running", "Wait for it before you open another world")
        return ui.open_world(b["path"], None if b.get("remote") is None else bool(b["remote"]))

    @staticmethod
    def post_config(ui, q, b):
        """Pick the config folder of the world the page shows (an empty path goes back to the one found for it) and open it again."""
        if ui.runner.busy():
            raise Refusal("a step is still running", "Wait for it before you change the config folder")
        shown = b.get("world") or (ui.opened and ui.opened.place.world)
        if not shown:
            raise Refusal("no world is open", "Pick your world on the World screen")
        path = (b.get("path") or "").strip()
        chosen = world.config_dir(path) if path else None
        target = world.locate(shown, exists=False).world
        ui.save_settings(configs=ui.settings.get("configs", {}) | {target: chosen})
        return ui.open_world(target)

    @staticmethod
    def get_world(ui, q, b):
        o = ui.need_world()
        o.claims = o.scene["claims"] = world.claims(o.place.world)
        teams = {t: sorted(c) for t, c in o.claims.items()}
        return {
            "card": o.card(),
            "claims": teams,
            "suggest": area.suggestion(o.table, set().union(*o.claims.values()) if o.claims else set()),
            "spawn": list(world.spawn(o.level)),
            "players": [p for p in o.scene["players"] if p["dim"] == 0],
            "layer": ui.maps.url("area"),
        }

    @staticmethod
    def post_area(ui, q, b):
        a, st, rm = ui.area_of(b)
        out = {
            "core": st["core"],
            "kept": a["kept"],
            "margin_chunks": st["margin_chunks"],
            "never_generated": len(st["never_generated"]),
            "unpopulated_core": len(st["unpopulated_core"]),
            "unreadable": _sentence(area.unreadable_refusal(st)),
            "dropped": st["dropped"],
            "parts": st["parts"],
            "parts_warning": area.parts_warning(st["parts"]),
            "removed": rm,
            "cells": area.removed_cells(a, ui.opened.table, ui.opened.scene),
        }
        if b.get("save"):
            folder = ui.ui_folder()
            os.makedirs(folder, exist_ok=True)
            path = os.path.join(folder, cli_search.AREA_FILE)
            write_atomic(path, (json.dumps(a, indent=1) + "\n").encode())
            keep = {k: b.get(k) for k in ("teams", "painted", "erased", "margin")}
            ui.save_settings(areas=ui.settings.get("areas", {}) | {ui.opened.place.world: keep | {"file": path}})
            out["file"] = path
        return out

    @staticmethod
    def get_look(ui, q, b):
        o = ui.need_world()
        saved = ui.settings.get("areas", {}).get(o.place.world)
        if not saved:
            raise Refusal("no area to keep was chosen yet", "Choose it on the Area screen and click Continue")
        a = jobs.read_json(saved["file"])
        ref = appearance.reference(o.sc, a["core"], o.release, q.get("base", [None])[0] or None)
        names = sorted({n for n in appearance.biome_names(o.release).values()}, key=str.casefold)
        bans = appearance.default_bans(ref)
        looks = [
            {"key": k, "label": appearance.LOOK_LABELS[k], "banned": k in bans, "fixed": k in appearance.NEVER_BANNED or k == ref.look}
            for k in appearance.LOOKS
        ]
        return {
            "reference": ref.__dict__ | {"colour": tiles.hexa(round(c) for c in ref.colour)},
            "biomes": names,
            "looks": looks,
        }

    @staticmethod
    def post_search(ui, q, b):
        o = ui.need_world()
        saved = ui.settings.get("areas", {}).get(o.place.world)
        if not saved:
            raise Refusal("no area to keep was chosen yet", "Choose it on the Area screen and click Continue")
        prefs = {"base": b.get("base") or None, "banned": b.get("banned"), "tolerance": float(b.get("tolerance") or 1.0)}
        path = os.path.join(ui.ui_folder(), "prefs.json")
        write_atomic(path, json.dumps(prefs).encode())
        eula = ["--accept-eula"] if ui.settings.get("eula") else []
        argv = ["search", o.place.world, "--area", saved["file"], "--prefs", path, *ui.config_flag(o.place.world), *eula, "--json"]
        return ui.runner.start("search", argv)

    @staticmethod
    def post_cancel(ui, q, b):
        jobs.find(ui.data, b["job"]).request_cancel()
        return {"asked": b["job"]}

    @staticmethod
    def get_job(ui, q, b):
        job = jobs.find(ui.data, q["id"][0])
        d, cut = job.data, ui.runner.cutoff(job.id)
        out = {"job": _job_view(job), "args": d.get("args", {})}
        if d.get("kind") == "search":
            out["candidates"] = jobs.read_json(os.path.join(job.folder, *search.CANDIDATES)) if cut is None else None
            best = [e for e in job.events() if e.get("event") == "best" and (cut is None or e.get("t", 0) <= cut)]
            out["best"] = best[-1]["top"] if best else []
            out["area"] = jobs.read_json(os.path.join(job.folder, cli_search.AREA_FILE))
            out["area_changed"] = ui.area_changed(job)
        if d.get("kind") == "preview":
            out["report"] = jobs.read_json(os.path.join(job.folder, staging.POSTCHECK, staging.REPORT))
            out["transplant"] = jobs.read_json(os.path.join(job.folder, staging.TRANSPLANT, staging.RECORD))
            out["area"] = jobs.read_json(os.path.join(job.folder, "area.json"))
            out["steps"] = preview_steps(job, cut)
            applied = d.get("applied")
            out["applied"] = bool(applied)
            out["rolled_back"] = bool(applied and jobs.Job(ui.data, applied).data.get("rolled_back"))
        return out

    @staticmethod
    def get_events(ui, q, b):
        """The job's events written after byte offset 'at' (0: all of them), the offset after them, and whether they are all there
        is: the job no longer runs and no step continues it. The page polls this instead of holding a connection open, which a
        browser allows only 6 of per host across all its tabs."""
        job = jobs.find(ui.data, q["job"][0])
        st = ui.runner.state() or {}
        end = job.state() != jobs.RUNNING and not (st.get("running") and st.get("job") in (job.id, None))
        events, at = _read_new(os.path.join(job.folder, jobs.EVENTS_FILE), int(q.get("at", ["0"])[0]), ui.runner.cutoff(job.id))
        return {"events": [labelled(e) for e in events], "at": at, "end": end}

    @staticmethod
    def get_layer(ui, q, b):
        return {"url": ui.maps.url(q["name"][0])}

    @staticmethod
    def get_heat(ui, q, b):
        """Seam steps along the kept border: a candidate's (from the generator), or a preview's working copy after smoothing."""
        o = ui.need_world()
        if "preview" in q:
            preview = jobs.find(ui.data, q["preview"][0])
            sc = ui.maps.staging_scan(preview)
            a = jobs.read_json(os.path.join(preview.folder, "area.json"))
            return {"dots": tiles.heat(*tiles.kept_border(sc, a["kept"], lambda p: sc.at("ground", p)))}
        cfg = [int(v) for v in q["config"][0].split("_")]
        scene = ui.candidate_scene(q["job"][0], cfg)
        want = scene.pick["generator"]

        def work():
            sampler = ui.maps.source.render(want)
            dots = tiles.kept_border(o.sc, scene.area["kept"], lambda p: _top(sampler, cfg, p))
            return tiles.heat(*dots)

        done, out = ui.maps.asks.ask(("heat", q["job"][0], q["config"][0]), work, HEAT_WAIT)
        if not done or (isinstance(out, dict) and out["error"]["action"] == FILLS_IN):
            return {"pending": ui.maps.source.status or "measuring"}
        if isinstance(out, dict):
            raise Refusal(out["error"]["reason"], out["error"]["action"])
        return {"dots": out}

    @staticmethod
    def get_aftermap(ui, q, b):
        """The Verify screen's map of a migrated world: what its preview planned, the world from before Apply and the world now,
        with the seam heat along the kept border now; {"pending": what is being read} until its scans are read."""
        a = ui.maps.migrated()
        p = a["parts"]
        if p is None:
            return {"pending": a["progress"] or "Reading your world for the map"}
        pid = p.preview.id
        names = [k for k in AFTER_LAYERS if k != "before" or p.before]
        layers = {f"{k}.{pid}": {"url": ui.maps.url(f"{k}.{pid}"), "label": AFTER_LAYERS[k]} for k in names}
        dots = tiles.heat(*tiles.kept_border(p.now, p.area["kept"], lambda pts: p.now.at("ground", pts)))
        return {"layers": layers, "left": f"planned.{pid}", "right": f"applied.{pid}", "core": p.area["core"], "kept": p.area["kept"],
                "heat": {f"applied.{pid}": dots}}  # fmt: skip

    @staticmethod
    def get_probe(ui, q, b):
        """The ground and biome at x, z on the map layer under the pointer: the world (NOW and the area map), a preview's working
        copy (SMOOTHED), or a match outside the kept area as its tile at that zoom was drawn (CANDIDATE)."""
        x, z = int(float(q["x"][0])), int(float(q["z"][0]))
        kind, _, rest = q.get("layer", [""])[0].partition(".")
        if kind in AFTER_LAYERS:
            return {"x": x, "z": z} | ui.maps.after_probe(kind, (x, z))
        o = ui.need_world()
        if kind == "cand":
            found = candidate_at(ui.candidate_scene(*_cand_name(rest)), (x, z), tiles.bpp_of(int(q["zoom"][0])))
            if found:
                return {"x": x, "z": z} | found
        sc = ui.maps.staging_scan(jobs.find(ui.data, rest)) if kind == "smooth" else o.sc
        return {"x": x, "z": z} | scan_point(sc, o.release, (x, z))

    @staticmethod
    def post_preview(ui, q, b):
        """The preview of a match, for the area chosen now: the step refuses a search made for another one."""
        o = ui.need_world()
        if not b.get("search") or not b.get("pick"):
            raise Refusal("the preview was asked for before the page knew which match it is for", PICK_AGAIN)
        job = jobs.find(ui.data, b["search"])
        argv = [
            "preview",
            o.place.world,
            "--area",
            ui.saved_area(o.place.world)[0] or os.path.join(job.folder, cli_search.AREA_FILE),
            "--search",
            job.id,
            "--pick",
            str(int(b["pick"])),
        ]
        argv += _flags(b, {"trees": "--trees", "blend": "--blend-biomes", "set_populated": "--set-populated"})
        argv += ui.config_flag(o.place.world) + (["--accept-eula"] if ui.settings.get("eula") else [])
        return ui.runner.start("preview", [*argv, "--json"])

    @staticmethod
    def get_apply_checklist(ui, q, b):
        preview = jobs.find(ui.data, q["preview"][0])
        place = world.Place(**preview.data["args"]["place"])
        remote = ui.remote_of(place.world)
        usage = usage_of(place, remote)
        a = jobs.read_json(os.path.join(preview.folder, "area.json")) or {}
        o = ui.opened
        now = bool(a.get("kept")) and o is not None and o.scene is not None and world.same(o.place.world, place.world)
        taint = preview.data["args"].get("taint")
        chosen = ui.config_choice(place.world)
        config, shown = chosen or place.config, chosen or world.locate(place.world, exists=False).config
        return {
            "preview": preview.id,
            "remote": remote,
            "ready": _ready(preview, place),
            "world": place.world,
            "kind": place.kind,
            "in_use": {"state": usage.state, "detail": usage.detail},
            "removed": area.removed(a, o.table, o.scene) if now else a.get("removed", {}),
            "dropped": a.get("status", {}).get("dropped", 0),
            "taint": taint,
            "taint_off_default": apply.taint_off(preview, apply.Options()),
            "other_saves": apply.other_rwg2_saves(place, config),
            "free_disk_mb": apply.free_bytes(place.root) // MB,
            "config": config,
            "config_shown": shown,
            "config_changed": not _same_folder(config, shown),
            "backup_folder": os.path.join(place.root, "rwg2-work"),
        }

    @staticmethod
    def post_apply(ui, q, b):
        preview = jobs.find(ui.data, b["preview"])
        remote = ui.remote_of(preview.data["world"])
        b = b | {"remote": remote, "zip": remote and b.get("zip")}
        argv = ["apply", preview.data["world"], "--preview", preview.id, "--yes", *_flags(b, APPLY_FLAGS)]
        argv += ["--taint-spread-off" if b.get("taint_off") else "--keep-taint-spread"]
        argv += ["--backup-copy", b["backup_copy"]] if b.get("backup_copy") else []
        return ui.runner.start("apply", [*argv, *ui.config_flag(preview.data["world"]), "--json"])

    @staticmethod
    def post_verify(ui, q, b):
        target = b.get("world") or (ui.opened.place.world if ui.opened else None)
        if not target:
            raise Refusal("no world is open", "Pick your world on the World screen")
        restore = ["--restore-offsets"] if b.get("restore") else []
        return ui.runner.start("verify", ["verify", target, *ui.config_flag(target), "--json", *restore])

    @staticmethod
    def post_rollback(ui, q, b):
        b = b | {"remote": ui.remote_of(b["world"])}
        flags = _flags(b, {"remote": "--remote", "confirm_stopped": "--confirm-stopped", "auto_restart_off": "--auto-restart-off"})
        return ui.runner.start("rollback", ["rollback", b["world"], "--yes", *flags, "--json"])

    @staticmethod
    def post_resume(ui, q, b):
        eula = ["--accept-eula"] if ui.settings.get("eula") else []
        return ui.runner.start("resume", ["jobs", "resume", b["job"], *eula, "--json"], b["job"])

    @staticmethod
    def post_discard(ui, q, b):
        return ui.runner.start("discard", ["jobs", "discard", b["job"]])

    @staticmethod
    def post_setup(ui, q, b):
        zip_path = ["--zip", world.normalise(b["zip"])] if b.get("zip") else []
        return ui.runner.start("setup", ["generator", "setup", "--accept-eula", *zip_path, "--json"])

    @staticmethod
    def post_clean(ui, q, b):
        return ui.runner.start("clean", ["clean", "--generator", "--json"])

    @staticmethod
    def post_stop(ui, q, b):
        ui.runner.stop()
        return {"stopped": True}

    @staticmethod
    def get_record(ui, q, b):
        """The migration record of a world, with its backup and JourneyMap paths resolved, for the After Apply screen."""
        place = world.locate(q["world"][0])
        rec = apply.record_of(place.world)
        rolled = [j for j in jobs.all_jobs(ui.data) if j.data.get("rolled_back") and world.same(j.data.get("world", ""), place.world)]
        if not rec and rolled:
            raise Refusal(
                f"the Apply of {place.world} was rolled back, so it uses the old terrain generator again",
                "Open it on the World screen to migrate it again",
            )
        if not rec:
            raise Refusal(f"{place.world} has no migration record", "Apply a preview first, or open the world you applied")
        out = {"record": rec, "map_tip": apply.MAP_TIP, "backup": apply.resolve(rec["backup"], place.root)}
        out["journeymap"] = apply.resolve(rec["journeymap"], place.root) if rec.get("journeymap") else None
        out["by_hand"] = apply.resolve(rec["by_hand"]["file"], place.root) if rec.get("by_hand") else None
        return out

    @staticmethod
    def post_open(ui, q, b):
        return open_folder(b["path"])


def _same_folder(a, b):
    """Whether two config folders, each a path or None, are the same one."""
    return a == b or bool(a and b and world.same(a, b))


def _sentence(refusal):
    return f"{refusal.reason}. {refusal.action}." if refusal else None


def _cand_name(rest):
    job_id, _, cfg = rest.partition(".")
    return job_id, [int(v) for v in cfg.split("_")]


def candidate_at(scene, p, bpp):
    """A match's ground and biome at p from the render its tile at bpp was drawn from ({"pending": True} until it is drawn); None
    in the kept area, which the match's map shows as it is."""
    if (p[0] // maps.CHUNK, p[1] // maps.CHUNK) in set(map(tuple, scene.area["kept"])):
        return None
    span = tiles.render_block(bpp) * bpp
    x0, z0 = p[0] // span * span, p[1] // span * span
    win = maps.render_window((x0, z0, x0 + span, z0 + span), bpp)
    renderer = maps.Renderer(scene.pick, None)
    if not renderer.cached(win, bpp):
        return {"pending": True}
    rec, names = renderer.render(win, bpp)
    v = rec[(p[1] - win[1]) // bpp, (p[0] - win[0]) // bpp]
    return {"ground": int(score.surface(v["h"])), "biome": names[int(v["palette"])]}


def _top(sampler, config, points):
    recs, _ = sampler.sample([config], np.asarray(points))
    return score.surface(recs[0]["h"]) if len(points) else np.zeros(0)


def _ready(preview, place):
    try:
        apply.ready(preview, place)
        return None
    except Refusal as e:
        return refusal(e)


class Handler(BaseHTTPRequestHandler):
    server_version = "rwg2-migrator"

    def log_message(self, fmt, *args):
        log.info("%s %s", self.address_string(), (fmt % args).replace(self.server.ui.token, "<token>"))

    def _allowed(self):
        ui = self.server.ui
        if self.headers.get("Host") not in ui.hosts:
            return False
        origin = self.headers.get("Origin")
        if origin and origin not in {f"http://{h}" for h in ui.hosts}:
            return False
        given = self.query.get("t", [""])[0] or self.headers.get("X-Token", "")
        return hmac.compare_digest(given.encode(), ui.token.encode())

    def _send(self, status, body, ctype="application/json", headers=NO_STORE):
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Content-Type-Options", "nosniff")
        for k, v in headers:
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, value, status=200):
        self._send(status, json.dumps(value, default=_plain).encode())

    def _route(self, method):
        split = urlsplit(self.path)
        self.query = parse_qs(split.query)
        if not self._allowed():
            self._send(403, b"Forbidden: open the address the rwg2-migrator window shows", "text/plain")
            return
        self.server.ui.shown.world = unquote(self.headers.get("X-World", "")) or None
        try:
            self._dispatch(method, split.path)
        except Refusal as e:
            self._json({"error": refusal(e)}, 400)
        except ConnectionError:
            pass
        except Exception as e:
            log.exception("request %s failed", split.path)
            self._json({"error": {"reason": f"{type(e).__name__}: {e}", "action": REPORT}}, 500)

    def _dispatch(self, method, path):
        ui = self.server.ui
        if method == "GET" and path == "/":
            html = static("index.html").read_text(encoding="utf-8").replace("__TOKEN__", ui.token)
            return self._send(200, html.encode(), "text/html; charset=utf-8")
        if method == "GET" and path.startswith("/static/"):
            return self._static(path[len("/static/") :])
        if method == "GET" and path.startswith("/tiles/"):
            return self._tile(path)
        if method == "GET" and path == "/api/tilestream":
            return _tile_stream(self, ui.maps.drawn)
        fn = getattr(Api, f"{method.lower()}_{path.removeprefix('/api/').replace('/', '_')}", None)
        if not path.startswith("/api/") or fn is None:
            return self._send(404, b"Not found", "text/plain")
        body = {}
        if method == "POST":
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
        return self._json(fn(ui, self.query, body))

    def _static(self, name):
        if ".." in name.split("/") or "\\" in name or ":" in name:
            return self._send(404, b"Not found", "text/plain")
        res = static(name)
        if not res.is_file():
            return self._send(404, b"Not found", "text/plain")
        ctype = mimetypes.guess_type(name)[0] or "application/octet-stream"
        return self._send(
            200, res.read_bytes(), ctype + ("; charset=utf-8" if ctype.startswith(("text/", "application/javascript")) else "")
        )

    def _tile(self, path):
        parts = path.split("/")
        if len(parts) != TILE_PATH_PARTS or not parts[5].endswith(".png"):
            return self._send(404, b"Not found", "text/plain")
        name, zoom, tx, ty = parts[2], int(parts[3]), int(parts[4]), int(parts[5][: -len(".png")])
        if zoom not in tiles.ZOOMS:
            return self._send(404, b"Not found", "text/plain")
        found, why = self.server.ui.maps.tile(name, self.query.get("v", [""])[0], (zoom, tx, ty))
        if found is None:
            return self._json({"error": why}, 503)
        with open(found, "rb") as f:
            age = f"{time.time() - os.fstat(f.fileno()).st_mtime:.3f}"
            return self._send(200, f.read(), "image/png", (("Cache-Control", TILE_CACHE), ("X-Tile-Age", age)))

    def do_GET(self):
        self._route("GET")

    def do_POST(self):
        self._route("POST")


class Server(ThreadingHTTPServer):
    """Errors go to the log file, never to the console the user keeps open."""

    daemon_threads = True

    def handle_error(self, request, client_address):
        log.exception("connection from %s failed", client_address)


def _plain(v):
    if isinstance(v, set | tuple):
        return list(v)
    if isinstance(v, np.generic):
        return v.item()
    raise TypeError(type(v).__name__)


def running_ui(data):
    """The URL of the UI already running for this data folder, if any."""
    held = jobs.read_json(os.path.join(data, UI_FILE))
    if held and procs.is_alive(procs.ProcessId(held["pid"], held["create_time"], "")):
        return held["url"]
    return None


def _quiet_logging(data):
    folder = os.path.join(data, "logs")
    os.makedirs(folder, exist_ok=True)
    handler = logging.FileHandler(os.path.join(folder, "ui.log"), encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(logging.INFO)


def open_browser(url):
    """Open url in the default browser; the log says whether the system took it (a headless machine has no browser to open)."""
    opened = webbrowser.open(url)
    log.info("webbrowser.open(%s) returned %s", url.split("?")[0], opened)
    return opened


def announce(data, url):
    """Write ui.json, which holds the page's address with its token: no other user can read it from the moment it exists."""
    me, path = psutil.Process(), os.path.join(data, UI_FILE)
    mask = os.umask(OWNER_ONLY)
    try:
        write_atomic(path, json.dumps({"pid": me.pid, "create_time": me.create_time(), "url": url}).encode())
    finally:
        os.umask(mask)
    os.chmod(path, PRIVATE_FILE)


def serve(args):
    """The ui command: open the running UI if there is one, else start it; the console shows only the address and one line."""
    data = os.path.abspath(args.data or jobs.data_root())
    cache = os.path.abspath(args.cache or scan.cache_root())
    _quiet_logging(data)
    url = running_ui(data)
    if url:
        print(url, flush=True)
        if not args.no_browser:
            open_browser(url)
        return 0
    jobs.refresh(data)
    replay = (args.replay, args.speed or REPLAY_SPEED) if getattr(args, "replay", None) else None
    ui = UI(data, cache, args.port or 0, replay)
    announce(data, ui.url)
    print(ui.url, flush=True)
    print(KEEP_OPEN, flush=True)
    if not args.no_browser:
        open_browser(ui.url)
    try:
        ui.server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        ui.maps.source.release()
        if (jobs.read_json(os.path.join(data, UI_FILE)) or {}).get("url") == ui.url:
            os.remove(os.path.join(data, UI_FILE))
    return 0
