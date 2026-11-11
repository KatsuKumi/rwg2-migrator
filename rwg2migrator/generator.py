"""The RWG2 generator: the official GTNH server pack plus our rwg2probe mod, run headless on this machine."""

import base64
import hashlib
import json
import os
import re
import secrets
import socket
import subprocess
import threading
import time
import uuid
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import psutil

from . import ISSUES, Refusal, forgecfg, procs
from .jobs import read_json, write_json
from .provision import PRISTINE, QUIET_JAVA, _quiet, check_pack, data, pack_dir
from .region import copy_tree, rmtree, write_atomic

PROBE_VERSION = "0.3.1"
PROBE_JAR = f"rwg2probe-{PROBE_VERSION}.jar"
AGAIN = f"Run generator setup again; if it repeats, report it at {ISSUES}"
SAMPLER, GEN = "sampler", "gen"
SAMPLER_WORLD = "probe-world"
DEFAULT_HEAP = "3G"
IDLE_MINUTES = 5.0
HEARTBEAT_SECONDS = 60
BOOT_TIMEOUT = 15 * 60
BOOT_EVENT_SECONDS = 15
STOP_TIMEOUT = 45
STOP_ASK_TIMEOUT = 10
STATUS_TIMEOUT = 10
REQUEST_TIMEOUT = 2 * 60 * 60
POLL = 0.5
MEMORY_POLL = 1.0
GB = 1 << 30
UNITS = {"": 1, "k": 1 << 10, "m": 1 << 20, "g": GB, "t": 1 << 40}
RAM_SPARE = GB
NICE_BELOW_NORMAL = 10
TOKEN_BYTES = 32
LOG_TAIL = 12
TMP = "tmp"
LOCALHOST = "127.0.0.1"
TOKEN_FILE = "rwg2probe.token"
PORT_FILE = "rwg2probe.port"
RECORD_FILE = "generator.json"
LOCK_FILE = "generator.lock"
SETTINGS_FILE = "generator-settings.json"
CONSOLE_LOG = os.path.join("logs", "rwg2probe-console.log")
MARKER = "-Drwg2probe.instance="
XMX = re.compile(r"^-Xmx(\d+)([kKmMgGtT]?)$")
RECORD = np.dtype([("h", ">f4"), ("palette", ">i2"), ("meta", "u1"), ("ocean", "u1")])
BATCH_RECORDS = 3_000_000
GEN_BATCH = 256
DIGEST_CHARS = 16
SELFTEST = "selftest/spec.json"
SU_CFG = os.path.join("serverutilities", "serverutilities.cfg")
SU_FORCED = {
    ("backups", "enable_backups"): ("B", "false"),
    ("backups", "backup_folder_path"): ("S", "./backups/"),
    ("auto_shutdown", "enabled"): ("B", "false"),
}
RWG2_CFG = os.path.join("config", "rwg2.cfg")
RWG2_CATEGORY = "expert / worldgen migration"
OFFSET_KEYS = ("Biome Placement Offset X", "Biome Placement Offset Z", "Landmass Placement Offset X", "Landmass Placement Offset Z")
PROPERTIES = {
    "server-ip": LOCALHOST,
    "server-port": "0",
    "white-list": "true",
    "enable-rcon": "false",
    "enable-query": "false",
    "snooper-enabled": "false",
    "allow-nether": "false",
    "motd": "rwg2probe (not a game server)",
}
SAMPLER_PROPERTIES = {
    "level-type": "FLAT",
    "generate-structures": "false",
    "spawn-animals": "false",
    "spawn-monsters": "false",
    "spawn-npcs": "false",
    "view-distance": "3",
}
EULA = "#Accepted by the user in rwg2-migrator (https://aka.ms/MinecraftEULA)\neula=true\n"


class GeneratorError(RuntimeError):
    pass


@dataclass(frozen=True)
class Boot:
    kind: str = SAMPLER
    world: str = SAMPLER_WORLD
    overlay: str | None = None
    offsets: tuple | None = None
    heap: str = DEFAULT_HEAP
    jvm: tuple = ()
    idle_minutes: float = IDLE_MINUTES
    properties: dict = field(default_factory=dict)


@dataclass
class Context:
    cache: str
    release: str
    entry: dict
    java: object = None
    accept_eula: bool = False
    emit: object = _quiet


def heap_bytes(text):
    m = XMX.match("-Xmx" + text)
    if not m:
        raise Refusal(f"{text!r} is not a Java heap size", "Use a size such as 3G or 4096M")
    return int(m.group(1)) * UNITS[m.group(2).lower()]


def _xmx(cmdline):
    sizes = [heap_bytes(a[len("-Xmx") :]) for a in cmdline if XMX.match(a)]
    return sizes[-1] if sizes else 0


def jvm_headroom():
    """Memory the running JVMs may still claim: their -Xmx minus what they hold now."""
    total = 0
    for p in procs.java_processes():
        try:
            total += max(0, _xmx(p.cmdline()) - p.memory_info().rss)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
    return total


def free_ram():
    """Memory free now, less what the running JVMs may still claim (a server on this machine among them)."""
    return psutil.virtual_memory().available - jvm_headroom()


def require_ram(heap):
    free = free_ram()
    need = heap_bytes(heap) + RAM_SPARE
    if free < need:
        raise Refusal(
            f"The generator needs {need / GB:.1f} GB free; {max(free, 0) / GB:.1f} GB is free now",
            "Close GTNH (or other Java programs), then run this again",
        )


def offset_edits(offsets):
    return {(RWG2_CATEGORY, k): ("I", str(int(v))) for k, v in zip(OFFSET_KEYS, offsets, strict=True)}


def require_eula(folder):
    path = os.path.join(folder, "eula.txt")
    text = Path(path).read_text(encoding="utf-8", errors="replace") if os.path.exists(path) else ""
    if not re.search(r"^eula=true\s*$", text, re.MULTILINE):
        raise Refusal(
            "the generator is a Minecraft server, and the Minecraft EULA has not been accepted",
            "Read https://aka.ms/MinecraftEULA, then run this again with --accept-eula",
        )


def _eula(folder, accept):
    if accept:
        write_atomic(os.path.join(folder, "eula.txt"), EULA.encode())
    else:
        require_eula(folder)


def _rebuild_config(folder, overlay):
    config = os.path.join(folder, "config")
    rmtree(config)
    copy_tree(os.path.join(folder, PRISTINE), config)
    if overlay:
        copy_tree(overlay, config)


def _install_probe(folder):
    mods = os.path.join(folder, "mods")
    for name in os.listdir(mods):
        if name.startswith("rwg2probe-") and name.endswith(".jar"):
            os.remove(os.path.join(mods, name))
    with open(os.path.join(mods, PROBE_JAR), "wb") as f:
        f.write(data(PROBE_JAR).read_bytes())


def _properties(folder, boot):
    values = PROPERTIES | {"level-name": boot.world}
    values |= SAMPLER_PROPERTIES if boot.kind == SAMPLER else {}
    values |= boot.properties
    text = "#rwg2-migrator generator (not a game server)\n" + "".join(f"{k}={v}\n" for k, v in values.items())
    write_atomic(os.path.join(folder, "server.properties"), text.encode())


def _token(folder):
    token = secrets.token_hex(TOKEN_BYTES)
    path = os.path.join(folder, TOKEN_FILE)
    tmp = path + ".tmp"
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(token)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    if os.path.exists(os.path.join(folder, PORT_FILE)):
        os.remove(os.path.join(folder, PORT_FILE))
    return token


def prepare(folder, boot, accept_eula):
    """Make the pack folder ready for one boot; returns the session token. Nothing is written without EULA consent."""
    _eula(folder, accept_eula)
    _rebuild_config(folder, boot.overlay)
    forgecfg.set_keys(os.path.join(folder, SU_CFG), SU_FORCED)
    if boot.offsets is not None:
        mine = os.path.join(boot.overlay or "", os.path.basename(RWG2_CFG))
        forgecfg.edit(os.path.join(folder, RWG2_CFG), offset_edits(boot.offsets), mine if os.path.isfile(mine) else None)
    _install_probe(folder)
    _properties(folder, boot)
    return _token(folder)


def jvm_args(java_path, boot, instance):
    """The pack's own launch line plus the probe's settings. The token is in a file, never here. Paths are relative to the pack
    folder the JVM starts in: Windows hands java.exe its arguments in the ANSI code page, which can turn the cache path into '?'."""
    return [
        java_path,
        f"-Xmx{boot.heap}",
        *QUIET_JAVA,
        f"-Djava.io.tmpdir={TMP}",
        f"-Djdk.net.unixdomain.tmpdir={TMP}",
        "-Dfml.readTimeout=180",
        "-Duser.language=en",
        "-Dfml.queryResult=confirm",
        "-Drwg2probe.port=0",
        f"{MARKER}{instance}",
        f"-Drwg2probe.parentPid={os.getpid()}",
        f"-Drwg2probe.idleMinutes={boot.idle_minutes}",
        f"-Drwg2probe.kind={boot.kind}",
        *boot.jvm,
        "@java9args.txt",
        "-jar",
        "lwjgl3ify-forgePatches.jar",
        "nogui",
    ]


def _detached():
    if os.name == "nt":
        flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.BELOW_NORMAL_PRIORITY_CLASS
        return {"creationflags": flags}
    return {"start_new_session": True}


def spawn(args, folder):
    os.makedirs(os.path.join(folder, "logs"), exist_ok=True)
    os.makedirs(os.path.join(folder, TMP), exist_ok=True)
    with open(os.path.join(folder, CONSOLE_LOG), "wb") as log:
        proc = subprocess.Popen(args, cwd=folder, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, **_detached())
    if os.name != "nt":
        child = psutil.Process(proc.pid)
        if child.nice() < NICE_BELOW_NORMAL:
            child.nice(NICE_BELOW_NORMAL)
    return proc


def tail(folder, n=LOG_TAIL):
    try:
        with open(os.path.join(folder, CONSOLE_LOG), encoding="utf-8", errors="replace") as f:
            return "".join(f.readlines()[-n:]).strip()
    except OSError:
        return ""


def read_record(cache):
    return read_json(os.path.join(cache, RECORD_FILE))


def ident(record):
    return procs.ProcessId(record["pid"], record["create_time"], MARKER + record["instance"])


class Client:
    """Line-delimited JSON over 127.0.0.1; every request carries the token."""

    def __init__(self, port, token, timeout=REQUEST_TIMEOUT):
        self.sock = socket.create_connection((LOCALHOST, port), timeout=timeout)
        self.reader = self.sock.makefile("rb")
        self.token = token
        self.lock = threading.Lock()

    def send(self, request):
        with self.lock:
            self.sock.sendall((json.dumps(request) + "\n").encode())
            line = self.reader.readline()
        if not line:
            raise GeneratorError("the generator closed the connection")
        return json.loads(line)

    def call(self, op, **request):
        reply = self.send({"op": op, "token": self.token, **request})
        if not reply.get("ok"):
            raise GeneratorError(f"{op}: {reply.get('error', 'no error given')}")
        if "data" in reply:
            reply["data"] = base64.b64decode(reply["data"])
        return reply

    def close(self):
        self.reader.close()
        self.sock.close()


class Generator:
    """A running generator: its record, its process when this tool started it, and a client."""

    def __init__(self, cache, record, proc=None):
        self.cache, self.record, self.proc = cache, record, proc
        self.client = None
        self._beat = threading.Event()

    def new_client(self, timeout=REQUEST_TIMEOUT):
        with open(os.path.join(self.record["folder"], TOKEN_FILE), encoding="utf-8") as f:
            return Client(self.record["port"], f.read().strip(), timeout)

    def connect(self, timeout=REQUEST_TIMEOUT):
        self.client = self.new_client(timeout)
        return self

    def call(self, op, **request):
        try:
            return self.client.call(op, **request)
        except OSError as e:
            log = os.path.join(self.record["folder"], CONSOLE_LOG)
            raise GeneratorError(f"{op}: the generator stopped answering ({e}); see {log}") from e

    def start_heartbeat(self, every=HEARTBEAT_SECONDS):
        threading.Thread(target=self._heartbeat, args=(every,), daemon=True, name="generator-heartbeat").start()

    def _heartbeat(self, every):
        while not self._beat.wait(every):
            try:
                self.call("status")
            except (OSError, GeneratorError):
                return

    def alive(self):
        return procs.is_alive(ident(self.record))

    def wait(self, timeout):
        """(exited, exit code or None when the code is not known)."""
        if self.proc is not None:
            try:
                return True, self.proc.wait(timeout)
            except subprocess.TimeoutExpired:
                return False, None
        if not self.alive():
            return True, None
        try:
            return True, psutil.Process(self.record["pid"]).wait(timeout)
        except psutil.TimeoutExpired:
            return False, None
        except psutil.NoSuchProcess:
            return True, None

    def stop(self, timeout=STOP_TIMEOUT, emit=_quiet):
        """Ask by token; kill only a process whose pid, create time and command-line marker are this generator's."""
        self._beat.set()
        started = time.monotonic()
        try:
            asker = self.new_client(STOP_ASK_TIMEOUT)
            asker.call("stop")
            asker.close()
        except (OSError, GeneratorError, ValueError) as e:
            emit({"step": "stop", "event": "no answer", "error": str(e)})
        exited, code = self.wait(max(timeout - (time.monotonic() - started), 0))
        if not exited and self.alive():
            emit({"step": "stop", "event": "kill", "pid": self.record["pid"]})
            psutil.Process(self.record["pid"]).kill()
            exited, code = self.wait(timeout)
        if exited and (read_record(self.cache) or {}).get("instance") == self.record["instance"]:
            os.remove(os.path.join(self.cache, RECORD_FILE))
        return code


def running(cache):
    """The generator recorded in the cache if that exact process still runs, else None."""
    record = read_record(cache)
    return Generator(cache, record) if record and procs.is_alive(ident(record)) else None


def stop_running(cache, emit=_quiet):
    """(whether a generator of this tool was running, its exit code when known)."""
    gen = running(cache)
    if gen is None:
        if os.path.exists(os.path.join(cache, RECORD_FILE)):
            os.remove(os.path.join(cache, RECORD_FILE))
        return False, None
    return True, gen.stop(emit=emit)


def _wait_port(proc, folder, emit, since):
    """The probe's port; the waiting events count from since, the start of the whole boot, as its ready event does."""
    path, started = os.path.join(folder, PORT_FILE), time.monotonic()
    next_event = started + BOOT_EVENT_SECONDS
    while time.monotonic() - started < BOOT_TIMEOUT:
        if os.path.exists(path):
            return int(open(path, encoding="ascii").read().strip())
        if proc.poll() is not None:
            raise Refusal(
                f"the generator stopped during its boot (exit code {proc.returncode}): {tail(folder)}",
                f"See {os.path.join(folder, CONSOLE_LOG)}",
            )
        if time.monotonic() >= next_event:
            emit({"step": "boot", "event": "waiting", "seconds": round(time.monotonic() - since)})
            next_event += BOOT_EVENT_SECONDS
        time.sleep(POLL)
    proc.kill()
    raise Refusal(f"the generator did not start within {BOOT_TIMEOUT // 60} minutes", f"See {os.path.join(folder, CONSOLE_LOG)}")


def _verify(gen, ctx, boot):
    s = gen.call("status")
    want = {"instance": gen.record["instance"], "kind": boot.kind, "rwg2": ctx.entry.get("rwg2") or s["rwg2"]}
    got = {k: s.get(k) for k in want}
    if got != want:
        gen.stop()
        raise Refusal(
            f"the generator that answered is not the one the tool just started (details for a report: got {got}, expected {want})", AGAIN
        )
    return s


def start(ctx, boot):
    """Boot one generator (stopping any other first) and return it connected and checked; its ready event counts the whole boot,
    the pack check and the folder's preparation included, as the preview's summary does."""
    started = time.monotonic()
    folder = pack_dir(ctx.cache, ctx.release)
    _eula(folder, ctx.accept_eula)
    check_pack(folder, ctx.entry, ctx.cache)
    stop_running(ctx.cache, ctx.emit)
    require_ram(boot.heap)
    prepare(folder, boot, ctx.accept_eula)
    instance = uuid.uuid4().hex
    proc = spawn(jvm_args(ctx.java.path, boot, instance), folder)
    record = {"pid": proc.pid, "create_time": psutil.Process(proc.pid).create_time(), "instance": instance, "port": None}
    record |= {"kind": boot.kind, "release": ctx.release, "folder": folder, "world": boot.world, "heap": boot.heap, "java": ctx.java.path}
    write_json(os.path.join(ctx.cache, RECORD_FILE), record)
    record["port"] = _wait_port(proc, folder, ctx.emit, started)
    write_json(os.path.join(ctx.cache, RECORD_FILE), record)
    gen = Generator(ctx.cache, record, proc).connect()
    status = _verify(gen, ctx, boot)
    ctx.emit({"step": "boot", "event": "ready", "seconds": round(time.monotonic() - started, 1), "kind": boot.kind, "java": status["java"]})
    return gen


@contextmanager
def lock(cache, url=None):
    """One tool process drives the generator at a time; a lock whose owner is gone is taken over."""
    path = os.path.join(cache, LOCK_FILE)
    me = {"pid": os.getpid(), "create_time": psutil.Process().create_time(), "url": url and url.split("?")[0]}
    held = read_json(path)
    if held and held.get("pid") != me["pid"] and procs.is_alive(procs.ProcessId(held["pid"], held["create_time"], "")):
        if held.get("url"):
            raise Refusal(
                f"the app is already running at {held['url'].split('?')[0]}",
                "Use the app that is open (rwg2-migrator ui shows it again), or close it first",
            )
        raise Refusal(f"another rwg2-migrator (pid {held['pid']}) is using the generator", "Wait until it finishes, or close it first")
    write_json(path, me)
    try:
        yield
    finally:
        if read_json(path) == me:
            os.remove(path)


def ring_outside(x0, size):
    edge = np.arange(x0 - 1, x0 + size + 1, dtype=np.int64)
    xs, zs = np.meshgrid(edge, edge)
    on = np.isin(xs, edge[[0, -1]]) | np.isin(zs, edge[[0, -1]])
    return np.column_stack([xs[on], zs[on]])


def offset_grid(biome, landmass):
    b = range(*biome)
    return np.array([(1, bx, bz, lx, -lx) for bx in b for bz in b for lx in range(*landmass)], np.int64)


def selftest():
    spec = json.loads(data(SELFTEST).read_text())
    return spec, ring_outside(**spec["ring"]), offset_grid(**spec["configs"])


def digest(h, meta, ocean, names):
    d = hashlib.sha256()
    for arr in (h.astype(">f4"), meta.astype("u1"), ocean.astype("u1")):
        d.update(np.ascontiguousarray(arr).tobytes())
    d.update("\n".join(names).encode())
    return d.hexdigest()[:DIGEST_CHARS]


def _named(reply):
    recs = np.frombuffer(reply["data"], RECORD)
    return recs, np.array(reply["palette"], object)[recs["palette"]]


def sample_digest(gen, seed, configs, points):
    batch = max(1, BATCH_RECORDS // len(points))
    parts = []
    for i in range(0, len(configs), batch):
        rows = configs[i : i + batch]
        r = gen.call("sample", seed=str(seed), continental=bool(rows[0, 0]), configs=rows[:, 1:].tolist(), points=points.tolist())
        parts.append(_named(r))
    recs = np.concatenate([p[0] for p in parts])
    return digest(recs["h"], recs["meta"], recs["ocean"], np.concatenate([p[1] for p in parts]).tolist())


def render_digest(gen, seed, spec):
    r = gen.call("render", seed=str(seed), continental=True, **spec)
    recs, names = _named(r)
    return digest(recs["h"], recs["meta"], np.zeros(0, "u1"), names.tolist())


def check(gen, entry, emit=_quiet):
    """The neutral self-test: ring points x configs plus one 2 km render, against the release's pinned digests."""
    spec, points, configs = selftest()
    started = time.monotonic()
    got = {"sample": sample_digest(gen, spec["seed"], configs, points), "render": render_digest(gen, spec["seed"], spec["render"])}
    want = {k: entry.get("probe_digests", {}).get(k) for k in got}
    emit({"step": "check", "event": "done", "got": got, "want": want, "seconds": round(time.monotonic() - started, 1)})
    if want != got and all(want.values()):
        raise Refusal("the generator's self-test gave different results than this tool expects", AGAIN)
    return got


def boot_and_check(ctx, heap):
    """Boot on the pack's own config (the pinned digests are for its settings), check, stop."""
    gen = start(ctx, Boot(heap=heap))
    try:
        gen.start_heartbeat()
        digests = check(gen, ctx.entry, ctx.emit)
    finally:
        code = gen.stop(emit=ctx.emit)
    return {"release": ctx.release, "java": ctx.java.path, "check": digests, "exit_code": code}


class PeakMemory:
    """The largest working set (peak_wset on Windows, else RSS) a process reached while watched."""

    def __init__(self, pid):
        self.proc, self.peak, self._done = psutil.Process(pid), 0, threading.Event()
        threading.Thread(target=self._run, daemon=True, name="generator-memory").start()

    def _run(self):
        while not self._done.wait(MEMORY_POLL):
            try:
                info = self.proc.memory_info()
            except psutil.Error:
                return
            self.peak = max(self.peak, getattr(info, "peak_wset", 0), info.rss)

    def stop(self):
        self._done.set()


def generate(gen, offsets, cells, emit=_quiet):
    """Load or generate every chunk in cells, row by row, in batches; each reply is a progress point."""
    order = sorted(cells, key=lambda c: (c[1], c[0]))
    totals = Counter()
    for i in range(0, len(order), GEN_BATCH):
        r = gen.call("generate", offsets=[int(v) for v in offsets], chunks=[list(map(int, c)) for c in order[i : i + GEN_BATCH]])
        totals.update({k: r[k] for k in ("existed", "generated", "populatedByProbe")})
        emit({"step": "generate", "event": "progress", "done": min(i + GEN_BATCH, len(order)), "total": len(order)} | dict(totals))
    return dict(totals)


def save_settings(cache, settings):
    write_json(os.path.join(cache, SETTINGS_FILE), settings)


def settings(cache):
    return read_json(os.path.join(cache, SETTINGS_FILE))
