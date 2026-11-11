"""Jobs: one folder each, with job.json, an append-only events.jsonl, one sub-folder per step with its output manifest."""

import ctypes
import hashlib
import json
import os
import secrets
import signal
import time
from contextlib import contextmanager, suppress

import psutil

from . import Refusal, procs
from .region import fsync_dir, write_atomic

APP = "rwg2-migrator"
QUEUED, RUNNING, DONE, FAILED, CANCELLED, INTERRUPTED, SUPERSEDED = (
    "queued",
    "running",
    "done",
    "failed",
    "cancelled",
    "interrupted",
    "superseded",
)
RESUMABLE = (INTERRUPTED, FAILED, CANCELLED)
JOB_FILE, EVENTS_FILE, CANCEL_FILE, MANIFEST_FILE = "job.json", "events.jsonl", "cancel", "manifest.json"
ID_BYTES = 3
COMMENTS = (b"#", b";")
ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
KILL_ENV = "RWG2_KILL_AT"
LISTED = 8
LOCK_TRIES = 3
_passed = {}


def checkpoint(step):
    """A point inside a step where the power-cut tests kill the tool: RWG2_KILL_AT=step:n kills it at the n-th point of that step."""
    spec = os.environ.get(KILL_ENV, "")
    name, _, n = spec.rpartition(":")
    if name != step:
        return
    _passed[step] = _passed.get(step, 0) + 1
    if str(_passed[step]) == n:
        os.kill(os.getpid(), getattr(signal, "SIGKILL", signal.SIGTERM))


def data_root(env=None):
    env = os.environ if env is None else env
    if env.get("RWG2_DATA"):
        return env["RWG2_DATA"]
    if os.name == "nt":
        return os.path.join(env.get("LOCALAPPDATA", os.path.expanduser("~")), APP, "data")
    mac = os.path.expanduser("~/Library/Application Support")
    if os.path.isdir(mac):
        return os.path.join(mac, APP)
    return os.path.join(env.get("XDG_DATA_HOME", os.path.expanduser("~/.local/share")), APP)


def read_json(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def write_json(path, value):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    write_atomic(path, (json.dumps(value, indent=1) + "\n").encode())


def _parsed(line):
    try:
        return json.loads(line)
    except ValueError:
        return None


def read_lines(path):
    """The JSON values of a JSON-lines file, skipping lines that do not parse (a torn last line)."""
    try:
        with open(path, encoding="utf-8") as f:
            return [v for v in map(_parsed, f) if v is not None]
    except FileNotFoundError:
        return []


def _torn(f):
    if f.seek(0, os.SEEK_END) == 0:
        return False
    f.seek(-1, os.SEEK_END)
    return f.read(1) != b"\n"


def append_line(path, value):
    """Append value as one fsynced JSON line; a line torn by a crash is ended first, so it never swallows this one."""
    with open(path, "ab+") as f:
        f.write((b"\n" if _torn(f) else b"") + json.dumps(value).encode() + b"\n")
        f.flush()
        os.fsync(f.fileno())


def count(n, word):
    """n and word, with the word in the plural unless n is 1: '1 chunk', '2 chunks'."""
    return f"{n:,} {word}" + ("" if n == 1 else "s")


def listed(texts, n=LISTED):
    """The first n texts, then how many more there are."""
    texts = list(texts)
    return ", ".join(texts[:n]) + (f" and {len(texts) - n} more" if len(texts) > n else "")


def listed_cells(cells, n=LISTED):
    return listed((f"{x},{z}" for x, z in sorted(cells)), n)


def file_hash(path):
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def settings_hash(path):
    """sha256 of a file's lines without blanks and comments: a server start rewrites the timestamp and listing comments of some
    config files, which must not count as a change."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith(COMMENTS):
                h.update(line + b"\n")
    return h.hexdigest()


def tree_hashes(folder):
    """{relative path: settings_hash} of every file under folder, sorted; None without a folder."""
    if not folder or not os.path.isdir(folder):
        return None
    files = sorted(os.path.relpath(os.path.join(d, f), folder).replace(os.sep, "/") for d, _, fs in os.walk(folder) for f in fs)
    return {f: settings_hash(os.path.join(folder, f)) for f in files}


def tree_key(folder):
    """A hash of the settings under folder (each file's relative path and settings_hash); None without a folder."""
    hashes = tree_hashes(folder)
    return None if hashes is None else key_of(list(hashes.items()))


def config_change(path, then, what):
    """How the config folder in use (path, or None) differs from the one `what` ran on (its tree_key then, False when it recorded
    none), in words; None when it is the same, no config folder on both sides included."""
    if then is False:
        return f"{what} did not record which config folder it ran on"
    now = tree_key(path)
    if now == then:
        return None
    if then is None:
        return f"{what} ran on GTNH's default RWG2 settings (no config folder), but the config folder {path} is in use now"
    if now is None:
        return f"{what} ran on a config folder, but none is in use now"
    return f"the config folder {path} changed since {what} ran"


def key_of(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()[:16]


class Job:
    def __init__(self, root, job_id):
        self.root, self.id = root, job_id
        self.folder = os.path.join(root, "jobs", job_id)
        self.listeners = []

    @property
    def data(self):
        return read_json(os.path.join(self.folder, JOB_FILE)) or {}

    def update(self, **changes):
        write_json(os.path.join(self.folder, JOB_FILE), self.data | changes | {"updated": time.time()})

    def state(self):
        return self.data.get("state")

    def emit(self, event):
        event = {"t": round(time.time(), 3), **event}
        append_line(os.path.join(self.folder, EVENTS_FILE), event)
        for listen in self.listeners:
            listen(event)

    def events(self):
        """Every event so far; a line torn by a crash is skipped."""
        return read_lines(os.path.join(self.folder, EVENTS_FILE))

    def step_dir(self, name):
        path = os.path.join(self.folder, name)
        os.makedirs(path, exist_ok=True)
        return path

    def cancel_requested(self):
        return os.path.exists(os.path.join(self.folder, CANCEL_FILE))

    def request_cancel(self):
        write_atomic(os.path.join(self.folder, CANCEL_FILE), b"cancel\n")

    def finish_step(self, name, files, inputs):
        """Record what a step wrote (sha256 and size of each file) and the key of its inputs."""
        folder = self.step_dir(name)
        entries = {f: {"sha256": file_hash(os.path.join(folder, f)), "size": os.path.getsize(os.path.join(folder, f))} for f in files}
        write_json(os.path.join(folder, MANIFEST_FILE), {"inputs": inputs, "files": entries, "finished": time.time()})

    def step_valid(self, name, inputs):
        """True when the step finished with these inputs and every file it wrote is still as written."""
        folder = os.path.join(self.folder, name)
        m = read_json(os.path.join(folder, MANIFEST_FILE))
        if not m or key_of(m.get("inputs")) != key_of(inputs):
            return False
        for f, e in m["files"].items():
            path = os.path.join(folder, f)
            if not os.path.isfile(path) or os.path.getsize(path) != e["size"] or file_hash(path) != e["sha256"]:
                return False
        return True

    def drop_step(self, name):
        path = os.path.join(self.folder, name, MANIFEST_FILE)
        if os.path.exists(path):
            os.remove(path)


def create(root, kind, args, world):
    job_id = time.strftime("%Y%m%d-%H%M%S") + "-" + secrets.token_hex(ID_BYTES)
    job = Job(root, job_id)
    os.makedirs(job.folder)
    fsync_dir(os.path.dirname(job.folder))
    record = {"id": job_id, "kind": kind, "args": args, "world": world, "state": QUEUED, "created": time.time()}
    write_json(os.path.join(job.folder, JOB_FILE), record)
    return job


def all_jobs(root):
    folder = os.path.join(root, "jobs")
    names = sorted(os.listdir(folder)) if os.path.isdir(folder) else []
    return [Job(root, n) for n in names if os.path.isfile(os.path.join(folder, n, JOB_FILE))]


def find(root, ref):
    """A job by id, unique id prefix, or 'latest' (the newest job)."""
    if not ref:
        raise Refusal("no job was named", "Name one; rwg2-migrator jobs list shows them")
    jobs = all_jobs(root)
    if ref == "latest" and jobs:
        return jobs[-1]
    hits = [j for j in jobs if j.id == ref] or [j for j in jobs if j.id.startswith(ref)]
    if len(hits) != 1:
        found = "no job" if not hits else f"{len(hits)} jobs"
        raise Refusal(f"{found} matches {ref!r} in {os.path.join(root, 'jobs')}", "List them with: rwg2-migrator jobs list")
    return hits[0]


def _lock_path(root, world):
    return os.path.join(root, "locks", hashlib.sha1(os.path.normcase(os.path.realpath(world)).encode()).hexdigest()[:16] + ".json")


def _owner_alive(held):
    return bool(held) and procs.is_alive(procs.ProcessId(held["pid"], held["create_time"], ""))


def refresh(root):
    """A running job whose lock owner is gone (pid gone, or reused by a process with another create time) becomes interrupted."""
    changed = []
    for job in all_jobs(root):
        d = job.data
        if d.get("state") != RUNNING:
            continue
        held = read_json(_lock_path(root, d["world"]))
        if not (_owner_alive(held) and held.get("job") == job.id):
            job.update(state=INTERRUPTED)
            changed.append(job.id)
    return changed


@contextmanager
def lock(root, world, job_id):
    """One job per world at a time; the lock holds pid + create time, so a pid reused after a power cut does not hold it."""
    path = _lock_path(root, world)
    me = {"pid": os.getpid(), "create_time": psutil.Process().create_time(), "job": job_id}
    for _ in range(LOCK_TRIES):
        if _claim(path, me):
            break
        held = read_json(path)
        if held and held.get("pid") != os.getpid() and _owner_alive(held):
            raise Refusal(
                f"job {held['job']} (pid {held['pid']}) is already working on {world}",
                f"Wait for it, or cancel it: rwg2-migrator jobs cancel {held['job']}",
            )
        if read_json(path) == held:
            with suppress(FileNotFoundError):
                os.remove(path)
    else:
        raise Refusal(f"another job is taking the lock of {world} right now", "Wait a moment, then run this again")
    try:
        yield
    finally:
        if read_json(path) == me:
            os.remove(path)


def _claim(path, me):
    """Create the lock file holding me only when there is none: a hard link of a complete file is atomic. A drive without hard
    links falls back to a plain replace."""
    tmp = f"{path}.{os.getpid()}.tmp"
    write_json(tmp, me)
    try:
        os.link(tmp, path)
        return True
    except FileExistsError:
        return False
    except OSError:
        if os.path.exists(path):
            return False
        os.replace(tmp, path)
        return True
    finally:
        with suppress(FileNotFoundError):
            os.remove(tmp)


@contextmanager
def keep_awake():
    """While a job runs on Windows, the idle-sleep timer must not suspend it."""
    kernel = ctypes.windll.kernel32 if os.name == "nt" else None
    if kernel:
        kernel.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
    try:
        yield
    finally:
        if kernel:
            kernel.SetThreadExecutionState(ES_CONTINUOUS)


def run(job, work):
    """Run work(job) under the world lock, keeping the machine awake; the job ends done, cancelled, interrupted or failed."""
    refresh(job.root)
    if job.state() in (RUNNING, DONE, SUPERSEDED):
        raise Refusal(f"job {job.id} is {job.state()}", "Start a new job, or resume one that was interrupted")
    cancel = os.path.join(job.folder, CANCEL_FILE)
    if os.path.exists(cancel):
        os.remove(cancel)
    with lock(job.root, job.data["world"], job.id), keep_awake():
        job.update(state=RUNNING, pid=os.getpid())
        job.emit({"step": "job", "event": "start", "pid": os.getpid()})
        try:
            result = work(job)
        except KeyboardInterrupt:
            job.update(state=INTERRUPTED)
            job.emit({"step": "job", "event": "interrupted"})
            raise
        except BaseException as e:
            job.update(state=FAILED, error=str(e))
            job.emit({"step": "job", "event": "failed", "error": str(e)})
            raise
        state = CANCELLED if job.cancel_requested() else DONE
        job.update(state=state)
        job.emit({"step": "job", "event": state})
        return result
