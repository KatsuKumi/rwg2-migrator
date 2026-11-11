import os
import sys
import time
from dataclasses import dataclass

import psutil

from . import Refusal

JAVA_NAMES = {"java", "javaw", "java.exe", "javaw.exe"}
RECENT_SAVE_SECONDS = 120
STOPPED, RUNNING, UNKNOWN = "stopped", "running", "unknown"
CREATE_TIME_TOLERANCE = 0.01


@dataclass(frozen=True)
class ProcessId:
    pid: int
    create_time: float
    marker: str


def is_alive(ident):
    """True only when the pid exists with the recorded create time and its command line holds the marker."""
    try:
        p = psutil.Process(ident.pid)
        if abs(p.create_time() - ident.create_time) > CREATE_TIME_TOLERANCE:
            return False
        return ident.marker in " ".join(p.cmdline())
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
        return False


@dataclass(frozen=True)
class Usage:
    state: str
    detail: str
    pids: tuple = ()


def inside(path, root):
    path, root = os.path.normcase(os.path.realpath(path)), os.path.normcase(os.path.realpath(root))
    return path == root or path.startswith(root.rstrip(os.sep) + os.sep)


def _same_dir(pid, root):
    """On Linux, whether a process works in root seen through another path, such as a container's mount of it."""
    try:
        return sys.platform.startswith("linux") and os.path.samefile(f"/proc/{pid}/cwd", root)
    except OSError:
        return False


def java_processes():
    for p in psutil.process_iter(["name"]):
        if (p.info.get("name") or "").lower() in JAVA_NAMES:
            yield p


def level_age(world, now=None):
    return (time.time() if now is None else now) - os.path.getmtime(os.path.join(world, "level.dat"))


def in_use(root, world, now=None):
    """Whether a game or server runs the world in root: a java process working in root, or level.dat saved moments ago.
    With world None only the processes are looked at."""
    unreadable = []
    for p in java_processes():
        try:
            cwd = p.cwd()
        except psutil.AccessDenied:
            unreadable.append(p.pid)
            continue
        except (psutil.NoSuchProcess, psutil.ZombieProcess):
            continue
        if cwd and (inside(cwd, root) or _same_dir(p.pid, root)):
            return Usage(RUNNING, f"java (pid {p.pid}) is running in {cwd}", (p.pid,))
    age = RECENT_SAVE_SECONDS if world is None else level_age(world, now)
    if age < RECENT_SAVE_SECONDS:
        return Usage(RUNNING, f"level.dat was saved {max(age, 0):.0f} s ago, so the world is probably still running")
    if unreadable:
        pids = ", ".join(map(str, unreadable))
        return Usage(UNKNOWN, f"java (pid {pids}) is running, but its folder cannot be read (another user or a service)", tuple(unreadable))
    return Usage(STOPPED, f"no java process is running in {root}")


def remote():
    return Usage(UNKNOWN, "this world was copied from another machine, so the tool cannot see whether its server is stopped")


def require_stopped(usage, confirmed):
    if usage.state == RUNNING:
        raise Refusal(f"the world is in use: {usage.detail}", "Stop the server or close the game, then run this again")
    if usage.state == UNKNOWN and not confirmed:
        raise Refusal(
            f"the world may be in use: {usage.detail}",
            "If your server is stopped, run this again with --confirm-stopped",
        )
