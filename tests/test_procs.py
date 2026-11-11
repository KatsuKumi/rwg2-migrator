import os
import sys
import time

import psutil
import pytest
from conftest import Unreadable, age

from rwg2migrator import Refusal, procs


def test_running_child_in_server_folder_is_in_use(server, java_child):
    usage = procs.in_use(server, os.path.join(server, "World"))
    assert usage.state == procs.RUNNING and usage.pids == (java_child.pid,)
    with pytest.raises(Refusal, match=f"pid {java_child.pid}") as e:
        procs.require_stopped(usage, confirmed=True)
    assert "Stop the server" in e.value.action


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="/proc is Linux only")
def test_a_server_seen_through_another_path_is_in_use(server, java_child, monkeypatch):
    monkeypatch.setattr(procs.psutil.Process, "cwd", lambda self: "/data")
    assert procs.in_use(server, os.path.join(server, "World")).state == procs.RUNNING


def test_child_elsewhere_does_not_count(server, java_child, tmp_path):
    other = tmp_path / "other"
    other.mkdir()
    (other / "World").mkdir()
    (other / "World" / "level.dat").write_bytes(b"")
    age(str(other / "World"))
    assert procs.in_use(str(other), str(other / "World")).state == procs.STOPPED


def test_unreadable_java_is_unknown_and_needs_confirmation(server, monkeypatch):
    monkeypatch.setattr(procs.psutil, "process_iter", lambda attrs: iter([Unreadable()]))
    usage = procs.in_use(server, os.path.join(server, "World"))
    assert usage.state == procs.UNKNOWN and usage.pids == (4242,)
    with pytest.raises(Refusal, match="the world may be in use: ") as e:
        procs.require_stopped(usage, confirmed=False)
    assert "--confirm-stopped" in e.value.action
    procs.require_stopped(usage, confirmed=True)


def test_recent_level_dat_means_running(server):
    w = os.path.join(server, "World")
    age(w, 5)
    assert procs.in_use(server, w).state == procs.RUNNING
    assert procs.in_use(server, w, now=time.time() + procs.RECENT_SAVE_SECONDS).state == procs.STOPPED


def test_process_identity(java_child):
    ident = procs.ProcessId(java_child.pid, psutil.Process(java_child.pid).create_time(), "-Dmarker=abc")
    assert procs.is_alive(ident)
    assert not procs.is_alive(procs.ProcessId(ident.pid, ident.create_time + 5, ident.marker))
    assert not procs.is_alive(procs.ProcessId(ident.pid, ident.create_time, "-Drwg2probe.instance=other"))
    java_child.kill()
    java_child.wait()
    assert not procs.is_alive(ident)


def test_remote_is_unknown():
    assert procs.remote().state == procs.UNKNOWN
