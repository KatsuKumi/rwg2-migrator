import os
import subprocess
from pathlib import Path

import pytest

HERE = Path(__file__).parent
TICK_QUEUE = HERE.parent / "probe-mod" / "src" / "main" / "java" / "rwg2probe" / "TickQueue.java"
JDK = os.environ.get("RWG2_JDK")
LIMIT_S = 60
pytestmark = pytest.mark.skipif(not JDK, reason="set RWG2_JDK to a JDK home to run the probe's tick queue against FML stand-ins")


def test_the_probe_tick_queue_fails_a_world_op_once_the_server_stops_ticking(tmp_path):
    sources = [*sorted((HERE / "java").rglob("*.java")), TICK_QUEUE]
    subprocess.run([Path(JDK, "bin", "javac"), "-d", tmp_path, *sources], check=True, timeout=LIMIT_S)
    run = subprocess.run([Path(JDK, "bin", "java"), "-cp", tmp_path, "rwg2probe.TickQueueCheck"], capture_output=True, text=True,
                         timeout=LIMIT_S)  # fmt: skip
    assert run.returncode == 0 and run.stdout.split() == ["ticking", "stalled", "stepping", "failing", "ok"], run.stderr
