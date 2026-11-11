import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dev"))

import pc_acceptance  # noqa: E402


def test_watcher_events_outside_the_skipped_folders_are_kept():
    lines = [
        "10:00:00.000 WATCHING 12 folders, all but D:\\work\\c",
        "10:00:01.000 Created D:\\work\\c\\packs\\x",
        "10:00:01.000 Changed D:\\work\\c",
        "10:00:01.100 Created D:\\work\\cache-b\\x",
        "10:00:02.000 Created C:\\Users\\me\\AppData\\Local\\Temp\\socket_1",
        "10:00:02.000 Created C:\\Users\\me\\AppData\\Local\\Temp\\socket_1",
        "10:00:03.000 OVERFLOW Too many changes at once",
        "10:00:04.000 END",
    ]
    events, problems = pc_acceptance.outside(lines, ["D:\\WORK\\c\\"])
    assert events == {"Created D:\\work\\cache-b\\x": 1, "Created C:\\Users\\me\\AppData\\Local\\Temp\\socket_1": 2}
    assert problems == ["10:00:03.000 OVERFLOW Too many changes at once"]
