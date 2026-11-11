import os

from e2e_fixture import moved


def test_moved_points_windows_paths_of_another_machine_at_folders_here():
    moves = {"D:\\gtnh\\server": "/srv/fixture", "C:\\Users\\a\\data": "/tmp/data"}
    record = {"world": "D:\\gtnh\\server\\World", "args": {"staging": "D:\\gtnh\\server\\rwg2-work\\j1\\staging", "pick": 2},
              "cells": [[1, 2]], "other": "D:\\gtnh\\server2\\World", "data": "C:\\Users\\a\\data"}  # fmt: skip
    out = moved(record, moves)
    assert out["world"] == os.path.join("/srv/fixture", "World")
    assert out["args"] == {"staging": os.path.join("/srv/fixture", "rwg2-work", "j1", "staging"), "pick": 2}
    assert out["cells"] == [[1, 2]] and out["other"] == record["other"] and out["data"] == "/tmp/data"
