import dataclasses
import http.client
import json
import logging
import os
import re
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from urllib.parse import quote

import psutil
import pytest
from conftest import compound, make_world, stop_tree, string, tag, terrain, write_nbt
from test_cli_search import _break_chunk
from test_scan import grave_te
from test_search import SMALL

from rwg2migrator import ISSUES, Refusal, apply, area, cli, generator, jobs, nbt, search, world
from rwg2migrator.web import server, tiles

STATIC = Path(server.__file__).parent / "static"
HERE = Path(__file__).parent
BANNED = re.compile(r"frontier|landmass|rc1|harvest", re.I)
WAIT = 120


def fake_world(root):
    specs = {(x, z): {"ids": terrain(top=64 + (x + z) % 5)} for x in range(16) for z in range(16)}
    for c in ((1, 1), (14, 2), (9, 13)):
        specs[c] = specs[c] | {"tes": ("Chest", "GT_TileEntity_Ores")}
    specs[(14, 2)]["tes"] += (("BaseMetaTileEntity", {"mOwnerName": string("Alice")}),)
    specs[(3, 14)] = specs[(3, 14)] | {"tes": (grave_te("Bob", 5),)}
    for c in ((6, 6), (7, 6), (6, 7)):
        specs[c] = specs[c] | {"inhabited": 2 * area.SUGGEST_TICKS}
    claims = {"alpha": [(7, 7), (8, 7), (7, 8)], "beta": [(2, 12)]}
    players = {"u1": ("Alice", 24, 24), "u2": ("Bob", 120, 120)}
    return make_world(str(root), claims=claims, players=players, chunk_specs=specs)


def tile_path(url):
    """A layer's tile URL as the page asks for it: the layer folder kept, the token sent as a header instead."""
    return url.split("&t=")[0]


class Live:
    """A UI served from a thread of this process; shows is the world a page would say it shows (the X-World header)."""

    def __init__(self, data, cache, replay=None):
        self.ui, self.shows = server.UI(str(data), str(cache), replay=replay), None
        threading.Thread(target=self.ui.server.serve_forever, daemon=True).start()

    def request(self, path, body=None, host=None, token=True):
        c = http.client.HTTPConnection(server.LOCALHOST, self.ui.port, timeout=WAIT)
        headers = {"Host": host or f"{server.LOCALHOST}:{self.ui.port}", "Content-Type": "application/json"}
        if token:
            headers["X-Token"] = self.ui.token
        if self.shows:
            headers["X-World"] = quote(str(self.shows))
        c.request("POST" if body is not None else "GET", path, json.dumps(body) if body is not None else None, headers)
        r = c.getresponse()
        return r.status, r.read()

    def api(self, path, body=None):
        status, data = self.request("/api/" + path, body)
        assert status == 200, data
        return json.loads(data)

    def open(self, world):
        self.api("world", {"path": str(world)})
        deadline = time.monotonic() + WAIT
        while self.ui.opened.sc is None and time.monotonic() < deadline:
            assert self.ui.opened.error is None, self.ui.opened.error
            time.sleep(0.05)
        return self.ui.opened

    def close(self):
        self.ui.server.shutdown()
        self.ui.server.server_close()


@pytest.fixture
def live(tmp_path):
    lv = Live(tmp_path / "data", tmp_path / "cache")
    yield lv
    lv.close()


def test_token_and_host_are_required_and_only_localhost_listens(live):
    assert live.ui.server.server_address[0] == "127.0.0.1"
    assert live.request("/api/state", token=False)[0] == 403
    assert live.request(f"/?t={live.ui.token[:-1]}x", token=False)[0] == 403
    assert live.request("/api/state", host="evil.example:80")[0] == 403
    assert live.request("/api/state", host=f"rebound.example:{live.ui.port}")[0] == 403
    assert live.request("/static/app.js", token=False)[0] == 403
    status, page = live.request(f"/?t={live.ui.token}", token=False)
    assert status == 200 and f'content="{live.ui.token}"' in page.decode()
    assert live.request(f"/static/app.js?t={live.ui.token}", token=False, host=f"localhost:{live.ui.port}")[0] == 200
    assert live.request("/static/../server.py")[0] == 404
    for windows_path in ("/static/..\\server.py", "/static/..\\..\\..\\secret.txt", "/static/C:/secret.txt"):
        assert live.request(windows_path)[0] == 404


def test_a_request_from_another_web_page_is_refused_even_with_the_token(live):
    def status(origin):
        c = http.client.HTTPConnection(server.LOCALHOST, live.ui.port, timeout=WAIT)
        c.request("GET", "/api/state", headers={"Host": f"{server.LOCALHOST}:{live.ui.port}", "X-Token": live.ui.token, "Origin": origin})
        return c.getresponse().status

    assert status("https://evil.example") == 403 and status(f"http://{server.LOCALHOST}:{live.ui.port}") == 200


def test_the_token_never_reaches_the_log(live, caplog):
    with caplog.at_level(logging.INFO, logger=server.log.name):
        assert live.request(f"/static/app.js?t={live.ui.token}", token=False)[0] == 200
    assert "/static/app.js?t=<token>" in caplog.text and live.ui.token not in caplog.text


def test_open_folder_refuses_in_plain_words_when_nothing_can_open_it(tmp_path, monkeypatch):
    monkeypatch.setattr(server.subprocess, "Popen", lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError("xdg-open")))
    monkeypatch.setattr(server.os, "startfile", lambda p: (_ for _ in ()).throw(OSError("no handler")), raising=False)
    with pytest.raises(Refusal, match="could not open"):
        server.open_folder(str(tmp_path))


def test_area_counts_equal_the_area_command(live, tmp_path):
    w = fake_world(tmp_path / "server")
    live.open(w)
    painted = sorted(area.rect("3,3:4,5"))
    got = live.api("area", {"teams": ["alpha"], "painted": painted, "erased": [[8, 7]], "margin": 2})
    out = tmp_path / "area.json"
    argv = ["area", w, "--claims", "alpha", "--add", "3,3:4,5", "--remove", "8,7:8,7", "--margin", "2", "--out", str(out)]
    assert cli.main([*argv, "--cache", str(tmp_path / "cache")]) == 0
    want = json.loads(out.read_text())
    assert got["core"] == len(want["core"]) and got["kept"] == want["kept"]
    assert got["dropped"] == want["status"]["dropped"] and got["removed"] == want["removed"]
    assert got["unreadable"] is None
    saved = live.api("area", {"teams": ["alpha"], "painted": painted, "erased": [[8, 7]], "margin": 2, "save": True})
    stored = json.loads(Path(saved["file"]).read_text())
    assert {k: stored[k] for k in ("core", "kept", "margin", "teams", "removed")} == {
        k: want[k] for k in ("core", "kept", "margin", "teams", "removed")
    }


def test_the_area_screen_flags_kept_chunks_that_do_not_read(live, tmp_path):
    w = fake_world(tmp_path / "server")
    _break_chunk(Path(w).parent, (7, 7))
    live.open(w)
    got = live.api("area", {"teams": ["alpha"], "painted": [], "erased": [], "margin": 1})
    assert got["unreadable"].startswith("1 chunk of the area to keep does not read in the world (7,7)"), got["unreadable"]


def test_a_page_still_showing_another_world_is_refused_and_nothing_is_saved_for_the_open_one(live, tmp_path):
    a, b = fake_world(tmp_path / "A"), fake_world(tmp_path / "Мир 日本")
    body = {"teams": ["alpha"], "painted": [[3, 3]], "erased": [], "margin": 1}
    live.open(a)
    live.shows = a
    assert live.api("area", body)["core"] == 4
    live.open(b)
    for path, sent in (("area", body | {"save": True}), ("world", None), ("look", None), ("search", {}), ("probe?x=1&z=1", None)):
        status, out = live.request(f"/api/{path}", sent)
        err = json.loads(out)["error"]
        assert status == 400 and "still shows" in err["reason"] and err["action"].startswith("Reload this page"), (path, err)
    assert not live.ui.settings.get("areas") and not os.path.exists(os.path.join(live.ui.ui_folder(), "area.json"))
    live.shows = b
    assert live.api("area", body | {"save": True})["core"] == 4
    assert json.loads(Path(live.ui.settings["areas"][b]["file"]).read_text())["world"] == b


def test_the_area_screen_reads_claims_made_since_the_world_was_opened(live, tmp_path):
    w = fake_world(tmp_path / "server")
    live.open(w)
    assert live.api("world")["claims"]["alpha"] == [[7, 7], [7, 8], [8, 7]]
    cells = [{"x": tag(nbt.INT, x), "z": tag(nbt.INT, z)} for x, z in ((7, 7), (8, 7), (7, 8), (9, 9))]
    claims = {"ClaimedChunks": compound(**{"0": tag(nbt.LIST, [nbt.COMPOUND, cells])})}
    write_nbt(os.path.join(w, world.CLAIMS, "alpha.dat"), claims)
    assert live.api("world")["claims"]["alpha"] == [[7, 7], [7, 8], [8, 7], [9, 9]]
    got = live.api("area", {"teams": ["alpha"], "painted": [], "erased": [], "margin": 1})
    assert got["core"] == 4


def test_the_folder_browser_lists_folders_and_names_worlds_servers_and_zips(live, tmp_path):
    w = fake_world(tmp_path / "server")
    (tmp_path / "gtnh.zip").write_bytes(b"PK")
    (tmp_path / ".hidden").mkdir()
    (tmp_path / ".minecraft" / "saves").mkdir(parents=True)
    top = live.api("browse?path=")
    assert top["parent"] is None and top["entries"] and all(e["kind"] == "drive" for e in top["entries"])
    here = live.api(f"browse?path={tmp_path}&ext=.zip")
    kinds = {e["name"]: e["kind"] for e in here["entries"]}
    assert kinds == {".minecraft": "folder", "server": "server", "gtnh.zip": "file"} and here["parent"] == str(tmp_path.parent)
    assert live.api(f"browse?path=%22{tmp_path / 'server'}%22")["entries"][0] == {"name": "World", "path": w, "kind": "world"}
    status, data = live.request(f"/api/browse?path={tmp_path / 'nowhere'}")
    assert status == 400 and json.loads(data)["error"]["action"]


def test_copy_diagnostic_info_names_the_versions_the_world_and_the_last_step(live, tmp_path):
    live.open(fake_world(tmp_path / "server"))
    text = live.api("diag")["text"]
    assert text.startswith("rwg2-migrator ") and "OS: " in text and "RAM: " in text and "GTNH 2.9.0" in text and "Step: " in text


def test_add_these_chunks_adds_exactly_the_overlays_chunks(live, tmp_path):
    w = fake_world(tmp_path / "server")
    live.open(w)
    body = {"teams": ["alpha"], "painted": [], "erased": [], "margin": 1}
    before = live.api("area", body)
    added = []
    for layer in ("yours", "graves", "claims", "players", "spawn"):
        cells = before["cells"][layer]
        if not cells:
            continue
        added.append(layer)
        core_before = set(map(tuple, live.ui.area_of(body)[0]["core"]))
        body["painted"] = sorted({tuple(c) for c in body["painted"]} | {tuple(c) for c in cells})
        after = live.api("area", body)
        assert after["cells"][layer] == []
        assert set(map(tuple, live.ui.area_of(body)[0]["core"])) == core_before | {tuple(c) for c in cells}
        before = after
    assert added[:3] == ["yours", "graves", "claims"] and before["removed"]["yours"]["count"] == 0 and before["removed"]["graves"] == []
    assert before["removed"]["players"] == [] and before["removed"]["spawn_in_kept"]


def test_suggest_offers_the_base_part_and_far_parts_warn(live, tmp_path):
    live.open(fake_world(tmp_path / "server"))
    assert live.api("world")["suggest"] == {"main": [[6, 6], [6, 7], [7, 6]], "extra": []}
    one = live.api("area", {"teams": ["alpha"], "painted": [[6, 6]], "erased": [], "margin": 1})
    two = live.api("area", {"teams": ["alpha", "beta"], "painted": [], "erased": [], "margin": 1})
    assert (one["parts"], one["parts_warning"]) == (1, None) and two["parts"] == 2 and two["parts_warning"] == area.parts_warning(2)


def test_world_tiles_are_drawn_once_and_a_revisit_draws_nothing(live, tmp_path):
    live.open(fake_world(tmp_path / "server"))
    path = tile_path(live.api("layer?name=area")["url"])
    drawn = tiles.Layer.drawn
    c = http.client.HTTPConnection(server.LOCALHOST, live.ui.port, timeout=WAIT)
    c.request("GET", path.format(z=-1, x=0, y=0), headers={"Host": f"127.0.0.1:{live.ui.port}", "X-Token": live.ui.token})
    r = c.getresponse()
    assert r.status == 200 and r.read()[:8] == b"\x89PNG\r\n\x1a\n" and 0 <= float(r.getheader("X-Tile-Age")) < WAIT
    assert tiles.Layer.drawn == drawn + 1
    for x, y in ((0, 0), (1, 0), (0, 1), (1, 1), (3, 3)):
        assert live.request(path.format(z=-1, x=x, y=y))[0] == 200
    assert tiles.Layer.drawn == drawn + 1
    assert live.request(path.format(z=-4, x=0, y=0))[0] == 200
    assert tiles.Layer.drawn == drawn + 2 and tiles.Layer.redrawn == 0
    assert live.request(path.format(z=-5, x=0, y=0))[0] == 404


def test_the_tile_stream_names_each_block_as_soon_as_it_is_drawn(live, tmp_path):
    live.open(fake_world(tmp_path / "server"))
    url = live.api("layer?name=area")["url"]
    c = http.client.HTTPConnection(server.LOCALHOST, live.ui.port, timeout=WAIT)
    c.request("GET", "/api/tilestream", headers={"Host": f"127.0.0.1:{live.ui.port}", "X-Token": live.ui.token})
    stream = c.getresponse()
    assert stream.status == 200 and stream.getheader("Content-Type") == "text/event-stream"
    assert live.request(tile_path(url).format(z=-2, x=1, y=0))[0] == 200
    line = stream.fp.readline().decode()
    while not line.startswith("data: "):
        line = stream.fp.readline().decode()
    e = json.loads(line[len("data: ") :])
    n = tiles.WORLD_BLOCK // tiles.TILE
    assert e | {"id": 0} == {"id": 0, "v": url.split("v=")[1].split("&")[0], "z": -2, "x": 0, "y": 0, "n": n}
    c.close()


def test_every_step_the_ui_starts_is_a_command_the_cli_parses(live, tmp_path, monkeypatch):
    started = []
    monkeypatch.setattr(server.subprocess, "Popen", lambda cmd, **kw: started.append(cmd) or FakeProc())
    monkeypatch.setattr(server, "cli_prefix", lambda: ["rwg2-migrator"])
    w = fake_world(tmp_path / "server")
    live.open(w)
    live.api("area", {"teams": ["alpha"], "painted": [], "erased": [], "margin": 1, "save": True})
    job = jobs.create(live.ui.data, "search", {}, w)
    preview = jobs.create(live.ui.data, "preview", {}, w)
    for path, body in (
        ("setup", {"zip": str(tmp_path / "gtnh.zip")}),
        ("search", {"tolerance": 1}),
        ("preview", {"search": job.id, "pick": 4, "trees": True, "blend": True}),
        ("apply", {"preview": preview.id, "taint_off": True, "auto_restart_off": True, "backup_copy": str(tmp_path)}),
        ("verify", {"restore": True}),
        ("rollback", {"world": w, "auto_restart_off": True}),
        ("resume", {"job": job.id}),
        ("discard", {"job": job.id}),
        ("clean", {}),
    ):
        live.api(path, body)
        args = cli.parser().parse_args(cli.joined(started[-1][1:]))
        assert args.fn and args.cache == live.ui.cache, path
        if path == "resume":
            assert live.api("state")["runner"]["job"] == job.id
    assert len(started) == 9 and cli.joined(started[-1][1:])[:2] == ["clean", "--generator"]


def test_a_step_starting_keeps_the_maps_off_the_generator(live, tmp_path, monkeypatch):
    seen = []
    live.open(fake_world(tmp_path / "server"))
    monkeypatch.setattr(server.subprocess, "Popen", lambda cmd, **kw: FakeProc())
    monkeypatch.setattr(live.ui.maps.source, "release", lambda: seen.append(live.ui.runner.busy()))
    live.ui.runner.start("verify", ["verify", "W"])
    assert seen == [True]
    with pytest.raises(server.Refusal, match="uses the generator"):
        live.ui.runner.starting = True
        live.ui.maps.source.sampler([server.seed_of(live.ui.opened), None])


class FakeProc:
    def poll(self):
        return 0


def test_a_map_that_fails_unexpectedly_is_reported_and_the_worker_carries_on():
    asks = server.Asks()

    def broken():
        raise ValueError("operands could not be broadcast")

    done, out = asks.ask("a", broken, wait=WAIT)
    assert done and "ValueError" in out["error"]["reason"] and out["error"]["action"] == server.REPORT
    assert asks.ask("b", lambda: "tile.png", wait=WAIT) == (True, "tile.png")


def test_a_world_tile_is_the_scan_drawn_like_the_map_command(tmp_path, cache):
    from PIL import Image

    from rwg2migrator import appearance, maps, scan

    w = fake_world(tmp_path / "server")
    sc = scan.scan(w, cache, workers=1)
    ref = appearance.reference(sc, {(7, 7)}, "2.9.0")
    lut = tiles.biome_lut(sc, ref, "2.9.0")
    layer = tiles.Layer(str(tmp_path / "t"), tiles.world_draw(sc, lut))
    rgb = Image.open(layer.tile(-1, 0, 0)).convert("RGB")
    a = {"core": [(7, 7)], "kept": sorted(area.dilate({(7, 7)}, 1))}
    look, ground = maps.real_looks(maps.Scene(sc, a, "2.9.0", None, []), ref, (0, 0, 512, 512), 2)
    assert (look[:128, :128] >= 0).all() and set(map(int, look.ravel())) - {-1} <= set(lut.tolist())
    assert rgb.size == (tiles.TILE, tiles.TILE) and rgb.getpixel((5, 5)) != maps.UNGENERATED


def test_a_working_copys_map_judges_the_worlds_biomes_as_the_worlds_map_does():
    import numpy as np

    from rwg2migrator import appearance

    green, snow, sand = (appearance.LOOKS.index(k) for k in ("green", "snow", "sand"))
    world, copy = np.full(tiles.BIOME_IDS, tiles.UNSEEN, np.int8), np.full(tiles.BIOME_IDS, tiles.UNSEEN, np.int8)
    world[236], copy[236], copy[324] = green, snow, sand
    lut = tiles.merge(world, copy)
    assert (lut[236], lut[324], lut[7]) == (green, sand, tiles.UNSEEN)
    sc = type("Scan", (), {"grid": lambda self, k, box, bpp: {"ground": np.full((1, 3), 70.0), "biome": np.array([[236, 324, 7]]),
                                                              "water": np.zeros((1, 3))}[k]})()  # fmt: skip
    look, _ = tiles.real_looks(sc, lut, (0, 0, 3, 1), 1)
    assert look.tolist() == [[green, sand, tiles.UNKNOWN_LOOK]]


def run_search(data, w, a_file, cache):
    from fake_sampler import FakeSampler

    from rwg2migrator import appearance, scan

    a = search.load(a_file)
    job = jobs.create(data, "search", {"prefs": appearance.load_prefs(None), "threads": 1, "cache": cache}, w)
    Path(job.folder, "area.json").write_text(json.dumps(a))
    inp = search.Inputs(scan.scan(w, cache, workers=1), a, appearance.load_prefs(None), "2.9.0", 1234567890123, FakeSampler, SMALL)
    jobs.run(job, lambda j: search.run_job(j, inp))
    return job


def test_candidate_tiles_heat_and_stream_from_a_search(live, tmp_path, monkeypatch):
    import fake_gen

    fake_gen.use(monkeypatch)
    w = fake_world(tmp_path / "server")
    live.open(w)
    saved = live.api("area", {"teams": ["alpha"], "painted": [], "erased": [], "margin": 1, "save": True})
    job = run_search(live.ui.data, w, saved["file"], live.ui.cache)
    view = live.api(f"job?id={job.id}")
    best = view["candidates"]["candidates"][0]
    assert view["best"] and view["best"][-1]["config"] in [c["config"] for c in view["candidates"]["candidates"]]
    name = f"cand.{job.id}.{'_'.join(map(str, best['config']))}"
    path = tile_path(live.api(f"layer?name={name}")["url"])
    drawn = tiles.Layer.drawn
    first = live.request(path.format(z=-1, x=0, y=0))
    assert first[0] == 503 and json.loads(first[1])["error"]["action"]
    for x, y in ((0, 0), (1, 1)):
        status, png = until(lambda x=x, y=y: (r := live.request(path.format(z=-1, x=x, y=y)))[0] == 200 and r, "the tile")
        assert png[:4] == b"\x89PNG"
    until(lambda: tiles.Layer.drawn == drawn + 1, "the block counted")
    assert all(live.request(path.format(z=-1, x=x, y=y))[0] == 200 for x in range(4) for y in range(4))
    assert tiles.Layer.drawn == drawn + 1 and tiles.Layer.redrawn == 0
    heat = live.api(f"heat?job={job.id}&config={'_'.join(map(str, best['config']))}")
    while "pending" in heat:
        heat = live.api(f"heat?job={job.id}&config={'_'.join(map(str, best['config']))}")
    kept = {tuple(c) for c in json.loads(Path(saved["file"]).read_text())["kept"]}
    assert (
        len(heat["dots"]) == sum(1 for c in kept for d in ((1, 0), (-1, 0), (0, 1), (0, -1)) if (c[0] + d[0], c[1] + d[1]) not in kept) * 16
    )
    out = live.api(f"events?job={job.id}")
    events = out["events"]
    assert out["end"] and len(events) == len(job.events()) and all(BANNED.search(e["label"]) is None for e in events if "label" in e)
    assert any(e.get("event") == "best" for e in events)
    path = Path(job.folder, jobs.EVENTS_FILE)
    third = sum(len(line) + 1 for line in path.read_bytes().split(b"\n")[:3])
    assert live.api(f"events?job={job.id}&at={third}")["events"] == events[3:]
    with path.open("ab") as f:
        f.write(b'{"event": "torn')
    assert live.api(f"events?job={job.id}&at={out['at']}") == {"events": [], "at": out["at"], "end": True}


def test_ui_strings_never_name_internal_steps():
    for f in ("index.html", "app.js"):
        text = (STATIC / f).read_text(encoding="utf-8")
        assert BANNED.search(text) is None, f
    labels = [label for _, label, _ in server.STAGES] + [label for _, label in server.PREVIEW_STEPS]
    assert all(BANNED.search(label) is None for label in labels)
    assert {server.stage_label(s)[0] for s in ("frontier-coarse-2", "refine-landmass-512", "landmass-full", "coarse")} <= set(labels)


RULE = "close the game or stop the server before you start, and keep it closed until Apply is done"
INVITES = re.compile(r"keep (running|playing)|catches up|(may|can) run while|while it plays|while (your|the) (world|game) runs", re.I)


def test_one_rule_said_the_same_way_everywhere_and_no_player_message_is_left():
    page = (STATIC / "index.html").read_text(encoding="utf-8")
    sections = dict(re.findall(r'<section id="screen-(\w+)"(.*?)</section>', page, re.S))
    for screen in ("home", "world", "area", "matches", "preview", "apply"):
        assert RULE in sections[screen], screen
    readme = (Path(__file__).parents[1] / "README.md").read_text(encoding="utf-8")
    code = page + (STATIC / "app.js").read_text(encoding="utf-8")
    assert RULE.lower() in " ".join(readme.split()).lower() and RULE in " ".join(cli.parser().format_help().split())
    assert INVITES.search(code + readme + cli.parser().format_help()) is None
    assert "running: `Close the game or stop the server first" in code
    assert not re.search(r"message for your players|players_note", code, re.I)
    assert "DIM0" in apply.MAP_TIP and "waypoints are kept" in apply.MAP_TIP and "VisualProspecting" in apply.MAP_TIP


def test_the_app_and_the_readme_give_the_same_measured_times():
    readme = (Path(__file__).parents[1] / "README.md").read_text(encoding="utf-8")
    page = (STATIC / "index.html").read_text(encoding="utf-8")
    in_readme = dict(re.findall(r"^\| (.+?) \| (.+?) \|$", readme, re.M))
    in_app = dict(re.findall(r"<tr><td>(.+?)</td><td>(.+?)</td></tr>", page))
    for step in ("Each start of the generator", "Map of your world", "Best-match search"):
        assert in_readme[step] == in_app[step], step


def test_the_page_states_the_memory_the_generator_refuses_below_and_asks_about_restarts_only_for_a_server():
    page = (STATIC / "index.html").read_text(encoding="utf-8")
    need = (generator.heap_bytes(generator.DEFAULT_HEAP) + generator.RAM_SPARE) // generator.GB
    assert f"The generator needs about {need} GB of free memory while it runs" in page
    script = (STATIC / "app.js").read_text(encoding="utf-8")
    assert "A.restarts = r.kind === 'server' && !remote;" in script and "(!A.restarts || A.ticks.auto_restart_off)" in script
    assert "$('#v-restart').parentElement.hidden = !!w && !(w.kind === 'server' && !w.remote);" in script


def test_the_world_card_shows_the_in_use_state_live(live, tmp_path, monkeypatch):
    live.open(fake_world(tmp_path / "server"))
    assert live.api("state")["world"]["in_use"]["state"] == "stopped"
    monkeypatch.setattr(server, "IN_USE_SECONDS", 0)
    monkeypatch.setattr(server.procs, "in_use", lambda root, world: server.procs.Usage("running", "java (pid 7) is running in it"))
    assert live.api("state")["world"]["in_use"] == {"state": "running", "detail": "java (pid 7) is running in it"}


def test_every_refusal_reaches_the_page_with_a_next_action(live, tmp_path):
    status, data = live.request("/api/world", {"path": str(tmp_path / "nowhere")})
    err = json.loads(data)["error"]
    assert status == 400 and err["reason"] and err["action"]
    text = "Refused: the world is in use: java (pid 7) is running in /srv. Next: Stop the server or close the game, then run this again."
    assert server.parse_error(text, "log") == {
        "reason": "the world is in use: java (pid 7) is running in /srv",
        "action": "Stop the server or close the game, then run this again",
    }
    boot = "job: failed (...)\nRefused: the generator stopped during its boot (exit code 0): [19:39:52] Starting Minecraft server\n"
    boot += "[19:39:52] **** FAILED TO BIND TO PORT!\n[19:39:52] Saving transformer cache. Next: See C:\\cache\\rwg2probe-console.log.\n"
    assert server.parse_error(boot, "log") == {
        "reason": "the generator stopped during its boot (exit code 0): [19:39:52] Starting Minecraft server\n"
        "[19:39:52] **** FAILED TO BIND TO PORT!\n[19:39:52] Saving transformer cache",
        "action": "See C:\\cache\\rwg2probe-console.log",
    }
    for embedded in ("Refused: FAILED", "Refused: x. Next: y. FAILED"):
        inner = boot.replace("[19:39:52] **** FAILED", embedded)
        assert inner.splitlines()[2].startswith(embedded)
        assert server.parse_error(inner, "log")["reason"].startswith("the generator stopped during its boot")
    assert server.parse_error("Failed: preliminary\nRefused: actual. Next: retry.\n", "log") == {"reason": "actual", "action": "retry"}
    after = server.parse_error(boot + "Traceback (most recent call last):\nRuntimeError: boom\n", "/logs/x.err")
    assert after["reason"] == "the step stopped unexpectedly (RuntimeError: boom)" and "/logs/x.err" in after["action"]
    crash = server.parse_error("Traceback (most recent call last):\n  File x\nKeyError: 'kept'\n", "/logs/x.err")
    assert "KeyError" in crash["reason"] and "/logs/x.err" in crash["action"]
    app = (STATIC / "app.js").read_text(encoding="utf-8")
    shown = re.findall(r"fail\(\{ reason: (.*?)\}\)", app)
    assert shown and all("action: " in s for s in shown)


def free_port():
    with socket.socket() as s:
        s.bind((server.LOCALHOST, 0))
        return s.getsockname()[1]


def start_ui(data, cache, port):
    env = os.environ | {"PYTHONPATH": str(HERE.parent)}
    return subprocess.Popen([sys.executable, str(HERE / "run_fake_ui.py"), str(data), str(cache), str(port)], cwd=HERE, env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True)  # fmt: skip


def http_json(port, token, path, body=None):
    c = http.client.HTTPConnection(server.LOCALHOST, port, timeout=WAIT)
    c.request("POST" if body is not None else "GET", "/api/" + path, json.dumps(body) if body is not None else None,
              {"Host": f"127.0.0.1:{port}", "X-Token": token, "Content-Type": "application/json"})  # fmt: skip
    r = c.getresponse()
    return json.loads(r.read())


def until(fn, what):
    deadline = time.monotonic() + WAIT
    while time.monotonic() < deadline:
        v = fn()
        if v:
            return v
        time.sleep(0.2)
    raise AssertionError(f"timed out waiting for {what}")


def test_console_single_instance_and_continue_after_the_console_closed(tmp_path):
    w = fake_world(tmp_path / "server")
    data, cache, port = tmp_path / "data", tmp_path / "cache", free_port()
    ui = start_ui(data, cache, port)
    lines = [ui.stdout.readline().rstrip("\n") for _ in range(2)]
    url = lines[0]
    assert re.fullmatch(rf"http://127\.0\.0\.1:{port}/\?t=[\w-]+", url) and lines[1] == server.KEEP_OPEN
    again = subprocess.run([sys.executable, "-m", "rwg2migrator", "ui", "--no-browser", "--data", str(data), "--cache", str(cache)],
                           capture_output=True, text=True, timeout=WAIT, env=os.environ | {"PYTHONPATH": str(HERE.parent)})  # fmt: skip
    assert again.returncode == 0 and again.stdout.strip() == url
    token = url.split("t=")[1]
    http_json(port, token, "world", {"path": w})
    until(lambda: http_json(port, token, "state")["world"]["scan"].get("chunks"), "the scan")
    http_json(port, token, "eula", {"accept": True})
    http_json(port, token, "area", {"teams": ["alpha"], "painted": [], "erased": [], "margin": 1, "save": True})
    http_json(port, token, "search", {"base": None, "banned": None, "tolerance": 1})
    job_id = until(lambda: http_json(port, token, "state")["runner"]["job"], "the search job")
    job = jobs.Job(str(data), job_id)
    until(lambda: any(e.get("event") == "batch" for e in job.events()), "a search batch")
    step = psutil.Process(job.data["pid"])
    stop_tree(ui, console_closed=True)
    ui.wait(WAIT)
    step.wait(WAIT)
    assert job.state() == jobs.RUNNING and ui.stdout.read() == "" and ui.stderr.read() == ""
    ui = start_ui(data, cache, port)
    url = ui.stdout.readline().strip()
    token = url.split("t=")[1]
    try:
        state = http_json(port, token, "state")
        assert next(j for j in state["jobs"] if j["id"] == job_id)["state"] == jobs.INTERRUPTED
        http_json(port, token, "world", {"path": w})
        until(lambda: http_json(port, token, "state")["world"]["scan"].get("chunks"), "the scan")
        http_json(port, token, "resume", {"job": job_id})
        until(lambda: not http_json(port, token, "state")["runner"]["running"], "the resumed search")
        assert job.state() == jobs.DONE and os.path.isfile(os.path.join(job.folder, *search.CANDIDATES))
    finally:
        stop_tree(ui)
        ui.wait(WAIT)
    assert ui.stdout.read() == server.KEEP_OPEN + "\n" and ui.stderr.read() == ""


def search_with_config(live, w, config_key):
    from fake_sampler import FakeSampler

    from rwg2migrator import appearance, scan

    saved = live.api("area", {"teams": ["alpha"], "painted": [], "erased": [], "margin": 1, "save": True})
    a = search.load(saved["file"])
    job = jobs.create(live.ui.data, "search", {"prefs": appearance.load_prefs(None), "threads": 1, "cache": live.ui.cache}, w)
    Path(job.folder, "area.json").write_text(json.dumps(a))
    sc = scan.scan(w, live.ui.cache, workers=1)
    inp = search.Inputs(sc, a, appearance.load_prefs(None), "2.9.0", 1234567890123, FakeSampler, SMALL, config=config_key)
    jobs.run(job, lambda j: search.run_job(j, inp))
    return job, "_".join(map(str, search.load(os.path.join(job.folder, *search.CANDIDATES))["candidates"][0]["config"]))


def candidate_tile(live, job, cfg):
    path = tile_path(live.api(f"layer?name=cand.{job.id}.{cfg}")["url"]).format(z=-1, x=0, y=0)
    return until(lambda: (r := live.request(path))[0] != 503 and r, "the candidate tile")


def heat_error(live, job, cfg):
    return until(lambda: json.loads(live.request(f"/api/heat?job={job.id}&config={cfg}")[1]).get("error"), "the heat refusal")


def test_each_world_gets_a_generator_on_its_own_seed_and_config_folder(live, tmp_path, monkeypatch):
    import fake_gen

    fake_gen.use(monkeypatch)
    boots = []
    make = fake_gen.FakeGens.sampler
    monkeypatch.setattr(
        fake_gen.FakeGens, "sampler", lambda self, seed, threads: boots.append((seed, self.config)) or make(self, seed, threads)
    )
    a = fake_world(tmp_path / "A")
    live.open(a)
    job_a, cfg_a = search_with_config(live, a, None)
    assert candidate_tile(live, job_a, cfg_a)[0] == 200
    b = fake_world(tmp_path / "B")
    Path(tmp_path, "B", "config").mkdir()
    Path(tmp_path, "B", "config", "rwg2.cfg").write_text('"expert / worldgen migration" {\n    I:"Biome Placement Offset X"=0\n}\n')
    key_b = jobs.tree_key(str(tmp_path / "B" / "config"))
    live.open(b)
    live.ui.opened.level["Data"]["RandomSeed"] = 999
    job_b, cfg_b = search_with_config(live, b, key_b)
    assert candidate_tile(live, job_b, cfg_b)[0] == 200
    assert boots == [(1234567890123, None), (999, key_b)]
    renders = list(Path(job_b.folder, "maps").glob(f"render-*/{jobs.MANIFEST_FILE}"))
    assert renders and all(json.loads(m.read_text())["inputs"]["generator"] == [999, key_b] for m in renders)
    status, body = live.request(f"/api/heat?job={job_a.id}&config={cfg_a}")
    assert status == 400 and "made for" in json.loads(body)["error"]["reason"]
    live.ui.maps.source.release()
    Path(tmp_path, "B", "config", "rwg2.cfg").write_text("changed\n")
    err = heat_error(live, job_b, cfg_b)
    assert "changed since this search ran" in err["reason"] and err["action"] == "Search again on the Best matches screen"
    assert boots == [(1234567890123, None), (999, key_b)]


def test_a_config_changed_after_opening_the_world_draws_the_next_search_and_refuses_the_older_one(live, tmp_path, monkeypatch):
    import fake_gen

    fake_gen.use(monkeypatch)
    w = fake_world(tmp_path / "S")
    cfg = Path(tmp_path, "S", "config")
    cfg.mkdir()
    (cfg / "rwg2.cfg").write_text('"expert / worldgen migration" {\n    I:"Biome Placement Offset X"=0\n}\n')
    live.open(w)
    old, cfg_old = search_with_config(live, w, jobs.tree_key(str(cfg)))
    assert candidate_tile(live, old, cfg_old)[0] == 200
    (cfg / "rwg2.cfg").write_text('"client" {\n    B:"some setting"=true\n}\n')
    new, cfg_new = search_with_config(live, w, jobs.tree_key(str(cfg)))
    assert candidate_tile(live, new, cfg_new)[0] == 200
    assert "dots" in live.api(f"heat?job={new.id}&config={cfg_new}")
    live.ui.maps.source.release()
    live.ui.maps.layers.clear()
    err = heat_error(live, old, cfg_old)
    assert "changed since this search ran" in err["reason"] and err["action"] == "Search again on the Best matches screen"


def test_a_map_generator_refusal_reaches_tiles_heat_and_the_page_and_is_not_booted_again_per_tile(live, tmp_path, monkeypatch):
    import fake_gen

    fake_gen.use(monkeypatch)
    w = fake_world(tmp_path / "R")
    live.open(w)
    job, cfg = search_with_config(live, w, None)
    boots = []
    problem = {"reason": "the generator needs 3.8 GB free; 1.2 GB is free now", "action": "Close GTNH, then try again"}

    def refuse(self, seed, threads):
        def make():
            boots.append(seed)
            raise server.Refusal(problem["reason"], problem["action"])

        return make

    monkeypatch.setattr(fake_gen.FakeGens, "sampler", refuse)
    tile = tile_path(live.api(f"layer?name=cand.{job.id}.{cfg}")["url"]).format(z=-1, x=0, y=0)
    answers = []
    for _ in range(12):
        answers.append(json.loads(live.request(tile)[1])["error"])
        time.sleep(0.2)
    assert problem in answers, answers
    status, body = live.request(f"/api/heat?job={job.id}&config={cfg}")
    assert status == 400 and json.loads(body)["error"] == problem
    assert live.api("state")["map_problem"] == problem and len(boots) == 1
    monkeypatch.setattr(server, "BOOT_RETRY", 0.0)
    until(lambda: live.request(tile) and len(boots) > 1, "a boot tried again once the wait is over")
    live.ui.maps.source.release()
    assert live.api("state")["map_problem"] is None


def test_the_readout_reads_the_map_under_the_pointer_not_todays_world(live, tmp_path, monkeypatch):
    import fake_gen
    import numpy as np
    from fake_sampler import NAMES, FakeSampler

    fake_gen.use(monkeypatch)
    w = fake_world(tmp_path / "S")
    live.open(w)
    job, cfg = search_with_config(live, w, None)
    cand = f"layer=cand.{job.id}.{cfg}"
    assert live.api(f"probe?x=8&z=8&{cand}&zoom=-2") == {"x": 8, "z": 8, "pending": True}
    assert candidate_tile(live, job, cfg)[0] == 200
    col = FakeSampler().columns(tuple(map(int, cfg.split("_"))), np.array([[8, 8]]))[0]
    want = {"x": 8, "z": 8, "ground": int(np.floor(col["h"])) + 1, "biome": NAMES[col["palette"]]}
    assert live.api(f"probe?x=8&z=8&{cand}&zoom=-1") == want != live.api("probe?x=8&z=8")
    assert live.api(f"probe?x=120&z=120&{cand}&zoom=-1") == live.api("probe?x=120&z=120")
    copy = make_world(str(tmp_path / "copy"), chunk_specs={(0, 0): {"ids": terrain(top=60)}})
    preview = jobs.create(live.ui.data, "preview", {"staging": copy}, w)
    smoothed = live.api(f"probe?x=8&z=8&layer=smooth.{preview.id}&zoom=0")
    assert smoothed["ground"] == live.api("probe?x=8&z=8")["ground"] - 4


def test_a_running_search_draws_only_the_matches_sampled_on_its_own_config(live, tmp_path, monkeypatch):
    from types import SimpleNamespace

    w = fake_world(tmp_path / "S")
    live.open(w)
    new = search_with_config(live, w, "new-key")[0]
    gen = SimpleNamespace(record={"kind": "sampler", "instance": "i1"}, connect=lambda: "the search's generator")
    monkeypatch.setattr(server.generator, "running", lambda cache: gen)
    monkeypatch.setattr(live.ui.runner, "state", lambda: {"running": True, "kind": "search", "job": new.id})
    monkeypatch.setattr(live.ui.runner, "busy", lambda: True)
    seed = server.seed_of(live.ui.opened)
    assert live.ui.maps.source.sampler([seed, "new-key"]).gen == "the search's generator"
    with pytest.raises(server.Refusal, match="uses the generator"):
        live.ui.maps.source.sampler([seed, "old-key"])


def test_the_copied_from_another_machine_choice_holds_until_apply_and_roll_back(live, tmp_path, monkeypatch):
    started = []
    monkeypatch.setattr(server.subprocess, "Popen", lambda cmd, **kw: started.append(cmd) or FakeProc())
    w = fake_world(tmp_path / "server")
    live.api("world", {"path": w, "remote": True})
    live.api("world", {"path": w})
    assert live.api("state")["world"]["remote"] is True
    again = Live(tmp_path / "data", tmp_path / "cache")
    try:
        again.api("world", {"path": w})
        assert again.api("state")["world"]["remote"] is True
    finally:
        again.close()
    preview = jobs.create(live.ui.data, "preview", {}, w)
    live.api("apply", {"preview": preview.id, "zip": True, "auto_restart_off": True})
    live.api("rollback", {"world": w, "auto_restart_off": True})
    assert {"--remote", "--zip"} <= set(started[0]) and "--remote" in started[1]
    live.api("world", {"path": w, "remote": False})
    live.api("apply", {"preview": preview.id, "zip": True, "auto_restart_off": True})
    assert live.api("state")["world"]["remote"] is False and not {"--remote", "--zip"} & set(started[2])


def test_the_config_folder_picked_on_the_world_screen_goes_to_every_step_that_reads_config(live, tmp_path, monkeypatch):
    started = []
    monkeypatch.setattr(server.subprocess, "Popen", lambda cmd, **kw: started.append(cmd) or FakeProc())
    monkeypatch.setattr(server, "cli_prefix", lambda: ["rwg2-migrator"])
    w = fake_world(tmp_path / "download")
    os.remove(tmp_path / "download" / "server.properties")
    live.open(w)
    card = live.api("state")["world"]
    assert (card["config"], card["config_from"], card["config_chosen"]) == (None, None, None)
    status, body = live.request("/api/config", {"world": w, "path": str(tmp_path)})
    assert status == 400 and "is not a config folder" in json.loads(body)["error"]["reason"]
    mine = tmp_path / "mine"
    mine.mkdir()
    (mine / "rwg2.cfg").write_text("")
    card = live.api("config", {"world": w, "path": str(mine)})
    assert (card["config"], card["config_from"], card["config_chosen"]) == (str(mine), world.CHOSEN, str(mine))
    again = Live(tmp_path / "data", tmp_path / "cache")
    try:
        assert again.open(w).place.config == str(mine)
        again.api("area", {"teams": ["alpha"], "painted": [], "erased": [], "margin": 1, "save": True})
        preview = jobs.create(again.ui.data, "preview", {}, w)
        for path, body in (
            ("search", {"tolerance": 1}),
            ("preview", {"search": jobs.create(again.ui.data, "search", {}, w).id, "pick": 1}),
            ("apply", {"preview": preview.id, "auto_restart_off": True}),
            ("verify", {}),
        ):
            again.api(path, body)
            assert cli.parser().parse_args(cli.joined(started[-1][1:])).config_dir == str(mine), path
    finally:
        again.close()
    (tmp_path / "download" / "config").mkdir()
    (tmp_path / "download" / "config" / "rwg2.cfg").write_text("")
    card = live.api("config", {"world": w, "path": ""})
    beside = str(tmp_path / "download" / "config")
    assert (card["config"], card["config_from"], card["config_chosen"]) == (beside, "found next to the world", None)
    live.api("verify", {})
    assert "--config-dir" not in started[-1]


def test_the_apply_checklist_stops_a_preview_made_on_a_config_folder_the_world_screen_no_longer_uses(live, tmp_path):
    w = fake_world(tmp_path / "download")
    os.remove(tmp_path / "download" / "server.properties")
    mine = tmp_path / "mine"
    mine.mkdir()
    (mine / "rwg2.cfg").write_text("")
    live.open(w)
    live.api("config", {"world": w, "path": str(mine)})
    place = dataclasses.asdict(world.with_config(world.locate(w), str(mine)))
    preview = jobs.create(live.ui.data, "preview", {"place": place}, w)
    r = live.api(f"apply/checklist?preview={preview.id}")
    assert (r["config"], r["config_shown"], r["config_changed"]) == (str(mine), str(mine), False)
    live.api("config", {"world": w, "path": ""})
    r = live.api(f"apply/checklist?preview={preview.id}")
    assert (r["config"], r["config_shown"], r["config_changed"]) == (str(mine), None, True)
    assert "!r.config_changed && stopped" in (STATIC / "app.js").read_text(encoding="utf-8")


def test_after_apply_names_the_file_with_the_lines_to_add_by_hand(live, tmp_path):
    w = migrate(tmp_path / "download")
    rec = json.loads(Path(w, apply.MARKER).read_text()) | {"backup": "b", "by_hand": {"file": "World" + apply.BY_HAND, "taint": False}}
    Path(w, apply.MARKER).write_text(json.dumps(rec))
    r = live.api(f"record?world={quote(w)}")
    assert r["by_hand"] == str(tmp_path / "download" / ("World" + apply.BY_HAND))
    app = (STATIC / "app.js").read_text(encoding="utf-8")
    assert "Apply wrote no config file. Before the server or game that will run this world starts" in app
    assert "so the 4 offset lines were not checked" in app and "No other problem found." in app


def migrate(root, state="applied"):
    w = make_world(str(root), generator="RWG2_CONTINENTS")
    rec = {"tool": apply.TOOL, "state": state, "date": "2026-11-04 10:00:00", "apply_job": "j1", "remote": False}
    Path(w, apply.MARKER).write_text(json.dumps(rec))
    return w


def test_a_migrated_world_opens_for_verify_the_rules_and_roll_back(live, tmp_path, monkeypatch):
    started = []
    monkeypatch.setattr(server.subprocess, "Popen", lambda cmd, **kw: started.append(cmd) or FakeProc())
    plain = make_world(str(tmp_path / "other"), generator="RWG2_CONTINENTS")
    status, body = live.request("/api/world", {"path": plain})
    assert status == 400 and "already uses RWG2" in json.loads(body)["error"]["reason"]
    w = migrate(tmp_path / "server")
    card = live.api("world", {"path": w})
    assert card["migrated"] == {"state": "applied", "date": "2026-11-04 10:00:00", "job": "j1"} and card["in_use"]["state"]
    status, body = live.request("/api/area", {"teams": [], "painted": [[0, 0]], "erased": [], "margin": 1})
    err = json.loads(body)["error"]
    assert status == 400 and "this tool migrated it" in err["reason"] and "Verify screen" in err["action"]
    live.api("verify", {})
    live.api("rollback", {"world": w, "auto_restart_off": True})
    assert [c[c.index(k) + 1] for c, k in zip(started, ("verify", "rollback"), strict=True)] == [w, w]


def test_a_world_cut_off_between_two_renames_names_the_way_back(live, tmp_path):
    w = fake_world(tmp_path / "server")
    job = jobs.create(live.ui.data, "apply", {"preview": "p"}, w)
    apply.Journal(os.path.join(job.folder, apply.JOURNAL)).note("rename1")
    os.rename(w, str(tmp_path / "aside"))
    status, body = live.request("/api/world", {"path": w})
    err = json.loads(body)["error"]
    assert status == 400 and "is not there now" in err["reason"] and err["action"].startswith("Click Continue for that Apply")
    job.update(state=jobs.DONE)
    Path(apply.rollback_journal(world.locate(w, exists=False))).parent.mkdir(parents=True)
    Path(apply.rollback_journal(world.locate(w, exists=False))).write_text('{"phase": "start"}\n')
    card = live.api("world", {"path": w})
    assert card["migrated"]["state"] == server.ROLLING_BACK and card["in_use"]["state"] == "stopped"
    status, body = live.request("/api/world", {"path": str(tmp_path / "nowhere")})
    assert status == 400 and "does not exist" in json.loads(body)["error"]["reason"]


def test_the_stop_button_stops_a_step_without_an_error_and_a_crash_still_shows_one(live, tmp_path, monkeypatch):
    monkeypatch.setenv(server.CLI_ENV, json.dumps([sys.executable, "-c", "import sys, time; time.sleep(60); sys.exit(3)"]))
    live.ui.runner.start("search", ["search", "W"])
    assert live.api("state")["runner"]["running"]
    live.api("stop", {})
    r = until(lambda: (st := live.api("state")["runner"]) and not st["running"] and st, "the stopped step")
    assert r["stopped"] and "error" not in r and r["code"] != 0
    monkeypatch.setenv(server.CLI_ENV, json.dumps([sys.executable, "-c", "import sys; sys.exit(3)"]))
    live.ui.runner.start("search", ["search", "W"])
    r = until(lambda: (st := live.api("state")["runner"]) and not st["running"] and st, "the crashed step")
    assert "stopped" not in r and "stopped unexpectedly" in r["error"]["reason"]


@pytest.mark.parametrize("missing", ["search", "pick"])
def test_a_preview_asked_for_without_its_match_is_a_plain_refusal(live, tmp_path, missing):
    w = fake_world(tmp_path / "server")
    live.open(w)
    live.shows = w
    body = {k: v for k, v in {"search": jobs.create(live.ui.data, "search", {}, w).id, "pick": 1}.items() if k != missing}
    status, out = live.request("/api/preview", body)
    err = json.loads(out)["error"]
    assert status == 400 and "which match" in err["reason"] and err["action"] == "Click Pick this one again on Best matches", err


def test_matches_of_another_area_are_flagged_and_a_preview_uses_the_area_chosen_now(live, tmp_path, monkeypatch):
    started = []
    monkeypatch.setattr(server.subprocess, "Popen", lambda cmd, **kw: started.append(cmd) or FakeProc())
    w = fake_world(tmp_path / "server")
    live.open(w)
    saved = live.api("area", {"teams": ["alpha"], "painted": [], "erased": [], "margin": 1, "save": True})
    job = jobs.create(live.ui.data, "search", {}, w)
    Path(job.folder, "area.json").write_text(Path(saved["file"]).read_text())
    assert live.api(f"job?id={job.id}")["area_changed"] is None
    live.api("area", {"teams": ["alpha"], "painted": [], "erased": [], "margin": 2, "save": True})
    assert live.api(f"job?id={job.id}")["area_changed"] == {"core": 3, "kept": 35}
    live.api("preview", {"search": job.id, "pick": 1})
    assert started[-1][started[-1].index("--area") + 1] == saved["file"]


def test_the_verify_map_shows_the_preview_the_world_before_apply_and_now_with_the_seam_heat(tmp_path, monkeypatch):
    from test_generate import RING, report, setup

    m = setup(tmp_path, monkeypatch)
    code, job = m.generate("--ring", RING)
    assert code == 0 and m.apply(job) == 0
    lv = Live(m.data, m.cache)
    try:
        status, body = lv.request("/api/aftermap")
        assert status == 400 and "no world is open" in json.loads(body)["error"]["reason"]
        lv.api("world", {"path": m.world})
        r = until(lambda: (v := lv.api("aftermap")) and "layers" in v and v, "the map after Apply")
        assert [v["label"] for v in r["layers"].values()] == ["PREVIEW", "BEFORE APPLY", "NOW"]
        assert (r["left"], r["right"]) == (f"planned.{job.id}", f"applied.{job.id}") and len(r["kept"]) > len(r["core"])
        after = report(job)["seams"]["after"]["kept_border"]
        steps = [d[2] for d in r["heat"][r["right"]]]
        assert len(steps) == after["pairs"] and max(steps) == after["max"], (len(steps), max(steps), after)
        assert lv.ui.maps.after["parts"].before.read == 0
        for name, layer in r["layers"].items():
            status, png = lv.request(tile_path(layer["url"]).format(z=-1, x=0, y=0))
            assert status == 200 and png.startswith(b"\x89PNG"), name
        kept = {tuple(c) for c in r["kept"]}
        out = next((x * 16 + 8, z * 16 + 8) for x in range(-2, 8) for z in range(-2, 8) if (x, z) not in kept and x >= 0 and z >= 0)
        planned = lv.api(f"probe?x={out[0]}&z={out[1]}&layer=planned.{job.id}&zoom=0")
        now = lv.api(f"probe?x={out[0]}&z={out[1]}&layer=applied.{job.id}&zoom=0")
        assert planned["biome"] and planned["ground"] is not None and now["ground"] is not None, (planned, now)
        status, body = lv.request("/api/layer?name=applied.another-preview")
        assert status == 400 and "another migrated world" in json.loads(body)["error"]["reason"]
        assert m.rollback() == 0
        tile = tile_path(r["layers"][r["right"]]["url"]).format(z=-2, x=0, y=0)
        for path in ("/api/aftermap", tile):
            status, body = lv.request(path)
            assert status == 400 and json.loads(body)["error"]["action"] == "Open it again on the World screen", (path, body)
    finally:
        lv.close()


def test_a_tile_url_of_a_world_no_longer_open_is_refused_never_answered_from_the_open_one(live, tmp_path):
    a, b = fake_world(tmp_path / "a"), fake_world(tmp_path / "b")
    live.open(a)
    tile_a = tile_path(live.api("layer?name=area")["url"]).format(z=-1, x=0, y=0)
    assert live.request(tile_a)[0] == 200
    live.open(b)
    tile_b = tile_path(live.api("layer?name=area")["url"]).format(z=-1, x=0, y=0)
    status, body = live.request(tile_a)
    assert tile_b != tile_a and status == 400 and json.loads(body)["error"]["action"] == "Reload this page"
    assert live.request(tile_b)[0] == 200 and live.request(tile_b.split("?")[0])[0] == 400
    live.shows = a
    status, body = live.request("/api/layer?name=area")
    assert status == 400 and "was opened since" in json.loads(body)["error"]["reason"]
    live.shows = None
    live.open(a)
    assert live.request(tile_a)[0] == 200


def test_every_report_it_names_the_issues_page():
    package = Path(server.__file__).parent.parent
    assert f'href="{ISSUES}"' in (STATIC / "index.html").read_text() and ISSUES in server.REPORT
    sources = [p.read_text(encoding="utf-8") for p in [*package.rglob("*.py"), STATIC / "app.js"]]
    assert not any("rwg2-migrator project" in s for s in sources)
    assert ISSUES in Path(package.parent, "README.md").read_text(encoding="utf-8")


def test_the_apply_checklist_counts_what_is_removed_from_the_open_world_by_todays_rule(live, tmp_path):
    w = fake_world(tmp_path / "server")
    live.open(w)
    body = {"teams": ["alpha"], "painted": [], "erased": [], "margin": 1}
    a = live.ui.area_of(body)[0]
    preview = jobs.create(live.ui.data, "preview", {"place": dataclasses.asdict(world.locate(w))}, w)
    stale = a | {"removed": {"yours": {"count": 99}, "graves": [], "other_teams_claims": {}}}
    Path(preview.folder, "area.json").write_text(json.dumps(stale), encoding="utf-8")
    removed = live.api(f"apply/checklist?preview={preview.id}")["removed"]
    assert removed == live.api("area", body)["removed"] and removed["yours"]["named"] == {"GregTech machines": 1}


def test_a_change_of_the_look_rules_gives_every_layer_a_new_tile_folder(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from rwg2migrator import appearance

    m = server.Maps(SimpleNamespace(cache=str(tmp_path)))
    before = m.folder("now", ["scan", "ref"])
    assert m.folder("now", ["scan", "ref"]) == before
    monkeypatch.setattr(appearance, "NAME_RULES", appearance.NAME_RULES[:-1])
    assert m.folder("now", ["scan", "ref"]) != before


def test_the_job_list_says_whether_an_apply_has_written_anything_yet(live, tmp_path):
    w = fake_world(tmp_path / "server")
    refused = jobs.create(live.ui.data, "apply", {"preview": "p1", "options": {}}, w)
    refused.update(state=jobs.FAILED)
    cut = jobs.create(live.ui.data, "apply", {"preview": "p2", "options": {}}, w)
    cut.update(state=jobs.INTERRUPTED)
    apply.Journal(os.path.join(cut.folder, apply.JOURNAL)).note("write_configs")
    preview = jobs.create(live.ui.data, "preview", {}, w)
    preview.update(state=jobs.DONE, applied=cut.id)
    views = {j["id"]: j for j in live.api("state")["jobs"]}
    assert (views[refused.id]["preview"], views[refused.id]["open"]) == ("p1", False)
    assert (views[cut.id]["preview"], views[cut.id]["open"]) == ("p2", True)
    assert views[preview.id]["applied"] == cut.id and "open" not in views[preview.id]


@pytest.mark.skipif(os.name == "nt", reason="file modes are POSIX")
def test_the_file_that_holds_the_address_with_its_token_is_never_readable_by_others(tmp_path, monkeypatch):
    modes, real = [], server.write_atomic
    monkeypatch.setattr(server, "write_atomic", lambda path, data: real(path, data) or modes.append(os.stat(path).st_mode & 0o777))
    for _ in range(2):
        server.announce(str(tmp_path), "http://127.0.0.1:1/?t=secret")
    assert modes == [0o600, 0o600] and os.stat(tmp_path / server.UI_FILE).st_mode & 0o777 == 0o600
    assert server.running_ui(str(tmp_path)) == "http://127.0.0.1:1/?t=secret"
