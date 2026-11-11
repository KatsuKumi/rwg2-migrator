"""ui --replay: a finished search and its preview play back on the page without running. Their events show by recorded time, the
recorded jobs stay hidden until played, a match other than the recorded one is refused, and nothing boots a generator."""

import json
import logging
import time

import pytest
from conftest import add_migration_parts, stop_tree
from test_web import WAIT, Live, fake_world, free_port, http_json, start_ui, tile_path, until

from rwg2migrator import generator, jobs, provision
from rwg2migrator.web import server

PLAY_S = 3.0


@pytest.fixture(scope="module")
def recorded(tmp_path_factory):
    """A fake world with a search and a preview of its match #1, both run through the UI with the fake generator."""
    base = tmp_path_factory.mktemp("recorded")
    w = fake_world(base / "server")
    add_migration_parts(str(base / "server"))
    data, cache, port = base / "data", base / "cache", free_port()
    ui = start_ui(data, cache, port)
    token = ui.stdout.readline().strip().split("t=")[1]
    try:
        http_json(port, token, "world", {"path": w})
        until(lambda: http_json(port, token, "state")["world"]["scan"].get("chunks"), "the scan")
        http_json(port, token, "eula", {"accept": True})
        http_json(port, token, "area", {"teams": ["alpha"], "painted": [], "erased": [], "margin": 1, "save": True})
        http_json(port, token, "search", {"base": None, "banned": None, "tolerance": 1})
        searched = until(lambda: http_json(port, token, "state")["runner"]["job"], "the search job")
        until(lambda: not http_json(port, token, "state")["runner"]["running"], "the search")
        http_json(port, token, "preview", {"search": searched, "pick": 1})
        runner = until(lambda: (r := http_json(port, token, "state")["runner"])["kind"] == "preview" and r["job"] and r, "the preview job")
        until(lambda: not http_json(port, token, "state")["runner"]["running"], "the preview")
    finally:
        stop_tree(ui)
        ui.wait()
    assert jobs.Job(str(data), searched).state() == jobs.DONE and jobs.Job(str(data), runner["job"]).state() == jobs.DONE
    return {"world": w, "data": data, "search": searched, "preview": runner["job"]}


def play_speed(data, job_id):
    times = [e["t"] for e in jobs.Job(str(data), job_id).events()]
    return (times[-1] - times[0]) / PLAY_S


def test_a_replayed_search_shows_its_events_by_recorded_time_and_its_matches_at_the_end(recorded, tmp_path):
    live = Live(recorded["data"], tmp_path / "bare-cache", (recorded["search"], play_speed(recorded["data"], recorded["search"])))
    listed = {j["id"] for j in live.api("state")["jobs"]}
    assert recorded["search"] not in listed and recorded["preview"] not in listed
    live.api("world", {"path": recorded["world"]})
    until(lambda: live.api("state")["world"]["scan"].get("chunks"), "the scan")
    live.api("area", {"teams": ["alpha"], "painted": [], "erased": [], "margin": 1, "save": True})
    started = live.api("search", {"base": None, "banned": None, "tolerance": 1})
    assert started["running"] and started["job"] == recorded["search"]
    early = live.api(f"events?job={recorded['search']}&at=0")
    every = jobs.Job(str(recorded["data"]), recorded["search"]).events()
    assert not early["end"] and 0 < len(early["events"]) < len(every)
    assert live.api(f"job?id={recorded['search']}")["candidates"] is None
    until(lambda: not live.api("state")["runner"]["running"], "the replay to end")
    late = live.api(f"events?job={recorded['search']}&at={early['at']}")
    assert late["end"] and len(early["events"]) + len(late["events"]) == len(every)
    assert live.api(f"job?id={recorded['search']}")["candidates"]["candidates"]
    assert recorded["search"] in {j["id"] for j in live.api("state")["jobs"]}


def test_a_replayed_preview_plays_only_the_recorded_match_and_no_generator_starts(recorded, tmp_path):
    live = Live(recorded["data"], tmp_path / "bare-cache", (recorded["search"], play_speed(recorded["data"], recorded["preview"])))
    live.api("world", {"path": recorded["world"]})
    until(lambda: live.api("state")["world"]["scan"].get("chunks"), "the scan")
    status, body = live.request("/api/preview", {"search": recorded["search"], "pick": 2})
    assert status == 400 and "the preview of match #1" in json.loads(body)["error"]["reason"]
    assert live.api("preview", {"search": recorded["search"], "pick": 1})["job"] == recorded["preview"]
    first = [s["done"] for s in live.api(f"job?id={recorded['preview']}")["steps"]]
    until(lambda: not live.api("state")["runner"]["running"], "the replay to end")
    last = [s["done"] for s in live.api(f"job?id={recorded['preview']}")["steps"]]
    assert not all(first) and all(last)
    cfg = "_".join(map(str, live.api(f"job?id={recorded['preview']}")["args"]["config"]))
    tile = tile_path(live.api(f"layer?name=cand.{recorded['search']}.{cfg}")["url"]).format(z=0, x=0, y=0)
    status, body = live.request(tile)
    deadline = time.monotonic() + WAIT
    while status == 503 and "replay" not in body.decode() and time.monotonic() < deadline:
        status, body = live.request(tile)
    assert status == 503 and json.loads(body)["error"] == {"reason": server.REPLAY_ONLY, "action": server.FILLS_IN}
    assert generator.running(str(tmp_path / "bare-cache")) is None and live.api("state")["map_problem"] is None


def test_a_generator_set_up_on_another_machine_is_tried_once_and_quietly(recorded, tmp_path):
    """A bundle's cache names the PC's Java: the boot fails here, the maps stay hatched with no error, and no retry follows soon."""
    cache = tmp_path / "pc-cache"
    provision.claim_cache(str(cache))
    generator.save_settings(str(cache), {"release": "2.9.0", "java": "C:/Program Files/nowhere/java.exe", "heap": "3G"})
    live = Live(recorded["data"], cache, (recorded["search"], play_speed(recorded["data"], recorded["preview"])))
    live.api("world", {"path": recorded["world"]})
    until(lambda: live.api("state")["world"]["scan"].get("chunks"), "the scan")
    live.api("preview", {"search": recorded["search"], "pick": 1})
    until(lambda: not live.api("state")["runner"]["running"], "the replay to end")
    cfg = "_".join(map(str, live.api(f"job?id={recorded['preview']}")["args"]["config"]))
    tile = tile_path(live.api(f"layer?name=cand.{recorded['search']}.{cfg}")["url"]).format(z=0, x=0, y=0)
    until(lambda: server.REPLAY_ONLY in live.request(tile)[1].decode(), "the quiet refusal")
    first = live.ui.maps.source.problem["at"]
    assert [live.request(tile)[0] for _ in range(4)] == [503] * 4 and live.ui.maps.source.problem["at"] == first
    assert live.api("state")["map_problem"] is None and live.api("state")["map_status"] == "" and generator.running(str(cache)) is None


def test_opening_the_browser_is_logged_without_the_token(monkeypatch, caplog):
    monkeypatch.setattr(server.webbrowser, "open", lambda url: False)
    with caplog.at_level(logging.INFO, logger="rwg2migrator.ui"):
        assert server.open_browser("http://127.0.0.1:5/?t=secret") is False
    assert "webbrowser.open(http://127.0.0.1:5/) returned False" in caplog.text and "secret" not in caplog.text
